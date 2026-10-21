import type { Matrix4 } from "../../maths/Matrix4";
import {
	type DrawSubmission,
	type FrameContext,
} from "../../pipeline/types";
import type {
	RenderGraphPhysicalBinding,
	RenderGraphResourceDescriptor,
} from "../../rendergraph/types";
import {
	renderGraphPhysicalResourceId,
	renderGraphResourceId,
} from "../../rendergraph/types";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../constants";

import {
	WEBGL_SHADOW_ATLAS_COLUMNS,
	WEBGL_SHADOW_ATLAS_ROWS,
} from "./constants";
import type { WebGLGeometryRegistry } from "./WebGLGeometryRegistry";
import type { WebGLAnimationPayloadPool } from "./WebGLAnimationPayloadPool";
import type { WebGLLightState, WebGLShadowData } from "./WebGLLightCollector";
import type { WebGLProgramCompiler, WebGLProgramWarmupHandle } from "./WebGLProgramCompiler";
import type {
	WebGLProgramWarmupContributor,
	WebGLProgramWarmupRequest,
	WebGLProgramWarmupTask,
} from "./WebGLWarmupCoordinator";
import {
	WebGLShadowRasterPass,
	type WebGLShadowRasterPlan,
	type WebGLShadowRasterSlice,
} from "./WebGLShadowRasterPass";
import type { WebGLFrameGraphResourceCatalogSnapshot } from "./rendergraph/types";

function getMaxShadowSize(values: WebGLShadowData[]): number {
	let maxSize = 0;
	for (const shadow of values) {
		if (!shadow.enabled || !shadow.viewProjectionMatrix) {
			continue;
		}
		maxSize = Math.max(maxSize, shadow.shadowMapBaseSize | 0);
	}
	return maxSize;
}

type WebGLShadowRuntimePhase =
	| "idle"
	| "begun"
	| "prepared"
	| "rendered"
	| "destroyed";

type MutableRasterSlice = {
	-readonly [Key in keyof WebGLShadowRasterSlice]: WebGLShadowRasterSlice[Key];
};

interface MutableRasterPlan {
	atlasTileSize: number;
	atlasWidth: number;
	atlasHeight: number;
	slices: MutableRasterSlice[];
	sliceCount: number;
	casterSubmissions: readonly DrawSubmission[];
	transmitterSubmissions: readonly DrawSubmission[];
	baselineFramebuffer: WebGLFramebuffer | null;
	baselineViewportWidth: number;
	baselineViewportHeight: number;
}

/** @internal Readonly, identity-stable shadow inputs for WebGL scene consumers. */
export interface WebGLShadowSamplingState {
	readonly enabled: boolean;
	readonly atlasTexture: WebGLTexture | null;
	readonly transmittanceTexture: WebGLTexture | null;
	readonly atlasTileSize: number;
	readonly transmittanceAvailable: boolean;
}

interface MutableShadowSamplingState {
	enabled: boolean;
	atlasTexture: WebGLTexture | null;
	transmittanceTexture: WebGLTexture | null;
	atlasTileSize: number;
	transmittanceAvailable: boolean;
}

/** @internal Device and frame-baseline dependencies for one WebGL shadow runtime. */
export interface WebGLShadowRuntimeHost {
	readonly gl: WebGL2RenderingContext;
	readonly programCompiler: WebGLProgramCompiler;
	readonly geometry: WebGLGeometryRegistry;
	readonly animationPayloads?: WebGLAnimationPayloadPool;
	readonly maxTextureSize: number;
	getSceneFramebuffer(): WebGLFramebuffer | null;
	getWidth(): number;
	getHeight(): number;
}

/** Owns all frame-aware WebGL shadow planning and sampling state. */
export class WebGLShadowRuntime implements WebGLProgramWarmupContributor {
	private readonly _host: WebGLShadowRuntimeHost;
	private readonly _rasterPass: WebGLShadowRasterPass;
	private readonly _slicePool: MutableRasterSlice[] = [];
	private readonly _plan: MutableRasterPlan = {
		atlasTileSize: 0,
		atlasWidth: 0,
		atlasHeight: 0,
		slices: [],
		sliceCount: 0,
		casterSubmissions: [],
		transmitterSubmissions: [],
		baselineFramebuffer: null,
		baselineViewportWidth: 1,
		baselineViewportHeight: 1,
	};
	private readonly _samplingState: MutableShadowSamplingState = {
		enabled: false,
		atlasTexture: null,
		transmittanceTexture: null,
		atlasTileSize: 0,
		transmittanceAvailable: false,
	};
	private _phase: WebGLShadowRuntimePhase = "idle";
	private _context: FrameContext | null = null;
	private _hasPreparedResources = false;

	constructor(host: WebGLShadowRuntimeHost) {
		this._host = host;
		this._rasterPass = new WebGLShadowRasterPass({
			gl: host.gl,
			programCompiler: host.programCompiler,
			geometry: host.geometry,
			animationPayloads: host.animationPayloads,
			maxTextureSize: host.maxTextureSize,
		});
	}

	public warmupPrograms(): WebGLProgramWarmupHandle[] {
		return this._rasterPass.warmupPrograms();
	}

	public collectWarmupTasks(
		request: WebGLProgramWarmupRequest,
	): readonly WebGLProgramWarmupTask[] {
		return request.plan.enableShadows
			? [
					{
						label: "WebGLShadowPrograms",
						priority: "optional",
						run: () => this.warmupPrograms(),
					},
				]
			: [];
	}

	/** Synchronizes metadata before WebGL light collection starts. */
	public beginFrame(context: FrameContext): void {
		this._assertCanBegin();
		this._phase = "begun";
		this._context = context;
		this._clearPreparedFrameState();
	}

	/** Builds the reusable plan and allocates predictable shadow targets. */
	public prepareFrame(context: FrameContext, lightState: WebGLLightState): void {
		this._assertPhaseAndContext("begun", context, "prepareFrame");
		this._resetPlan(context);
		const tileSize = Math.max(
			getMaxShadowSize(lightState.directionalShadows),
			getMaxShadowSize(lightState.spotShadows),
		);
		const hasPotentialCasters =
			context.scene.shadowCasterSubmissions.length > 0 ||
			context.scene.shadowTransmitterSubmissions.length > 0;
		if (context.shadowPlan?.hasRasterWork !== true || tileSize <= 0 || !hasPotentialCasters) {
			this._setLightAtlasTileSize(lightState, 0);
			this._phase = "prepared";
			return;
		}

		this._plan.atlasTileSize = tileSize;
		this._plan.atlasWidth = tileSize * WEBGL_SHADOW_ATLAS_COLUMNS;
		this._plan.atlasHeight = tileSize * WEBGL_SHADOW_ATLAS_ROWS;
		this._setLightAtlasTileSize(lightState, tileSize);
		this._buildRasterSlices(lightState, tileSize);
		try {
			const prepared = this._rasterPass.prepare(this._plan as WebGLShadowRasterPlan);
			this._hasPreparedResources = !!prepared.atlasTexture && !!prepared.transmittanceTexture;
			this._samplingState.atlasTexture = prepared.atlasTexture;
			this._samplingState.transmittanceTexture = prepared.transmittanceTexture;
			this._samplingState.atlasTileSize = prepared.atlasTileSize;
			this._samplingState.enabled =
				this._hasPreparedResources && prepared.depthProgramAvailable;
			this._samplingState.transmittanceAvailable =
				this._samplingState.enabled &&
				prepared.transmittanceProgramAvailable &&
				!!prepared.transmittanceTexture;
			this._phase = "prepared";
		} catch (error) {
			this.abortFrame();
			throw error;
		}
	}

	/** Executes the prepared shadow raster plan once. */
	public renderPreparedFrame(context: FrameContext): void {
		this._assertPhaseAndContext("prepared", context, "renderPreparedFrame");
		try {
			if (this._hasPreparedResources) {
				this._rasterPass.render(this._plan as WebGLShadowRasterPlan);
			}
			this._phase = "rendered";
		} catch (error) {
			this.abortFrame();
			throw error;
		}
	}

	/** Clears pending frame identity while retaining reusable native resources. */
	public abortFrame(): void {
		if (this._phase === "destroyed") return;
		this._phase = "idle";
		this._context = null;
		this._clearPreparedFrameState();
	}

	public getSamplingState(): WebGLShadowSamplingState {
		return this._samplingState;
	}

	/** Describes only prepared shadow resources and never exposes native handles. */
	public describeGraphResources(): WebGLFrameGraphResourceCatalogSnapshot {
		if (!this._hasPreparedResources) {
			return { resources: [], bindings: [] };
		}
		const tileSize = this._samplingState.atlasTileSize;
		const resources: RenderGraphResourceDescriptor[] = [];
		const bindings: RenderGraphPhysicalBinding[] = [];
		this._addGraphTexture(
			resources,
			bindings,
			"shadow:atlas",
			"depth",
			this._plan.atlasWidth,
			this._plan.atlasHeight,
		);
		this._addGraphTexture(
			resources,
			bindings,
			"shadow:transmittance",
			"rgba8unorm",
			this._plan.atlasWidth,
			this._plan.atlasHeight,
		);
		return {
			resources: Object.freeze(resources),
			bindings: Object.freeze(bindings),
		};
	}

	public destroy(): void {
		if (this._phase === "destroyed") return;
		this._rasterPass.destroy();
		this._clearPreparedFrameState();
		this._context = null;
		this._phase = "destroyed";
	}

	private _resetPlan(context: FrameContext): void {
		this._plan.atlasTileSize = 0;
		this._plan.atlasWidth = 0;
		this._plan.atlasHeight = 0;
		this._plan.sliceCount = 0;
		this._plan.casterSubmissions = context.scene.shadowCasterSubmissions;
		this._plan.transmitterSubmissions =
			context.scene.shadowTransmitterSubmissions;
		this._plan.baselineFramebuffer = this._host.getSceneFramebuffer();
		this._plan.baselineViewportWidth = this._host.getWidth();
		this._plan.baselineViewportHeight = this._host.getHeight();
	}

	private _buildRasterSlices(lightState: WebGLLightState, tileSize: number): void {
		const directionalCount = Math.min(
			MAX_DIRECTIONAL_LIGHTS,
			lightState.directionalShadows.length,
		);
		for (let index = 0; index < directionalCount; index++) {
			const shadow = lightState.directionalShadows[index];
			const cascadeCount =
				shadow?.enabled && shadow.strategyType === "csm" && shadow.cascadeCount > 1
					? Math.max(1, Math.min(4, shadow.cascadeCount | 0))
					: 1;
			for (let cascadeIndex = 0; cascadeIndex < cascadeCount; cascadeIndex++) {
				this._appendRasterSlice(
					"directional",
					index,
					cascadeIndex,
					shadow,
					index,
					tileSize,
				);
			}
		}
		const spotCount = Math.min(MAX_SPOT_LIGHTS, lightState.spotShadows.length);
		for (let index = 0; index < spotCount; index++) {
			this._appendRasterSlice(
				"spot",
				index,
				0,
				lightState.spotShadows[index],
				MAX_DIRECTIONAL_LIGHTS + index,
				tileSize,
			);
		}
	}

	private _appendRasterSlice(
		kind: WebGLShadowRasterSlice["kind"],
		lightIndex: number,
		cascadeIndex: number,
		shadow: WebGLShadowData | undefined,
		tileIndex: number,
		atlasTileSize: number,
	): void {
		const resolved = this._resolveShadowSlice(shadow, tileIndex, cascadeIndex, atlasTileSize);
		if (!resolved) return;
		const planIndex = this._plan.sliceCount++;
		let slice = this._slicePool[planIndex];
		if (!slice) {
			slice = {
				kind,
				lightIndex,
				cascadeIndex,
				viewportX: 0,
				viewportY: 0,
				viewportWidth: 1,
				viewportHeight: 1,
				viewProjectionMatrix: resolved.viewProjectionMatrix,
			};
			this._slicePool.push(slice);
			this._plan.slices.push(slice);
		}
		slice.kind = kind;
		slice.lightIndex = lightIndex;
		slice.cascadeIndex = cascadeIndex;
		slice.viewportX = resolved.viewportX;
		slice.viewportY = resolved.viewportY;
		slice.viewportWidth = resolved.shadowSize;
		slice.viewportHeight = resolved.shadowSize;
		slice.viewProjectionMatrix = resolved.viewProjectionMatrix;
	}

	private _resolveShadowSlice(
		shadow: WebGLShadowData | undefined,
		tileIndex: number,
		cascadeIndex: number,
		atlasTileSize: number,
	): {
		viewProjectionMatrix: Matrix4;
		shadowSize: number;
		viewportX: number;
		viewportY: number;
	} | null {
		if (!shadow?.enabled) return null;
		const isCSM = shadow.strategyType === "csm" && shadow.cascadeCount > 1;
		const cascadeCount = isCSM ? Math.max(1, Math.min(4, shadow.cascadeCount | 0)) : 1;
		const resolvedCascade = Math.max(0, Math.min(cascadeCount - 1, cascadeIndex | 0));
		const viewProjectionMatrix = isCSM
			? shadow.cascadeViewProjectionMatrices[resolvedCascade]
			: shadow.viewProjectionMatrix;
		if (!viewProjectionMatrix) return null;
		const localTileSpan = isCSM ? 2 : 1;
		const subTileSize = Math.max(1, Math.floor(atlasTileSize / localTileSpan));
		const shadowSize = Math.max(1, Math.min(shadow.shadowMapSize | 0, subTileSize));
		const split = shadow.cascadeSplits[resolvedCascade] ?? [0, 0, 0, 0];
		const localTileX = isCSM ? Math.max(0, Math.min(1, Math.floor(split[2] + 0.5))) : 0;
		const localTileY = isCSM ? Math.max(0, Math.min(1, Math.floor(split[3] + 0.5))) : 0;
		const tileX = tileIndex % WEBGL_SHADOW_ATLAS_COLUMNS;
		const tileY = Math.floor(tileIndex / WEBGL_SHADOW_ATLAS_COLUMNS);
		return {
			viewProjectionMatrix,
			shadowSize,
			viewportX: tileX * atlasTileSize + localTileX * subTileSize,
			viewportY: tileY * atlasTileSize + localTileY * subTileSize,
		};
	}

	private _setLightAtlasTileSize(lightState: WebGLLightState, size: number): void {
		for (const shadow of lightState.directionalShadows) shadow.atlasTileSize = size;
		for (const shadow of lightState.spotShadows) shadow.atlasTileSize = size;
	}

	private _addGraphTexture(
		resources: RenderGraphResourceDescriptor[],
		bindings: RenderGraphPhysicalBinding[],
		id: string,
		format: string,
		width: number,
		height: number,
	): void {
		resources.push({
			id: renderGraphResourceId(id),
			origin: "imported",
			kind: "texture",
			residency: "frame",
			initialContent: "unknown",
			format,
			width,
			height,
			depthOrArrayLayers: 1,
			dimension: "2d",
			sampleCount: 1,
			mipLevelCount: 1,
		});
		bindings.push({
			resourceId: renderGraphResourceId(id),
			physicalId: renderGraphPhysicalResourceId(`webgl:slot:${id}`),
			kind: "texture",
		});
	}

	private _clearPreparedFrameState(): void {
		this._hasPreparedResources = false;
		this._plan.atlasTileSize = 0;
		this._plan.sliceCount = 0;
		this._plan.casterSubmissions = [];
		this._plan.transmitterSubmissions = [];
		this._samplingState.enabled = false;
		this._samplingState.atlasTexture = null;
		this._samplingState.transmittanceTexture = null;
		this._samplingState.atlasTileSize = 0;
		this._samplingState.transmittanceAvailable = false;
	}

	private _assertCanBegin(): void {
		if (this._phase === "destroyed") {
			throw new Error("WebGLShadowRuntime.beginFrame called after destroy");
		}
		if (this._phase === "begun" || this._phase === "prepared") {
			throw new Error(
				`WebGLShadowRuntime.beginFrame called while phase=${this._phase}; abort the pending frame first`,
			);
		}
	}

	private _assertPhaseAndContext(
		expected: WebGLShadowRuntimePhase,
		context: FrameContext,
		operation: string,
	): void {
		if (this._phase !== expected) {
			throw new Error(
				`WebGLShadowRuntime.${operation} expected phase=${expected}, actual=${this._phase}`,
			);
		}
		if (this._context !== context) {
			throw new Error(
				`WebGLShadowRuntime.${operation} received a frame context different from beginFrame`,
			);
		}
	}
}
