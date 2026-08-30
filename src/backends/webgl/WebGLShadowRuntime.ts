import { Logger } from "../../foundation/Logger";
import type { Matrix4 } from "../../maths/Matrix4";
import {
	createParticleShadowVolumeGrid,
	hasParticleShadowCastingBatches,
	injectParticleBatchIntoShadowVolume,
} from "../../pipeline/ParticleShadowVolume";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type DrawPacket,
	type FrameContext,
} from "../../pipeline/types";
import type { ParticleRenderBatch } from "../../particles/ParticleRenderBatch";
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
	WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH,
	WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES,
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

const PARTICLE_ATLAS_WIDTH =
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH *
	WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS;
const PARTICLE_ATLAS_ROWS = Math.ceil(
	(WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH *
		WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES) /
		WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS,
);
const PARTICLE_ATLAS_HEIGHT =
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT * PARTICLE_ATLAS_ROWS;

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
	casterPackets: readonly DrawPacket[];
	transmitterPackets: readonly DrawPacket[];
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
	readonly particleVolumeTexture: WebGLTexture | null;
	readonly particleVolumeActiveSliceCount: number;
	readonly particleVolumeAtlasSize: Readonly<Float32Array>;
	readonly particleVolumeGridSize: Readonly<Float32Array>;
	readonly particleVolumeSliceParams: Readonly<Float32Array>;
}

interface MutableShadowSamplingState {
	enabled: boolean;
	atlasTexture: WebGLTexture | null;
	transmittanceTexture: WebGLTexture | null;
	atlasTileSize: number;
	transmittanceAvailable: boolean;
	particleVolumeTexture: WebGLTexture | null;
	particleVolumeActiveSliceCount: number;
	particleVolumeAtlasSize: Float32Array;
	particleVolumeGridSize: Float32Array;
	particleVolumeSliceParams: Float32Array;
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
		casterPackets: [],
		transmitterPackets: [],
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
		particleVolumeTexture: null,
		particleVolumeActiveSliceCount: 0,
		particleVolumeAtlasSize: new Float32Array(2),
		particleVolumeGridSize: new Float32Array(4),
		particleVolumeSliceParams: new Float32Array(WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES * 4),
	};
	private _phase: WebGLShadowRuntimePhase = "idle";
	private _context: FrameContext | null = null;
	private _lightState: WebGLLightState | null = null;
	private _hasPreparedResources = false;
	private _particlePreparedThisFrame = false;
	private _particleVolumeTexture: WebGLTexture | null = null;
	private _particleVolumePixels = new Float32Array(0);

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
		this._lightState = null;
		this._clearPreparedFrameState();
	}

	/** Builds the reusable plan and allocates predictable shadow targets. */
	public prepareFrame(context: FrameContext, lightState: WebGLLightState): void {
		this._assertPhaseAndContext("begun", context, "prepareFrame");
		this._lightState = lightState;
		this._resetPlan(context);
		const tileSize = Math.max(
			getMaxShadowSize(lightState.directionalShadows),
			getMaxShadowSize(lightState.spotShadows),
		);
		const hasPotentialCasters =
			context.scene.shadowCasterPackets.length > 0 ||
			context.scene.shadowTransmitterPackets.length > 0 ||
			context.scene.particleSystems.length > 0;
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
			if (context.scene.particleSystems.length > 0) {
				this._prepareParticleVolumeTexture();
			}
			this._phase = "prepared";
		} catch (error) {
			this.abortFrame();
			throw error;
		}
	}

	/** Uploads particle density and executes the prepared raster plan once. */
	public renderPreparedFrame(context: FrameContext): void {
		this._assertPhaseAndContext("prepared", context, "renderPreparedFrame");
		try {
			if (this._hasPreparedResources) {
				this._updateParticleVolumes(context);
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
		this._lightState = null;
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
		if (this._particlePreparedThisFrame) {
			this._addGraphTexture(
				resources,
				bindings,
				"shadow:particle-volume",
				"r32float",
				PARTICLE_ATLAS_WIDTH,
				PARTICLE_ATLAS_HEIGHT,
			);
		}
		return {
			resources: Object.freeze(resources),
			bindings: Object.freeze(bindings),
		};
	}

	public destroy(): void {
		if (this._phase === "destroyed") return;
		this._rasterPass.destroy();
		if (this._particleVolumeTexture) {
			this._host.gl.deleteTexture(this._particleVolumeTexture);
			this._particleVolumeTexture = null;
		}
		this._particleVolumePixels = new Float32Array(0);
		this._clearPreparedFrameState();
		this._context = null;
		this._lightState = null;
		this._phase = "destroyed";
	}

	private _resetPlan(context: FrameContext): void {
		this._plan.atlasTileSize = 0;
		this._plan.atlasWidth = 0;
		this._plan.atlasHeight = 0;
		this._plan.sliceCount = 0;
		this._plan.casterPackets = context.scene.shadowCasterPackets;
		this._plan.transmitterPackets = context.scene.shadowTransmitterPackets;
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

	private _prepareParticleVolumeTexture(): void {
		if (
			PARTICLE_ATLAS_WIDTH > this._host.maxTextureSize ||
			PARTICLE_ATLAS_HEIGHT > this._host.maxTextureSize
		) {
			this._warn(
				"webgl-particle-shadow-volume-atlas-limit",
				`WebGL particle shadow volume atlas ${PARTICLE_ATLAS_WIDTH}x` +
					`${PARTICLE_ATLAS_HEIGHT} exceeds MAX_TEXTURE_SIZE=` +
					`${this._host.maxTextureSize}; disabling particle volume shadows ` +
					"for this frame.",
			);
			return;
		}
		if (!this._particleVolumeTexture) {
			const gl = this._host.gl;
			const texture = gl.createTexture();
			if (!texture) {
				const message = "Failed to create WebGL particle shadow volume atlas";
				this._warn("webgl-particle-shadow-volume-create-failed", message);
				throw new Error(`[webgl-particle-shadow-volume-create-failed] ${message}`);
			}
			this._particleVolumeTexture = texture;
			try {
				gl.bindTexture(gl.TEXTURE_2D, texture);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
				gl.texImage2D(
					gl.TEXTURE_2D,
					0,
					gl.R32F,
					PARTICLE_ATLAS_WIDTH,
					PARTICLE_ATLAS_HEIGHT,
					0,
					gl.RED,
					gl.FLOAT,
					null,
				);
			} catch (error) {
				gl.deleteTexture(texture);
				this._particleVolumeTexture = null;
				this._warn(
					"webgl-particle-shadow-volume-create-failed",
					"Failed to allocate WebGL particle shadow volume storage.",
				);
				throw error;
			} finally {
				gl.bindTexture(gl.TEXTURE_2D, null);
			}
		}
		const requiredPixels = PARTICLE_ATLAS_WIDTH * PARTICLE_ATLAS_HEIGHT;
		if (this._particleVolumePixels.length !== requiredPixels) {
			this._particleVolumePixels = new Float32Array(requiredPixels);
		}
		this._particlePreparedThisFrame = true;
		this._samplingState.particleVolumeTexture = this._particleVolumeTexture;
	}

	private _updateParticleVolumes(context: FrameContext): void {
		this._clearParticleSamplingData();
		const batches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) as
			| readonly ParticleRenderBatch[]
			| undefined;
		const shadow = this._lightState?.directionalShadows[0];
		if (
			!this._samplingState.enabled ||
			!this._particlePreparedThisFrame ||
			!this._samplingState.particleVolumeTexture ||
			!shadow?.enabled ||
			!hasParticleShadowCastingBatches(batches)
		)
			return;
		this._particleVolumePixels.fill(0);
		const matrices =
			shadow.strategyType === "csm"
				? shadow.cascadeViewProjectionMatrices
				: [shadow.viewProjectionMatrix];
		const cascadeCount =
			shadow.strategyType === "csm" ? Math.max(1, Math.min(4, shadow.cascadeCount | 0)) : 1;
		let activeSlices = 0;
		for (
			let sliceIndex = 0;
			sliceIndex < Math.min(WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES, cascadeCount);
			sliceIndex++
		) {
			const matrix = matrices[sliceIndex];
			if (!matrix) continue;
			const grid = createParticleShadowVolumeGrid({
				width: WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH,
				height: WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT,
				depth: WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH,
			});
			for (const batch of batches ?? []) {
				injectParticleBatchIntoShadowVolume(grid, matrix, batch);
			}
			if (!grid.active) continue;
			this._packSlice(grid.density, sliceIndex);
			const offset = sliceIndex * 4;
			this._samplingState.particleVolumeSliceParams[offset] = 1;
			this._samplingState.particleVolumeSliceParams[offset + 1] =
				sliceIndex * WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH;
			activeSlices++;
		}
		if (activeSlices === 0) return;
		this._samplingState.particleVolumeActiveSliceCount = activeSlices;
		this._samplingState.particleVolumeAtlasSize[0] = PARTICLE_ATLAS_WIDTH;
		this._samplingState.particleVolumeAtlasSize[1] = PARTICLE_ATLAS_HEIGHT;
		this._samplingState.particleVolumeGridSize[0] = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH;
		this._samplingState.particleVolumeGridSize[1] = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT;
		this._samplingState.particleVolumeGridSize[2] = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH;
		this._samplingState.particleVolumeGridSize[3] = WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS;
		const gl = this._host.gl;
		try {
			gl.bindTexture(gl.TEXTURE_2D, this._samplingState.particleVolumeTexture);
			gl.texImage2D(
				gl.TEXTURE_2D,
				0,
				gl.R32F,
				PARTICLE_ATLAS_WIDTH,
				PARTICLE_ATLAS_HEIGHT,
				0,
				gl.RED,
				gl.FLOAT,
				this._particleVolumePixels,
			);
		} finally {
			gl.bindTexture(gl.TEXTURE_2D, null);
		}
	}

	private _packSlice(density: Float32Array, sliceIndex: number): void {
		const width = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH;
		const height = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT;
		const depth = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH;
		for (let z = 0; z < depth; z++) {
			const tileIndex = sliceIndex * depth + z;
			const tileX = tileIndex % WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS;
			const tileY = Math.floor(tileIndex / WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS);
			for (let y = 0; y < height; y++) {
				const sourceOffset = z * width * height + y * width;
				const targetOffset = (tileY * height + y) * PARTICLE_ATLAS_WIDTH + tileX * width;
				this._particleVolumePixels.set(
					density.subarray(sourceOffset, sourceOffset + width),
					targetOffset,
				);
			}
		}
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
		this._particlePreparedThisFrame = false;
		this._plan.atlasTileSize = 0;
		this._plan.sliceCount = 0;
		this._plan.casterPackets = [];
		this._plan.transmitterPackets = [];
		this._samplingState.enabled = false;
		this._samplingState.atlasTexture = null;
		this._samplingState.transmittanceTexture = null;
		this._samplingState.atlasTileSize = 0;
		this._samplingState.transmittanceAvailable = false;
		this._samplingState.particleVolumeTexture = null;
		this._clearParticleSamplingData();
	}

	private _clearParticleSamplingData(): void {
		this._samplingState.particleVolumeActiveSliceCount = 0;
		this._samplingState.particleVolumeAtlasSize.fill(0);
		this._samplingState.particleVolumeGridSize.fill(0);
		this._samplingState.particleVolumeSliceParams.fill(0);
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

	private _warn(key: string, message: string): void {
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGLShadowRuntime",
			onceKey: key,
		});
	}
}
