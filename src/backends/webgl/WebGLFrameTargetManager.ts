import type { FrameContext } from "../../pipeline/types";
import type {
	LogicalGBufferBridge,
	LogicalGBufferChannel,
	LogicalGBufferSemantic,
} from "../../postprocess";
import type {
	RenderGraphPhysicalBinding,
	RenderGraphResourceDescriptor,
} from "../../rendergraph/types";
import type {
	WebGLFrameGraphResourceCatalogSnapshot,
} from "./rendergraph/types";

import {
	bindWebGLOITSingleColorTarget,
	bindWebGLPostSingleColorTarget,
	destroyWebGLFrameTargets,
	ensureWebGLFrameTargets,
	resolveWebGLPostProcessTargetTexture,
	type WebGLFrameTargetFormat,
	type WebGLFrameTargetLifecycleHost,
} from "./WebGLFrameTargetLifecycle";

/** Owns all frame-sized WebGL framebuffer attachments for one context. */
export class WebGLFrameTargetManager implements WebGLFrameTargetLifecycleHost {
	public readonly _gl: WebGL2RenderingContext;
	public readonly _maxTextureSize: number;
	public readonly _maxRenderbufferSize: number;
	public _sceneFramebuffer: WebGLFramebuffer | null = null;
	public _sceneColorTexture: WebGLTexture | null = null;
	public _sceneColorFormat: WebGLFrameTargetFormat = "rgba8unorm";
	public _sceneMotionTexture: WebGLTexture | null = null;
	public _sceneMotionFormat: WebGLFrameTargetFormat = "rgba8unorm";
	public _sceneNormalTexture: WebGLTexture | null = null;
	public _sceneNormalFormat: WebGLFrameTargetFormat = "rgba8unorm";
	public _sceneAlbedoTexture: WebGLTexture | null = null;
	public _sceneAlbedoFormat: WebGLFrameTargetFormat = "rgba8unorm";
	public _sceneSpecularTexture: WebGLTexture | null = null;
	public _sceneSpecularFormat: WebGLFrameTargetFormat = "rgba8unorm";
	public _materialGBufferEnabled = false;
	public _sceneDepthBuffer: WebGLRenderbuffer | null = null;
	public _oitFramebuffer: WebGLFramebuffer | null = null;
	public _oitAccumTexture: WebGLTexture | null = null;
	public _oitRevealTexture: WebGLTexture | null = null;
	public _postFramebuffer: WebGLFramebuffer | null = null;
	public _postColorTexture: WebGLTexture | null = null;
	public _postColorFormat: WebGLFrameTargetFormat = "rgba8unorm";
	public _presentSourceTexture: WebGLTexture | null = null;
	public _targetWidth = 0;
	public _targetHeight = 0;
	public _targetMaterialGBufferEnabled = false;
	public _supportsFloatColorBuffer: boolean | null = null;
	private readonly _graphPhysicalResources = new Map<string, object>();

	public constructor(
		gl: WebGL2RenderingContext,
		maxTextureSize: number,
		maxRenderbufferSize: number,
	) {
		this._gl = gl;
		this._maxTextureSize = maxTextureSize;
		this._maxRenderbufferSize = maxRenderbufferSize;
	}

	public ensure(
		width: number,
		height: number,
		materialGBufferRequested: boolean,
	): void {
		ensureWebGLFrameTargets(
			this,
			width,
			height,
			materialGBufferRequested,
		);
	}

	public resize(): void {
		this.destroy();
	}

	public destroy(): void {
		this._graphPhysicalResources.clear();
		destroyWebGLFrameTargets(this);
	}

	public collectGraphResources(
		shadowAtlasTexture: WebGLTexture | null,
		shadowTransmittanceTexture: WebGLTexture | null,
	): readonly string[] {
		const resources = new Set<string>();
		if (this._sceneColorTexture) {
			resources.add("frame:scene-color");
			resources.add("frame:present-source");
		}
		if (this._sceneMotionTexture) resources.add("frame:motion-depth");
		if (this._sceneNormalTexture) resources.add("frame:normal");
		if (this._sceneDepthBuffer) resources.add("frame:depth");
		if (this._postColorTexture) resources.add("post:color");
		if (this._oitAccumTexture) resources.add("oit:accum");
		if (this._oitRevealTexture) resources.add("oit:reveal");
		if (shadowAtlasTexture) resources.add("shadow:atlas");
		if (shadowTransmittanceTexture) resources.add("shadow:transmittance");
		return Array.from(resources);
	}

	/**
	 * @internal Returns descriptor and stable physical identity metadata only.
	 * Shadow slots are declared only for frames that schedule shadow work.
	 */
	public collectGraphResourceCatalog(
		shadowAtlasTexture: WebGLTexture | null,
		shadowTransmittanceTexture: WebGLTexture | null,
		includeShadowResources = true,
	): WebGLFrameGraphResourceCatalogSnapshot {
		const resources: RenderGraphResourceDescriptor[] = [];
		const bindings: RenderGraphPhysicalBinding[] = [];
		this._graphPhysicalResources.clear();
		const width = Math.max(1, this._targetWidth);
		const height = Math.max(1, this._targetHeight);
		const addTexture = (
			id: string,
			handle: WebGLTexture | WebGLRenderbuffer | null,
			format?: string,
			options: {
				readonly physicalId?: string;
				readonly declareWithoutHandle?: boolean;
				readonly extent?: "frame" | "unknown";
			} = {},
		): void => {
			if (!handle && !options.declareWithoutHandle) return;
			const physicalId = options.physicalId ?? `webgl:slot:${id}`;
			resources.push({
				id,
				origin: "imported",
				kind: "texture",
				residency: "frame",
				initialContent: "unknown",
				format,
				...(options.extent === "unknown" ? {} : {
					width,
					height,
					depthOrArrayLayers: 1,
					dimension: "2d" as const,
					sampleCount: 1,
					mipLevelCount: 1,
				}),
			});
			bindings.push({
				resourceId: id,
				physicalId,
				kind: "texture",
			});
			if (handle) this._graphPhysicalResources.set(physicalId, handle);
		};
		addTexture("frame:scene-color", this._sceneColorTexture, this._sceneColorFormat);
		const presentPhysicalId = this._presentSourceTexture === this._sceneColorTexture
			? "webgl:slot:frame:scene-color"
			: this._presentSourceTexture === this._postColorTexture
				? "webgl:slot:post:color"
				: "webgl:slot:frame:present-source";
		addTexture("frame:present-source", this._presentSourceTexture ?? this._sceneColorTexture,
			this._sceneColorFormat, { physicalId: presentPhysicalId });
		addTexture("frame:motion-depth", this._sceneMotionTexture, this._sceneMotionFormat);
		addTexture("frame:normal", this._sceneNormalTexture, this._sceneNormalFormat);
		addTexture("frame:albedo", this._sceneAlbedoTexture, this._sceneAlbedoFormat);
		addTexture("frame:specular", this._sceneSpecularTexture, this._sceneSpecularFormat);
		addTexture("frame:depth", this._sceneDepthBuffer, "depth24plus-stencil8");
		addTexture("post:color", this._postColorTexture, this._postColorFormat);
		addTexture("oit:accum", this._oitAccumTexture, "rgba16float");
		addTexture("oit:reveal", this._oitRevealTexture, "r16float");
		if (includeShadowResources) {
			const lazyShadowSlot = {
				declareWithoutHandle: true,
				extent: "unknown",
			} as const;
			addTexture("shadow:atlas", shadowAtlasTexture, "depth", lazyShadowSlot);
			addTexture(
				"shadow:transmittance",
				shadowTransmittanceTexture,
				"rgba8unorm",
				lazyShadowSlot,
			);
		}
		resources.push({
			id: "canvas:color",
			origin: "imported",
			kind: "texture",
			residency: "external",
			initialContent: "unknown",
			width,
			height,
			depthOrArrayLayers: 1,
			dimension: "2d",
			sampleCount: 1,
			mipLevelCount: 1,
		});
		bindings.push({
			resourceId: "canvas:color",
			physicalId: "webgl:canvas-color",
			kind: "texture",
		});
		return Object.freeze({
			resources: Object.freeze(resources),
			bindings: Object.freeze(bindings),
		});
	}

	/** @internal Resolves backend-owned native state outside the logical graph. */
	public resolveGraphPhysicalResource(physicalId: string): object | null {
		return this._graphPhysicalResources.get(physicalId) ?? null;
	}

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		const width = Math.max(1, context.attachments.width);
		const height = Math.max(1, context.attachments.height);
		return {
			width,
			height,
			normalSpace: "world",
			depthEncoding: "hardware",
			motionEncoding: "ndc-delta",
			channels: {
				color: this._channel("color", this._sceneColorTexture, width, height,
					this._sceneColorFormat),
				depth: this._channel("depth", this._sceneMotionTexture, width, height,
					this._sceneMotionFormat, "motion-depth.z"),
				motion: this._channel("motion", this._sceneMotionTexture, width, height,
					this._sceneMotionFormat, "motion-depth.xy"),
				normal: this._channel("normal", this._sceneNormalTexture, width, height,
					this._sceneNormalFormat, "encoded-world-normal"),
				albedo: this._channel("albedo", this._sceneAlbedoTexture, width, height,
					this._sceneAlbedoFormat, "linear-rgb-alpha"),
				roughness: this._materialGBufferEnabled ?
					this._channel("roughness", this._sceneNormalTexture, width, height,
						this._sceneNormalFormat, "normal-roughness-metallic.z")
				: undefined,
				metallic: this._materialGBufferEnabled ?
					this._channel("metallic", this._sceneNormalTexture, width, height,
						this._sceneNormalFormat, "normal-roughness-metallic.w")
				: undefined,
				specular: this._channel("specular", this._sceneSpecularTexture, width, height,
					this._sceneSpecularFormat, "specular-color-factor.rgba"),
			},
			worldPosition: {
				source: "derived",
				available: !!this._sceneMotionTexture,
			},
		};
	}

	public resolvePostProcessTargetTexture(
		sourceTexture: WebGLTexture,
	): WebGLTexture | null {
		return resolveWebGLPostProcessTargetTexture(this, sourceTexture);
	}

	public bindPostSingleColorTarget(texture: WebGLTexture): void {
		bindWebGLPostSingleColorTarget(this, texture);
	}

	public bindOITSingleColorTarget(texture: WebGLTexture): void {
		bindWebGLOITSingleColorTarget(this, texture);
	}

	public supportsFloatColorBuffer(): boolean {
		if (this._supportsFloatColorBuffer === null) {
			this._supportsFloatColorBuffer = !!this._gl.getExtension(
				"EXT_color_buffer_float",
			);
		}
		return this._supportsFloatColorBuffer;
	}

	private _channel(
		semantic: LogicalGBufferSemantic,
		texture: WebGLTexture | null,
		width: number,
		height: number,
		format: WebGLFrameTargetFormat,
		encoding?: string,
	): LogicalGBufferChannel | undefined {
		if (!texture) return undefined;
		return {
			semantic,
			handle: { backend: "webgl", texture },
			width,
			height,
			format,
			encoding,
		};
	}
}
