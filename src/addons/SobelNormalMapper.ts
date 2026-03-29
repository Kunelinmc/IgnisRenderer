import { Texture } from "../core/Texture";
import { Renderer } from "../renderers/Renderer";
import { WebGPUBackend } from "../renderers/WebGPUBackend";
import { loadPostProcessShaderPart } from "../shaders/webgpu/shaderSource";
import { WEBGPU_TEXTURE_SLOT } from "../renderers/webgpu/constants";
import {
	BufferUsage,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
	TextureFormat,
	TextureUsage,
} from "../renderers/types";

const SOBEL_WORKGROUP_SIZE = 8;
const DEFAULT_STRENGTH = 2.0;

export interface SobelNormalMapperOptions {
	strength?: number;
	invertX?: boolean;
	invertY?: boolean;
}

/**
 * Converts a source texture to a normal map by running a Sobel compute pass.
 *
 * Lifecycle:
 * 1. Construct with source texture.
 * 2. Call `init(renderer)` once (or `attach(renderer)` for auto updates).
 * 3. Read `normalMap` and assign it to material normal-map slots.
 * 4. Call `destroy()` when done.
 */
export class SobelNormalMapper {
	private _backend: WebGPUBackend | null = null;
	private _renderer: Renderer | null = null;
	private _shaderModule: IShaderModule | null = null;
	private _pipeline: IComputePipeline | null = null;
	private _paramsBuffer: IRenderBuffer | null = null;
	private _destTexture: Texture;
	private _destResource: IRenderTexture | null = null;
	private _bindGroup: IBindingGroup | null = null;
	private _bindSourceTexture: IRenderTexture | null = null;
	private _bindDestTexture: IRenderTexture | null = null;
	private _isInitialized = false;
	private _destroyed = false;
	private _lastSourceVersion = -1;
	private _lastParamsKey = "";
	private _pendingForceUpdate = true;

	private _strength = DEFAULT_STRENGTH;
	private _invertX = false;
	private _invertY = false;

	private _onPostAnimation = (): void => {
		this.update();
	};

	public get strength(): number {
		return this._strength;
	}

	public set strength(value: number) {
		const next =
			Number.isFinite(value) ?
				Math.max(0, Number(value))
			:	DEFAULT_STRENGTH;
		if (next === this._strength) {
			return;
		}
		this._strength = next;
		this.invalidate();
	}

	public get invertX(): boolean {
		return this._invertX;
	}

	public set invertX(value: boolean) {
		const next = !!value;
		if (next === this._invertX) {
			return;
		}
		this._invertX = next;
		this.invalidate();
	}

	public get invertY(): boolean {
		return this._invertY;
	}

	public set invertY(value: boolean) {
		const next = !!value;
		if (next === this._invertY) {
			return;
		}
		this._invertY = next;
		this.invalidate();
	}

	constructor(
		private _source: Texture,
		options: SobelNormalMapperOptions = {}
	) {
		this._strength =
			Number.isFinite(options.strength) ?
				Math.max(0, Number(options.strength))
			:	DEFAULT_STRENGTH;
		this._invertX = !!options.invertX;
		this._invertY = !!options.invertY;
		this._destTexture = new Texture(
			null,
			Math.max(1, _source.width | 0),
			Math.max(1, _source.height | 0),
			"Linear"
		);
	}

	public get normalMap(): Texture {
		return this._destTexture;
	}

	public get isInitialized(): boolean {
		return this._isInitialized;
	}

	public get isAttached(): boolean {
		return this._renderer !== null;
	}

	/**
	 * Initializes GPU resources and binds the destination texture into
	 * WebGPU texture resolution for material slots.
	 */
	public async init(renderer: Renderer): Promise<void> {
		if (this._destroyed) {
			throw new Error("SobelNormalMapper has been destroyed.");
		}
		if (renderer.backend.type !== "webgpu") {
			throw new Error("SobelNormalMapper requires WebGPU backend.");
		}
		const backend = renderer.backend as WebGPUBackend;
		if (this._isInitialized && this._backend === backend) {
			return;
		}
		if (this._isInitialized) {
			this._releaseGPUResources();
		}
		this._backend = backend;

		const code = await loadPostProcessShaderPart("sobelNormal");
		this._shaderModule = await this._backend.createShaderModule({
			code,
			label: "SobelNormalComputeShader",
			language: "wgsl",
			stage: "compute",
			sourceKind: "postprocess",
		});

		this._pipeline = this._backend.createComputePipeline({
			compute: {
				module: this._shaderModule,
				entryPoint: "csMain",
			},
			label: "SobelNormalComputePipeline",
		});

		this._paramsBuffer = this._backend.createBuffer({
			size: 16, // 4 * float32
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			label: "SobelNormalParamsBuffer",
		});

		this._recreateDestTexture();
		this._lastSourceVersion = -1;
		this._lastParamsKey = "";
		this._pendingForceUpdate = true;
		this._isInitialized = true;
	}

	/**
	 * Attaches to renderer lifecycle and auto-updates the normal map every
	 * render frame in `postanimation`.
	 */
	public async attach(renderer: Renderer): Promise<this> {
		if (this._renderer === renderer && this._isInitialized) {
			return this;
		}
		this.detach();
		await this.init(renderer);
		this._renderer = renderer;
		renderer.on("postanimation", this._onPostAnimation);
		this.invalidate();
		return this;
	}

	/**
	 * Detaches automatic frame hook. Does not destroy GPU resources.
	 */
	public detach(): void {
		if (this._renderer) {
			this._renderer.off("postanimation", this._onPostAnimation);
		}
		this._renderer = null;
	}

	/**
	 * Marks Sobel output as dirty and schedules a render if attached.
	 */
	public invalidate(): void {
		this._pendingForceUpdate = true;
		this._renderer?.requestRender();
	}

	/**
	 * Executes one Sobel compute update.
	 * Returns true if a compute dispatch happened.
	 */
	public update(force: boolean = false): boolean {
		if (
			!this._isInitialized ||
			this._destroyed ||
			!this._backend ||
			!this._pipeline ||
			!this._paramsBuffer
		) {
			return false;
		}

		const sourceWidth = this._source.width | 0;
		const sourceHeight = this._source.height | 0;
		if (sourceWidth <= 0 || sourceHeight <= 0) {
			return false;
		}
		const width = sourceWidth;
		const height = sourceHeight;

		// Check if source resized
		if (
			width !== this._destTexture.width ||
			height !== this._destTexture.height
		) {
			this._recreateDestTexture();
		}

		if (!this._destResource) {
			return false;
		}

		const sourceVersion = this._source.version;
		const paramsKey = this._buildParamsKey();
		const sourceChanged = sourceVersion !== this._lastSourceVersion;
		const paramsChanged = paramsKey !== this._lastParamsKey;
		const shouldDispatch =
			force || this._pendingForceUpdate || sourceChanged || paramsChanged;
		if (!shouldDispatch) {
			return false;
		}

		const srcResource = this._backend.getTextureForSlot(
			this._source,
			WEBGPU_TEXTURE_SLOT.BASE_COLOR
		);
		if (!srcResource) {
			return false;
		}

		const params = new Float32Array([
			this._strength,
			this._invertX ? -1.0 : 1.0,
			this._invertY ? -1.0 : 1.0,
			0.0,
		]);
		this._backend.writeBuffer(this._paramsBuffer, params);

		this._ensureBindGroup(srcResource);
		if (!this._bindGroup) {
			return false;
		}

		const encoder = this._backend.createCommandEncoder();
		encoder.beginComputePass({ label: "SobelNormalPass" });
		encoder.setComputePipeline(this._pipeline);
		encoder.setBindingGroup(0, this._bindGroup);
		encoder.dispatchWorkgroups(
			ceilDiv(width, SOBEL_WORKGROUP_SIZE),
			ceilDiv(height, SOBEL_WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		this._backend.submit([encoder.finish()]);

		this._lastSourceVersion = sourceVersion;
		this._lastParamsKey = paramsKey;
		this._pendingForceUpdate = false;
		return true;
	}

	/**
	 * Releases renderer hooks and GPU resources.
	 */
	public destroy(): void {
		if (this._destroyed) {
			return;
		}
		this.detach();
		this._releaseGPUResources();
		this._destroyed = true;
	}

	private _ensureBindGroup(sourceTexture: IRenderTexture): void {
		if (
			!this._backend ||
			!this._pipeline ||
			!this._paramsBuffer ||
			!this._destResource
		) {
			return;
		}
		const canReuse =
			this._bindGroup &&
			this._bindSourceTexture === sourceTexture &&
			this._bindDestTexture === this._destResource;
		if (canReuse) {
			return;
		}
		this._destroyBindingGroup();
		this._bindGroup = this._backend.createBindingGroup({
			pipeline: this._pipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: sourceTexture },
				{ binding: 1, resource: this._destResource },
				{ binding: 2, resource: this._paramsBuffer },
			],
			label: "SobelNormalBindGroup",
		});
		this._bindSourceTexture = sourceTexture;
		this._bindDestTexture = this._destResource;
	}

	private _recreateDestTexture(): void {
		if (!this._backend) return;
		const width = Math.max(1, this._source.width | 0);
		const height = Math.max(1, this._source.height | 0);

		this._destroyBindingGroup();
		if (this._destResource) {
			this._backend.unregisterExternalTexture(this._destTexture);
			this._destResource.destroy();
		}

		this._destTexture.width = width;
		this._destTexture.height = height;

		this._destResource = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA8Unorm,
			usage: TextureUsage.StorageBinding | TextureUsage.TextureBinding,
			label: "SobelNormalResultTexture",
		});
		this._backend.registerExternalTexture(
			this._destTexture,
			this._destResource,
			this._destTexture.version,
			1
		);
		this._bindSourceTexture = null;
		this._bindDestTexture = null;
		this._pendingForceUpdate = true;
	}

	private _releaseGPUResources(): void {
		if (this._backend) {
			this._backend.unregisterExternalTexture(this._destTexture);
		}
		this._destroyBindingGroup();
		this._destroyResource(this._paramsBuffer);
		this._paramsBuffer = null;
		this._destroyResource(this._destResource);
		this._destResource = null;
		this._destroyResource(this._pipeline);
		this._pipeline = null;
		this._destroyResource(this._shaderModule);
		this._shaderModule = null;
		this._bindSourceTexture = null;
		this._bindDestTexture = null;
		this._lastSourceVersion = -1;
		this._lastParamsKey = "";
		this._pendingForceUpdate = true;
		this._isInitialized = false;
		this._backend = null;
	}

	private _buildParamsKey(): string {
		return `${this._strength}|${this._invertX ? 1 : 0}|${this._invertY ? 1 : 0}`;
	}

	private _destroyBindingGroup(): void {
		this._destroyResource(this._bindGroup);
		this._bindGroup = null;
		this._bindSourceTexture = null;
		this._bindDestTexture = null;
	}

	private _destroyResource(resource: unknown): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(resource);
		}
	}
}

function ceilDiv(value: number, divisor: number): number {
	return Math.max(1, Math.ceil(value / Math.max(1, divisor)));
}
