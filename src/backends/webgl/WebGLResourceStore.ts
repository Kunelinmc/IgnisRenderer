import type {
	BindingEntry,
	IBindingGroup,
	IRenderBuffer,
	IRenderPipeline,
	IRenderTexture,
	ISampler,
	IShaderModule,
	PipelineDesc,
	ShaderModuleDesc,
} from "../types";
import type { WebGLColorRenderTargetFormat } from "./WebGLRenderTargetFormat";

/** @internal WebGL resource ownership; applications use renderer resource facades. */
export interface WebGLTextureState {
	texture: WebGLTexture;
	colorFormat?: WebGLColorRenderTargetFormat;
	maxMipLevel: number;
}

/** @internal WebGL resource ownership; applications use renderer resource facades. */
export interface WebGLPipelineState {
	program: WebGLProgram;
	desc: PipelineDesc;
	parent?: IRenderPipeline;
}

interface ResourceLifetime {
	destroyed: boolean;
	dispose(): void;
}

interface ResourceRecord<T> extends ResourceLifetime {
	value: T;
}

/**
 * Owns private state for one WebGL resource/context lifetime.
 *
 * @internal WebGL runtime service; applications use `Renderer` and its facades.
 */
export class WebGLResourceStore {
	private readonly _textures = new WeakMap<object, ResourceRecord<WebGLTextureState>>();
	private readonly _buffers = new WeakMap<object, ResourceRecord<WebGLBuffer>>();
	private readonly _samplers = new WeakMap<object, ResourceRecord<WebGLSampler>>();
	private readonly _shaders = new WeakMap<object, ResourceRecord<ShaderModuleDesc>>();
	private readonly _pipelines = new WeakMap<object, ResourceRecord<WebGLPipelineState>>();
	private readonly _groups = new WeakMap<object, ResourceRecord<readonly BindingEntry[]>>();
	private readonly _scopes = new WeakMap<object, { active: boolean }>();
	private readonly _lifetimes = new Set<ResourceLifetime>();
	private _destroyed = false;

	/** @internal WebGL owner validation; applications use renderer resource facades. */
	public assertAlive(): void {
		if (this._destroyed) throw new Error("WebGL resource owner has been destroyed.");
	}

	/** @internal Registers WebGL texture state; applications use renderer resource facades. */
	public createTexture(
		metadata: Omit<IRenderTexture, "kind" | "destroy">,
		state: WebGLTextureState,
		dispose: () => void,
	): IRenderTexture {
		return this._register(this._textures, { kind: "texture" as const, ...metadata }, state, dispose);
	}

	/** @internal Registers WebGL buffer state; applications use renderer resource facades. */
	public createBuffer(size: number, buffer: WebGLBuffer, dispose: () => void): IRenderBuffer {
		return this._register(this._buffers, {
			kind: "buffer" as const, size,
			unmap: () => {},
		}, buffer, dispose);
	}

	/** @internal Registers WebGL sampler state; applications use renderer resource facades. */
	public createSampler(label: string | undefined, sampler: WebGLSampler,
		dispose: () => void): ISampler {
		return this._register(this._samplers, { kind: "sampler" as const, label }, sampler, dispose);
	}

	/** @internal Stores WebGL shader source; applications use renderer resource facades. */
	public createShader(desc: ShaderModuleDesc): IShaderModule {
		return this._register(this._shaders, { kind: "shader-module" as const, label: desc.label },
			{ ...desc }, () => {});
	}

	/** @internal Registers WebGL pipeline state; applications use renderer resource facades. */
	public createPipeline(label: string | undefined, state: WebGLPipelineState,
		dispose: () => void): IRenderPipeline {
		return this._register(this._pipelines, { kind: "render-pipeline" as const, label }, state, dispose);
	}

	/** @internal Creates a scoped WebGL cache alias; applications use renderer resource facades. */
	public aliasPipeline(pipeline: IRenderPipeline): IRenderPipeline {
		const state = this.getPipeline(pipeline);
		return this.createPipeline(pipeline.label, { ...state, parent: pipeline }, () => {});
	}

	/** @internal Stores WebGL binding entries; applications use renderer resource facades. */
	public createBindingGroup(label: string | undefined,
		entries: readonly BindingEntry[]): IBindingGroup {
		for (const entry of entries) this._assertBindingResource(entry.resource);
		return this._register(this._groups, { kind: "binding-group" as const, label },
			entries.map((entry) => ({ ...entry, resource: "buffer" in entry.resource ?
				{ ...entry.resource } : entry.resource })), () => {});
	}

	/** @internal Resolves owned WebGL textures; applications use renderer resource facades. */
	public getTexture(resource: unknown): WebGLTextureState {
		return this._require(this._textures, resource, "texture");
	}

	/** @internal Resolves owned WebGL buffers; applications use renderer resource facades. */
	public getBuffer(resource: unknown): WebGLBuffer {
		return this._require(this._buffers, resource, "buffer");
	}

	/** @internal Resolves owned WebGL samplers; applications use renderer resource facades. */
	public getSampler(resource: unknown): WebGLSampler {
		return this._require(this._samplers, resource, "sampler");
	}

	/** @internal Resolves private WebGL source; applications use renderer resource facades. */
	public getShader(resource: unknown): ShaderModuleDesc {
		return this._require(this._shaders, resource, "shader module");
	}

	/** @internal Resolves owned WebGL pipelines; applications use renderer resource facades. */
	public getPipeline(resource: unknown): WebGLPipelineState {
		const state = this._require(this._pipelines, resource, "pipeline");
		if (state.parent) this.getPipeline(state.parent);
		return state;
	}

	/** @internal Resolves owned WebGL bindings; applications use renderer resource facades. */
	public getBindingGroup(resource: unknown): readonly BindingEntry[] {
		const entries = this._require(this._groups, resource, "binding group");
		for (const entry of entries) this._assertBindingResource(entry.resource);
		return entries;
	}

	/** @internal Binds WebGL auxiliary scope ownership; applications use scheduled raster work. */
	public markScope<T extends object>(resource: T, scope: { active: boolean }): T {
		this.assertResource(resource);
		this._scopes.set(resource, scope);
		return resource;
	}

	/** @internal Validates WebGL auxiliary scope ownership; applications use scheduled raster work. */
	public assertScope(resource: unknown, scope: { active: boolean }): void {
		const range = resource as { buffer?: unknown } | null;
		if (range?.buffer) resource = range.buffer;
		this.assertResource(resource);
		if (this._scopes.get(resource as object) !== scope) {
			throw new Error("WebGL auxiliary raster resources must belong to the active scope.");
		}
	}

	/** @internal Validates WebGL resource ownership; applications use renderer resource facades. */
	public assertResource(resource: unknown): void {
		const range = resource as { buffer?: unknown } | null;
		if (range?.buffer) {
			this.getBuffer(range.buffer);
			return;
		}
		const kind = (resource as { kind?: string } | null)?.kind;
		switch (kind) {
			case "texture": this.getTexture(resource); return;
			case "buffer": this.getBuffer(resource); return;
			case "sampler": this.getSampler(resource); return;
			case "shader-module": this.getShader(resource); return;
			case "render-pipeline": this.getPipeline(resource); return;
			case "binding-group": this.getBindingGroup(resource); return;
			default: throw new Error("WebGL resource is invalid or belongs to another owner.");
		}
	}

	/** @internal Releases the WebGL owner's allocations; applications destroy their renderer. */
	public destroy(): void {
		if (this._destroyed) return;
		this._destroyed = true;
		for (const lifetime of Array.from(this._lifetimes).reverse()) this._release(lifetime);
	}

	private _register<T extends object, S>(
		table: WeakMap<object, ResourceRecord<S>>,
		metadata: T,
		value: S,
		dispose: () => void,
	): T & { destroy(): void } {
		this.assertAlive();
		const record: ResourceRecord<S> = { value, destroyed: false, dispose };
		const resource = Object.assign(metadata, {
			destroy: () => {
				this._release(record);
				table.delete(resource);
			},
		});
		table.set(resource, record);
		this._lifetimes.add(record);
		return resource;
	}

	private _require<T>(table: WeakMap<object, ResourceRecord<T>>,
		resource: unknown, kind: string): T {
		const record = resource && typeof resource === "object" ? table.get(resource) : undefined;
		if (this._destroyed || !record || record.destroyed) {
			throw new Error(`WebGL ${kind} resource is unavailable, destroyed, or belongs to another owner.`);
		}
		const scope = this._scopes.get(resource as object);
		if (scope && !scope.active) {
			throw new Error("WebGL auxiliary raster scope is no longer active.");
		}
		return record.value;
	}

	private _release(lifetime: ResourceLifetime): void {
		if (lifetime.destroyed) return;
		lifetime.destroyed = true;
		this._lifetimes.delete(lifetime);
		lifetime.dispose();
	}

	private _assertBindingResource(resource: unknown): void {
		const kind = (resource as { kind?: string } | null)?.kind;
		const buffer = (resource as { buffer?: unknown } | null)?.buffer;
		if (kind !== "texture" && kind !== "buffer" && kind !== "sampler" && !buffer) {
			throw new Error("WebGL binding resource must be an owned texture, buffer, or sampler.");
		}
		this.assertResource(resource);
	}
}
