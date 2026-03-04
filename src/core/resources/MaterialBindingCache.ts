import {
	BufferUsage,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
} from "../ral/types";
import type { WebGPUBackend } from "../backend/WebGPUBackend";
import type { DrawPacket } from "../pipeline/types";
import {
	WEBGPU_MODEL_UNIFORM_FLOATS,
	createWebGPUMaterialUniformData,
	packModelUniformData,
	type WebGPUMaterialUniformData,
} from "../bridge/webgpuUtils";
import type { WebGPUPipelineLayouts } from "../backend/webgpu/WebGPUPipelineLayouts";

interface MaterialBindingEntry {
	uniformBuffer: IRenderBuffer;
	bindingGroup: IBindingGroup | null;
	pipeline: IRenderPipeline | null;
	textures: IRenderTexture[];
	samplers: ISampler[];
}

export class MaterialBindingCache {
	private _backend: WebGPUBackend;
	private _layouts: WebGPUPipelineLayouts;
	private _cache = new Map<string, MaterialBindingEntry>();

	constructor(backend: WebGPUBackend, layouts: WebGPUPipelineLayouts) {
		this._backend = backend;
		this._layouts = layouts;
	}

	public getBinding(
		packet: DrawPacket,
		pipeline: IRenderPipeline,
		materialData: WebGPUMaterialUniformData,
		textures: IRenderTexture[],
		samplers: ISampler[]
	): IBindingGroup {
		let cached = this._cache.get(packet.id);
		if (!cached) {
			cached = {
				uniformBuffer: this._backend.createBuffer({
					size: WEBGPU_MODEL_UNIFORM_FLOATS * 4,
					usage: BufferUsage.Uniform | BufferUsage.CopyDst,
					label: `ModelUniform_${packet.id}`,
				}),
				bindingGroup: null,
				pipeline: null,
				textures: [],
				samplers: [],
			};
			this._cache.set(packet.id, cached);
		}

		const uniformData = packModelUniformData(
			packet.worldMatrix,
			packet.normalMatrix as any,
			materialData
		);
		this._backend.writeBuffer(
			cached.uniformBuffer,
			new Float32Array(uniformData)
		);

		if (
			!cached.bindingGroup ||
			cached.pipeline !== pipeline ||
			!areTexturesEqual(cached.textures, textures) ||
			!areSamplersEqual(cached.samplers, samplers)
		) {
			const entries: Array<{ binding: number; resource: any }> = [
				{ binding: 0, resource: cached.uniformBuffer },
			];
			for (let i = 0; i < textures.length; i++) {
				entries.push({ binding: 1 + i * 2, resource: textures[i] });
				entries.push({ binding: 2 + i * 2, resource: samplers[i] });
			}
			cached.bindingGroup = this._backend.createBindingGroup({
				label: `ModelBinding_${packet.id}`,
				layout: this._layouts.modelBindGroupLayout,
				entries,
			});
			cached.pipeline = pipeline;
			cached.textures = textures.slice();
			cached.samplers = samplers.slice();
		}

		return cached.bindingGroup;
	}
}

function areTexturesEqual(
	left: IRenderTexture[],
	right: IRenderTexture[]
): boolean {
	if (left.length !== right.length) return false;
	for (let i = 0; i < left.length; i++) {
		if (left[i] !== right[i]) return false;
	}
	return true;
}

function areSamplersEqual(left: ISampler[], right: ISampler[]): boolean {
	if (left.length !== right.length) return false;
	for (let i = 0; i < left.length; i++) {
		if (left[i] !== right[i]) return false;
	}
	return true;
}
