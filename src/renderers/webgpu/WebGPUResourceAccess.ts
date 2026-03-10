import type {
	IBindingGroup,
	IComputePipeline,
	IRenderBuffer,
	IRenderPipeline,
	IRenderTexture,
	ISampler,
	IShaderModule,
} from "../types"

export interface WebGPUTexture {
	texture: GPUTexture
	view: GPUTextureView
}

interface WebGPUBackedRenderBuffer extends IRenderBuffer {
	_gpuResource?: GPUBuffer
}

interface WebGPUBackedRenderTexture extends IRenderTexture {
	_gpuResource?: GPUTexture
	_gpuTexture?: GPUTexture
	_gpuView?: GPUTextureView
	_webgpuTexture?: WebGPUTexture
}

interface WebGPUBackedSampler extends ISampler {
	_gpuResource?: GPUSampler
}

interface WebGPUBackedShaderModule extends IShaderModule {
	_gpuResource?: GPUShaderModule
}

interface WebGPUBackedRenderPipeline extends IRenderPipeline {
	_gpuResource?: GPURenderPipeline
}

interface WebGPUBackedComputePipeline extends IComputePipeline {
	_gpuResource?: GPUComputePipeline
}

interface WebGPUBackedBindingGroup extends IBindingGroup {
	_gpuResource?: GPUBindGroup
}

export function createWebGPUTexture(
	texture: GPUTexture,
	view?: GPUTextureView
): WebGPUTexture {
	return {
		texture,
		view: view ?? texture.createView(),
	}
}

export function attachWebGPUTexture(
	target: IRenderTexture,
	resource: WebGPUTexture
): void {
	const internal = target as WebGPUBackedRenderTexture
	internal._gpuResource = resource.texture
	internal._gpuTexture = resource.texture
	internal._gpuView = resource.view
	internal._webgpuTexture = resource
}

export function getWebGPUTexture(texture: IRenderTexture): WebGPUTexture {
	const resolved = tryGetWebGPUTexture(texture)
	if (!resolved) {
		throw new Error("Render texture is not backed by a WebGPU texture.")
	}
	return resolved
}

export function tryGetWebGPUTextureHandle(resource: unknown): GPUTexture | null {
	if (!resource || typeof resource !== "object") {
		return null
	}

	const internal = resource as WebGPUBackedRenderTexture
	return (
		internal._webgpuTexture?.texture ??
		internal._gpuTexture ??
		internal._gpuResource ??
		null
	)
}

export function tryGetWebGPUTexture(resource: unknown): WebGPUTexture | null {
	if (!resource || typeof resource !== "object") {
		return null
	}

	const internal = resource as WebGPUBackedRenderTexture
	if (
		internal._webgpuTexture?.texture &&
		internal._webgpuTexture?.view
	) {
		return internal._webgpuTexture
	}

	const texture = tryGetWebGPUTextureHandle(resource)
	if (!texture || typeof texture.createView !== "function") {
		return null
	}

	const view = internal._gpuView ?? texture.createView()
	const resolved = createWebGPUTexture(texture, view)
	internal._webgpuTexture = resolved
	internal._gpuResource = texture
	internal._gpuTexture = texture
	internal._gpuView = view
	return resolved
}

export function getWebGPUBuffer(buffer: IRenderBuffer): GPUBuffer {
	const resource = (buffer as WebGPUBackedRenderBuffer)._gpuResource
	if (!resource || typeof resource.destroy !== "function") {
		throw new Error("Render buffer is not backed by a WebGPU buffer.")
	}
	return resource
}

export function tryGetWebGPUBuffer(resource: unknown): GPUBuffer | null {
	const gpuBuffer = (resource as WebGPUBackedRenderBuffer)?._gpuResource
	if (!gpuBuffer || typeof gpuBuffer.destroy !== "function") {
		return null
	}
	return gpuBuffer
}

export function getWebGPUSampler(sampler: ISampler): GPUSampler {
	const resource = (sampler as WebGPUBackedSampler)._gpuResource
	if (!resource) {
		throw new Error("Sampler is not backed by a WebGPU sampler.")
	}
	return resource
}

export function getWebGPUShaderModule(module: IShaderModule): GPUShaderModule {
	const resource = (module as WebGPUBackedShaderModule)._gpuResource
	if (!resource) {
		throw new Error("Shader module is not backed by a WebGPU shader module.")
	}
	return resource
}

export function getWebGPURenderPipeline(
	pipeline: IRenderPipeline
): GPURenderPipeline {
	const resource = (pipeline as WebGPUBackedRenderPipeline)._gpuResource
	if (!resource) {
		throw new Error("Render pipeline is not backed by a WebGPU pipeline.")
	}
	return resource
}

export function getWebGPUComputePipeline(
	pipeline: IComputePipeline
): GPUComputePipeline {
	const resource = (pipeline as WebGPUBackedComputePipeline)._gpuResource
	if (!resource) {
		throw new Error("Compute pipeline is not backed by a WebGPU pipeline.")
	}
	return resource
}

export function getWebGPUBindGroup(group: IBindingGroup): GPUBindGroup {
	const resource = (group as WebGPUBackedBindingGroup)._gpuResource
	if (!resource) {
		throw new Error("Binding group is not backed by a WebGPU bind group.")
	}
	return resource
}

export function tryGetWebGPUPipeline(
	pipeline: IRenderPipeline | IComputePipeline | null | undefined
): GPURenderPipeline | GPUComputePipeline | null {
	const resource = (pipeline as {
		_gpuResource?: GPURenderPipeline | GPUComputePipeline
	})?._gpuResource
	if (!resource || typeof resource.getBindGroupLayout !== "function") {
		return null
	}
	return resource
}

export function getWebGPUPipeline(
	pipeline: IRenderPipeline | IComputePipeline | null | undefined
): GPURenderPipeline | GPUComputePipeline {
	const resolved = tryGetWebGPUPipeline(pipeline)
	if (!resolved) {
		throw new Error("Pipeline is not backed by a WebGPU pipeline.")
	}
	return resolved
}
