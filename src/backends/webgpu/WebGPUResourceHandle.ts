import { WebGPUResourceManager } from "./WebGPUResourceManager";

/**
 * Resolves a native identity through an authentic resource's private owner.
 *
 * @internal WebGPU backend implementation detail. Shared renderer contracts
 * do not carry native handles; applications use renderer resource factories.
 */
export function getWebGPUResourceHandle(resource: unknown): object | undefined {
	const owner = WebGPUResourceManager.ownerOf(resource);
	if (!owner) return undefined;
	const kind = (resource as { kind: string }).kind;
	switch (kind) {
		case "buffer": return owner.getBuffer(resource);
		case "texture": return owner.getTexture(resource).texture;
		case "sampler": return owner.getSampler(resource);
		case "shader-module": return owner.getShaderModule(resource);
		case "render-pipeline": return owner.getRenderPipeline(resource);
		case "compute-pipeline": return owner.getComputePipeline(resource);
		case "binding-group": return owner.getBindingGroup(resource);
		default: throw new Error("Unsupported WebGPU resource kind.");
	}
}
