# WebGPU Post-Process Extension Contract
## Scope
This document defines the WebGPU-specific behavior behind the cross-backend post-process contract.

## Background
WebGPU post-processing is driven through `PostProcessPipeline`, `PostProcessBackendAdapter`, and `IPostProcessExecutor`. The public extension point is `renderer.postProcess.registerPass(pass)`. WebGPU-specific runtime objects remain internal implementation details used by `WebGPUBackend` and frame delegates.

## API/Contract
- `renderer.postProcess.registerPass(pass)` must register a `PostProcessPass` instance.
- A WebGPU custom pass must include a `webgpu` entry in `PostProcessPassConfig.implementations`.
- A WebGPU custom pass should use `PostProcessPass.placement` and optional `PostProcessPass.order` to enter the fixed post-process sequence.
- `WebGPUBackend` must register a `PostProcessBackendAdapter` during construction.
- `resolvePostProcessBackendAdapter(webGPUBackend).backend` must be `"webgpu"`.
- `resolvePostProcessBackendAdapter(webGPUBackend).executor.executePass(passId, request)` must dispatch backend-owned fallback post-process passes.
- `resolvePostProcessBackendAdapter(webGPUBackend).executor.getPassExecutionContext(request)` may provide low-level helpers for pass-owned WebGPU implementations.
- `PostProcessPassImplementation.metadata.context.backend` must be `"webgpu"` for WebGPU context packing.
- `PostProcessPassImplementation.metadata.context.kind` must be `"screen"` or `"present"`.
- `PostProcessPassImplementation.metadata.context` may request `publishColorTarget`, `frameBinding`, `lightingState`, history bindings, transient bindings, and a motion-history copy callback.
- `WebGPUPostProcessContextMetadata.transients` must map context properties to `PostProcessPassRequest.transients` ids.
- `PostProcessPassImplementation.metadata.warmupHints` may list WebGPU runtime warmup hint ids.
- `PostProcessPassExecutionContextRequest.implementation.metadata.context` must define whether WebGPU provides low-level helpers.
- `PostProcessPassExecutionContextRequest.pass.builtIn` must classify engine-owned passes and must not be required for WebGPU context packing.
- Pass-owned WebGPU implementations must use `PostProcessPassImplementation.execute(request, context)` instead of WebGPU runtime registration.
- WebGPU warmup must call `PostProcessPassImplementation.warmup(context)` for pass-owned implementations when it is present.
- WebGPU warmup must collect runtime hints from `PostProcessPassImplementation.metadata.warmupHints`.
- `resolvePostProcessBackendAdapter(webGPUBackend).createGBufferBridge(context)` must return a `LogicalGBufferBridge` that wraps WebGPU texture handles.
- WebGPU depth channels must declare `depthEncoding: "hardware"` unless the implementation provides a linearized depth texture.
- WebGPU motion channels must declare `motionEncoding: "ndc-delta"` when motion vectors are available.
- WebGPU temporal passes must read history resources from `request.histories`.
- WebGPU transient resources must be read from injected context properties declared by `WebGPUPostProcessContextMetadata.transients`.
- WebGPU temporal passes must return `updatedHistoryIds` or `historyUpdated` when they write pipeline-owned history resources.
- The built-in `taa`, `fxaa`, `ssao`, `ssr`, and `volumetric` WebGPU kernels must be pass-owned implementations.
- The built-in WebGPU `ssao` pass must request `ssao:raw` and `ssao:blur` transients sized by its resolved `downsample` option.
- The built-in WebGPU `ssr` pass must request `ssr:raw` sized by its resolved `downsample` option and the shared `hiz` full-chain transient.
- The built-in WebGPU `volumetric` pass must request the same shared `hiz` full-chain transient as `ssr`.
- `WebGPUFrameTargets` must not contain post-process transient textures such as SSAO intermediates, SSR intermediates, or Hi-Z textures.
- WebGPU executor resource allocation must use backend-owned texture creation and destruction APIs.
- `resolvePostProcessBackendAdapter(webGPUBackend).executor.createResource(desc)` must create a full mip chain when `desc.mipMode` is `"full-chain"`.
- `resolvePostProcessBackendAdapter(webGPUBackend).executor.invalidateResourceBindings()` must invalidate post-process binding caches when transient resources are recreated.
- WebGPU backends must not expose a public `postProcess` facade or backend-level post-process registration methods.
- WebGPU backends must not expose public `postProcessExecutor` or `createPostProcessGBufferBridge(context)` members.

## Usage
```ts
import {
	PostProcessPass,
	type PostProcessPassResolveRequest,
	type PostProcessTransientDescriptor,
} from "ignisrenderer";

class CustomWebGPUResolvePass extends PostProcessPass {
	public constructor() {
		super({
			id: "custom-webgpu-resolve",
			placement: "ldr",
			order: 5,
			enabled: true,
			implementations: {
				webgpu: {
					id: "custom-webgpu-resolve:webgpu",
					metadata: {
						context: {
							backend: "webgpu",
							kind: "screen",
							publishColorTarget: true,
							transients: [{
								property: "scratch",
								transientId: "custom-webgpu-resolve:scratch",
							}],
						},
					},
					execute(_request, context) {
						const webgpuContext = context as
							| { scratch?: unknown }
							| undefined;
						return webgpuContext?.scratch ? { ran: true } : { ran: false };
					},
				},
			},
		});
	}

	public override getTransientResourceDescriptors(
		_request: PostProcessPassResolveRequest
	): readonly PostProcessTransientDescriptor[] {
		return [{
			id: "custom-webgpu-resolve:scratch",
			format: "rgba16float",
			usage: ["sampled", "storage", "render-target"],
		}];
	}
}

renderer.postProcess.registerPass(new CustomWebGPUResolvePass());
```

```bash
bun tests/test_postprocess_public_api.mjs
bun tests/test_webgpu_postprocess_runtime_spatial.mjs
bun tests/test_webgpu_postprocess_runtime_temporal.mjs
bun tests/test_webgpu_postprocess_runtime_screen.mjs
```

## Errors & Diagnostics
- `renderer.postProcess.registerPass(pass)` must throw when `pass` is not a `PostProcessPass`.
- `renderer.postProcess.registerPass(pass)` must throw when `pass.id` is already registered.
- `postprocess-requirement-missing-<passId>` must be emitted when the WebGPU G-buffer bridge lacks a required semantic channel.
- `postprocess-transient-conflict-<transientId>` must be emitted when eligible passes request incompatible transient descriptors.
- WebGPU device allocation failures during `createResource(desc)` must propagate as backend resource allocation errors.

## Compatibility / Breaking Changes
- `WebGPUPostProcessPassPlugin` is removed from the public API.
- `WebGPUBackend.postProcess` is removed.
- `WebGPUBackend.postProcess.registerPass(pass)` and `WebGPUBackend.postProcess.unregisterPass(id)` are removed.
- `WebGPUBackend.postProcessExecutor` and `WebGPUBackend.createPostProcessGBufferBridge(context)` are removed; use `resolvePostProcessBackendAdapter(webGPUBackend)` for internal backend adapter access.
- Public custom passes must migrate to `PostProcessPass` and `renderer.postProcess.registerPass(pass)`.
- `PostProcessPassDescriptor.dependsOn` is removed. Custom passes must migrate to `placement` and optional `order`.
- `WebGPUFrameTargets.aoRaw`, `WebGPUFrameTargets.aoBlur`, `WebGPUFrameTargets.ssrRaw`, and `WebGPUFrameTargets.hiZ` are removed.
- WebGPU pass-owned implementations that need temporary textures must declare `getTransientResourceDescriptors(request)` and `metadata.context.transients`.
