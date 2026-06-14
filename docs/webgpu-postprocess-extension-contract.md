# WebGPU Post-Process Runtime Contract
## Scope
This document defines the WebGPU-specific behavior behind the cross-backend
post-process contract.

## Background
WebGPU post-processing is owned by `WebGPUBackendSession`. `Renderer` owns only
`renderer.postProcess`, while `WebGPUBackendSession` owns `BackendPostProcessRuntime`,
`WebGPUPostProcessExecutor`, concrete texture allocation, frame target
integration, and lifecycle invalidation.

## API/Contract
- `WebGPUBackendSession.profile.capabilities.postProcess` must be `true`.
- `WebGPUBackendSession.extensions` must not expose `renderer.postprocess`.
- `WebGPUBackendSession.executePass({ stage: "postprocess" }, context)` must delegate
  to backend-owned post-process runtime execution.
- `WebGPUBackendSession.endFrame()` must commit post-process histories only after the
  WebGPU frame executor ends successfully.
- `WebGPUBackendSession.abortFrame(error)` must abort the post-process runtime before
  clearing WebGPU frame executor and particle state.
- WebGPU resize, device loss, MSAA changes, shader runtime changes, and destroy
  must call post-process runtime invalidation or destruction directly.
- `renderer.postProcess.registerPass(pass)` must register a `PostProcessPass`
  instance for WebGPU execution.
- A WebGPU custom pass must include a `webgpu` entry in
  `PostProcessPassConfig.implementations`.
- A WebGPU custom pass should use `PostProcessPass.placement` and optional
  `PostProcessPass.order` to enter the fixed post-process sequence.
- `PostProcessPassImplementation.metadata.context.backend` must be `"webgpu"`
  for WebGPU context packing.
- `WebGPUPostProcessContextMetadata`,
  `WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA`, and
  `WEBGPU_PRESENT_POST_PROCESS_CONTEXT_METADATA` are internal extension
  contracts for pass-owned implementations. Application code should use
  `PostProcessPass` and `renderer.postProcess.registerPass(pass)` instead.
- `PostProcessPassImplementation.metadata.context.kind` must be `"screen"` or
  `"present"`.
- `PostProcessPassImplementation.metadata.context` may request
  `publishColorTarget`, `frameBinding`, `lightingState`, history bindings,
  transient bindings, and a motion-history copy callback.
- `WebGPUPostProcessContextMetadata.publishColorTarget` must expose a callback
  that records the pass color output for executor-owned completion.
- `WebGPUPostProcessContextMetadata.publishColorTarget` must not allow a pass
  implementation to mutate `WebGPUFrameExecutor` frame targets directly.
- `WebGPUFrameExecutor` must apply a recorded color output only from
  `IPostProcessExecutor.completePass(request, result)` after the pass completes
  with `result.ran !== false`.
- `WebGPUFrameExecutor` must reject recorded color outputs that are not owned by
  the active frame color target set.
- `WebGPUPostProcessContextMetadata.transients` must map context properties to
  `PostProcessPassRequest.transients` ids.
- `PostProcessPassImplementation.metadata.warmupHints` may list WebGPU runtime
  warmup hint ids.
- WebGPU warmup must collect ordered logical pass descriptors from
  `BackendPostProcessRuntime.compileWarmupGraph(context)`.
- WebGPU warmup must call `PostProcessPassImplementation.warmup(context)` for
  pass-owned implementations when it is present.
- WebGPU warmup must collect runtime hints from
  `PostProcessPassImplementation.metadata.warmupHints`.
- `WebGPUPostProcessExecutor.createGBufferBridge(context)` must return a
  `LogicalGBufferBridge` that wraps WebGPU texture handles.
- WebGPU depth channels must declare `depthEncoding: "hardware"` unless the
  implementation provides a linearized depth texture.
- WebGPU motion channels must declare `motionEncoding: "ndc-delta"` when motion
  vectors are available.
- WebGPU temporal passes must read history resources from `request.histories`.
- WebGPU transient resources must be read from injected context properties
  declared by `WebGPUPostProcessContextMetadata.transients`.
- WebGPU temporal passes must return `updatedHistoryIds` or `historyUpdated`
  when they write runtime-owned history resources.
- The engine-provided `taa`, `fxaa`, `ssao`, `ssr`, `ssrefraction`, and
  `volumetric` WebGPU kernels must be pass-owned implementations.
- The engine-provided WebGPU `ssao` pass must request `ssao:raw` and `ssao:blur`
  transients sized by its resolved `downsample` option.
- The engine-provided WebGPU `ssr` pass must request `ssr:raw` sized by its
  resolved `downsample` option and the shared `hiz` full-chain transient.
- The engine-provided WebGPU `ssrefraction` pass must request
  `ssrefraction:raw` sized by its resolved `downsample` option and the shared
  `hiz` full-chain transient.
- The engine-provided WebGPU `volumetric` pass must request the same shared
  `hiz` full-chain transient as `ssr` and `ssrefraction`.
- `WebGPUFrameTargets` must not contain post-process transient textures such as
  SSAO intermediates, SSR intermediates, or Hi-Z textures.
- WebGPU executor resource allocation must use backend-owned texture creation
  and destruction APIs.
- `WebGPUPostProcessExecutor.createResource(desc)` must create a full mip chain
  when `desc.mipMode` is `"full-chain"`.
- `WebGPUPostProcessExecutor.invalidateResourceBindings()` must invalidate
  post-process binding caches when transient resources are recreated.
- WebGPU backends must not expose public `postProcess`, `postProcessExecutor`,
  `postProcessAdapter`, or `createPostProcessGBufferBridge(context)` members.

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
							| {
									publishColorTarget?: (texture: unknown) => void;
									scratch?: unknown;
									targets?: { postPing?: unknown; sceneColor?: unknown };
							  }
							| undefined;
						const output =
							webgpuContext?.targets?.postPing ??
							webgpuContext?.targets?.sceneColor;
						if (webgpuContext?.scratch && output) {
							webgpuContext.publishColorTarget?.(output);
						}
						return webgpuContext?.scratch && output ?
								{ ran: true }
							:	{ ran: false };
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
bun tests/static/webgpu/test_webgpu_post_graph.mjs
bun tests/static/webgpu/test_webgpu_backend_cache_and_dependency.mjs
bun tests/static/webgpu/test_webgpu_postprocess_runtime_screen.mjs
```

## Errors & Diagnostics
- `renderer.postProcess.registerPass(pass)` must throw when `pass` is not a
  `PostProcessPass`.
- `renderer.postProcess.registerPass(pass)` must throw when `pass.id` is already
  registered.
- `postprocess-requirement-missing-<passId>` must be emitted when the WebGPU
  G-buffer bridge lacks a required semantic channel during execution.
- `postprocess-transient-conflict-<transientId>` must be emitted when eligible
  passes request incompatible transient descriptors.
- WebGPU device allocation failures during `createResource(desc)` must propagate
  as backend resource allocation errors.

## Compatibility / Breaking Changes
- `renderer.postprocess` is no longer a WebGPU backend extension.
- `resolvePostProcessBackendExtension(new WebGPUBackend())` is removed.
- `WebGPUPostProcessPassPlugin` is removed from the public API.
- `WebGPUBackend.postProcess` is removed.
- `WebGPUBackend.postProcess.registerPass(pass)` and
  `WebGPUBackend.postProcess.unregisterPass(id)` are removed.
- `WebGPUBackend.postProcessAdapter`, `WebGPUBackend.postProcessExecutor`, and
  `WebGPUBackend.createPostProcessGBufferBridge(context)` are removed.
- Public custom passes must migrate to `PostProcessPass` and
  `renderer.postProcess.registerPass(pass)`.
- `PostProcessPassDescriptor.dependsOn` is removed. Custom passes must migrate
  to `placement` and optional `order`.
- `WebGPUFrameTargets.aoRaw`, `WebGPUFrameTargets.aoBlur`,
  `WebGPUFrameTargets.ssrRaw`, and `WebGPUFrameTargets.hiZ` are removed.
- WebGPU pass-owned implementations that need temporary textures must declare
  `getTransientResourceDescriptors(request)` and `metadata.context.transients`.
- WebGPU pass-owned implementations must treat `context.targets` as read-only
  and must publish color output through `publishColorTarget(texture)`.
