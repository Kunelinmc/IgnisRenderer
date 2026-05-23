# WebGPU Post-Process Extension Contract
## Scope
This document defines the WebGPU-specific behavior behind the cross-backend post-process contract.

## Background
WebGPU post-processing is now driven through `PostProcessPipeline` and `IPostProcessExecutor`. The public extension point is `renderer.postProcess.registerPass(descriptor)`. WebGPU-specific runtime objects remain internal implementation details used by `WebGPUPostProcessExecutor` and frame delegates.

## API/Contract
- `renderer.postProcess.registerPass(descriptor)` must register a logical `PostProcessPassDescriptor`.
- A WebGPU custom pass must include `descriptor.implementations.webgpu`.
- A WebGPU custom pass should use `descriptor.placement` and optional `descriptor.order` to enter the fixed post-process sequence.
- `WebGPUBackend.postProcessExecutor.backend` must be `"webgpu"`.
- `WebGPUBackend.postProcessExecutor.executePass(passId, request)` must dispatch backend-owned fallback post-process passes.
- `WebGPUBackend.postProcessExecutor.getPassExecutionContext(passId, request)` may provide low-level helpers for pass-owned WebGPU implementations.
- Pass-owned WebGPU implementations must use `PostProcessPassImplementation.execute(request, context)` instead of WebGPU runtime registration.
- WebGPU warmup must call `PostProcessPassImplementation.warmup(context)` for pass-owned implementations when it is present.
- `WebGPUBackend.createPostProcessGBufferBridge(context)` must return a `LogicalGBufferBridge` that wraps WebGPU texture handles.
- WebGPU depth channels must declare `depthEncoding: "hardware"` unless the implementation provides a linearized depth texture.
- WebGPU motion channels must declare `motionEncoding: "ndc-delta"` when motion vectors are available.
- WebGPU temporal passes must read history resources from `request.histories`.
- WebGPU temporal passes must return `updatedHistoryIds` or `historyUpdated` when they write pipeline-owned history resources.
- The built-in `taa`, `fxaa`, and `ssr` WebGPU kernels must be pass-owned implementations.
- WebGPU executor resource allocation must use backend-owned texture creation and destruction APIs.
- WebGPU backends must not expose a public `postProcess` facade or backend-level post-process registration methods.

## Usage
```ts
import type { PostProcessPassDescriptor } from "ignisrenderer";

const descriptor: PostProcessPassDescriptor = {
	id: "custom-webgpu-edge",
	placement: "ldr",
	order: 5,
	incremental: {
		firstPass: "custom-webgpu-edge",
		grade: "light",
		inflationRadius: 2,
	},
	isEnabled(state) {
		return state.enabled["custom-webgpu-edge"] === true;
	},
	implementations: {
		webgpu: {
			id: "custom-webgpu-edge",
		},
	},
};

renderer.postProcess.registerPass(descriptor);
renderer.postProcess.enable("custom-webgpu-edge");
```

```bash
bun tests/test_webgpu_postprocess_runtime_spatial.mjs
bun tests/test_webgpu_postprocess_runtime_temporal.mjs
bun tests/test_webgpu_postprocess_runtime_screen.mjs
```

## Errors & Diagnostics
- `Unknown post-process pass "<id>".` must be thrown when `renderer.postProcess.enable(id)` is called before `renderer.postProcess.registerPass(descriptor)`.
- `postprocess-requirement-missing-<passId>` must be emitted when the WebGPU G-buffer bridge lacks a required semantic channel.
- WebGPU device allocation failures during `createResource(desc)` must propagate as backend resource allocation errors.

## Compatibility / Breaking Changes
- `WebGPUPostProcessPassPlugin` is removed from the public API.
- `WebGPUBackend.postProcess` is removed.
- `WebGPUBackend.postProcess.registerPass(pass)` and `WebGPUBackend.postProcess.unregisterPass(id)` are removed.
- Public custom passes must migrate to `PostProcessPassDescriptor` and `renderer.postProcess.registerPass(descriptor)`.
- `PostProcessPassDescriptor.dependsOn` is removed. Custom passes must migrate to `placement` and optional `order`.
