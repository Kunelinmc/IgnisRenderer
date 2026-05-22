# WebGPU Post-Process Extension Contract
## Scope
This document defines the WebGPU-specific behavior behind the cross-backend post-process contract.

## Background
WebGPU post-processing is now driven through `PostProcessPipeline` and `IPostProcessExecutor`. The public extension point is `renderer.postProcess.registerPass(descriptor)`. WebGPU-specific runtime objects remain internal implementation details used by `WebGPUPostProcessExecutor` and frame delegates.

## API/Contract
- `renderer.postProcess.registerPass(descriptor)` must register a logical `PostProcessPassDescriptor`.
- A WebGPU custom pass must include `descriptor.implementations.webgpu`.
- `WebGPUBackend.postProcess.executor.backend` must be `"webgpu"`.
- `WebGPUBackend.postProcess.executor.executePass(passId, request)` must dispatch the corresponding WebGPU post-process implementation.
- `WebGPUBackend.postProcess.createGBufferBridge(context)` must return a `LogicalGBufferBridge` that wraps WebGPU texture handles.
- WebGPU depth channels must declare `depthEncoding: "hardware"` unless the implementation provides a linearized depth texture.
- WebGPU motion channels must declare `motionEncoding: "ndc-delta"` when motion vectors are available.
- WebGPU temporal passes must read history resources from `request.histories`.
- WebGPU temporal passes must return `updatedHistoryIds` or `historyUpdated` when they write pipeline-owned history resources.
- WebGPU executor resource allocation must use backend-owned texture creation and destruction APIs.
- WebGPU backends must not expose public `postProcess.registerPass` or `postProcess.unregisterPass`.

## Usage
```ts
import type { PostProcessPassDescriptor } from "ignisrenderer";

const descriptor: PostProcessPassDescriptor = {
	id: "custom-webgpu-edge",
	dependsOn: ["tonemap"],
	incremental: {
		firstPass: "tonemap",
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
- `postprocess-dependency-missing-<passId>-<dependencyId>` must be emitted when a logical descriptor references an unknown dependency.
- `postprocess-cycle-<passId>` must be emitted when logical descriptors form a cycle.
- WebGPU device allocation failures during `createResource(desc)` must propagate as backend resource allocation errors.

## Compatibility / Breaking Changes
- `WebGPUPostProcessPassPlugin` is removed from the public API.
- `WebGPUBackend.postProcess.registerPass(pass)` is removed.
- `WebGPUBackend.postProcess.unregisterPass(id)` is removed.
- Public custom passes must migrate to `PostProcessPassDescriptor` and `renderer.postProcess.registerPass(descriptor)`.
