# Compute Contract

This document defines the backend-agnostic compute runtime and kernel contracts used by application-defined GPU compute workflows.

## Contract

### Compute runtime abstraction

- `IComputeRuntime` must provide resource creation APIs:
  `createBuffer`, `createTexture`, `createSampler`.
- `IComputeRuntime` must provide data upload APIs:
  `writeBuffer`, `writeTexture`.
- `IComputeRuntime` must provide kernel lifecycle APIs:
  `createKernel`, `destroy`.
- `IComputeRuntime` must provide readback APIs:
  `readBuffer`, `readTexture`.
- WebGPU facade-backed `IComputeRuntime` implementations must await
  `IWebGPUComputeFacade.createComputePipeline(desc)` before exposing a created
  `IComputeKernel`.
- WebGPU `ComputeRuntime` sources must be direct `WebGPUComputeFacadeSource`
  values, such as `WebGPUBackend`, `IWebGPUComputeFacade`, or backend-like
  objects with the required WebGPU compute, `device`, and `queue` members.
- WebGPU `ComputeRuntime` implementations must not resolve `Renderer`
  instances, renderer-like `{ backend }` wrappers, or recursive source chains.
- `TextureReadbackResult` must expose raw `bytes`, dimensions, row layout,
  `toFloat32`, `toRGBAFloat32`, and `toNormalizedRGBA8Float32`.
- `toRGBAFloat32` must decode `RGBA8Unorm`, `BGRA8Unorm`, and `RGBA16Float`
  texture readbacks to linear or normalized RGBA `Float32Array` data while
  skipping GPU row padding.
- `createKernel` must return `Promise<IComputeKernel>`.
- `IComputeKernel` must expose `label`, `bindings`, and `workgroupSize`.
- `IComputeKernel` must provide `dispatch` and `destroy`.
- `dispatch` input must follow `ComputeDispatchOptions`:
  - `resources` keys must match declared binding schema keys.
  - Caller must supply exactly one dispatch mode:
    `dispatch` or `dispatch2D`.
- `IComputeRuntime` implementations must enforce runtime validity checks and
  throw deterministic errors for invalid contracts.

## Usage

### Compute runtime abstraction

```ts
import type { IComputeRuntime } from "../src/backends/IComputeRuntime";
import { ComputeRuntime } from "../src/backends/webgpu/ComputeRuntime";
import type { WebGPUComputeFacadeSource } from "../src/backends/webgpu/ComputeFacade";

function createRuntime(source: WebGPUComputeFacadeSource): IComputeRuntime {
	return new ComputeRuntime(source);
}

async function run(runtime: IComputeRuntime): Promise<void> {
	const kernel = await runtime.createKernel({
		code: "@compute @workgroup_size(1) fn csMain() {}",
		bindings: [{ key: "params", binding: 0, type: "buffer" }],
		workgroupSize: { x: 1 },
	});
	kernel.destroy();
	runtime.destroy();
}
```

## Diagnostics

### Compute runtime abstraction

- `IComputeRuntime` implementations should throw when kernel descriptors are
  invalid (for example: duplicate binding keys or unsupported binding types).
- `dispatch` should throw when required resources are missing or type-mismatched.
- `dispatch` should throw when both `dispatch` and `dispatch2D` are set.
- Runtime methods should throw after `destroy()` is called.
- WebGPU `ComputeRuntime` should throw when a source does not expose an
  initialized WebGPU `device` and `queue`.

## Verification

```bash
bunx tsc --noEmit
```

## Related Documents

- [Compute runtime guide](../public/compute-runtime.md)
- [Renderer contract](renderer.md)
- [WebGPU contract](webgpu.md)
