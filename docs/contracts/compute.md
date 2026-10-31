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

### WebGPU texture creation

- Concrete WebGPU `ComputeRuntime.createTexture(desc)` must accept either an
  allocation `TextureDesc` or a source descriptor containing `source: Texture`,
  optional `usage`, and optional `label`. Source descriptors must not be required
  by the backend-agnostic `IComputeRuntime` or lower-level allocation APIs.
- Source descriptors must reject allocation fields and other unsupported fields
  before allocation. Source presence must select source initialization; a missing
  or invalid `source` value must not fall back to allocation.
- Allocation descriptors must retain their explicit dimensions, format, usage,
  layer, sample-count, mip, and view configuration without adding upload flags.
- Source initialization must derive dimensions, sampling format, and authored mip
  count from `source`. It must create an independent runtime-owned single-layer
  2D snapshot without generating additional mips or registering it in the material
  texture registry. Invalid source dimensions or layer counts must reject before
  allocation; dimensions must be positive integers and each mip must have one layer.
- `usage` must describe requested resource uses. Source descriptors must default
  to `TextureBinding` only when `usage` is omitted. `CopyDst` must always be added
  for source upload, and external image sources must also add `RenderAttachment`.
  Explicit `usage` must replace the sampling default. Known incompatible source
  format/usage combinations must reject instead of removing requested flags.
  Render-attachment capability and enabled feature requirements must be checked
  before allocation, including usage added for external source initialization.
  Depth/stencil source snapshots must support only `depth16unorm` and `stencil8`;
  other depth/stencil formats must reject before allocation because their upload
  requires an unsupported host copy or explicit aspect selection.
- Source snapshots must use the resource manager's source upload operation,
  including raw layout validation, format conversion, and external image copies.
  Native format fallback must not discard sRGB decoding or reinterpret compressed
  blocks as uncompressed pixels.
  Tightly packed native raw float16/float32 sources must support conversion
  between the R, RG, and RGBA floating-point formats defined by the WebGPU upload
  contract. Unsupported raw conversions must throw before queue submission.
- Source changes after the call must not automatically update the snapshot.
  The snapshot must support explicit `destroy()` and runtime-owned cleanup.
  Both descriptor forms must share allocation, ownership tracking, and actual
  format metadata recording. Any failure after allocation must release the new
  resource before throwing.

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
