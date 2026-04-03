# Compute Runtime Abstraction Contract

## Scope

This document defines the abstract contracts `IComputeRuntime` and
`IComputeKernel` in `src/renderers/IComputeRuntime.ts`.
Implementations must satisfy these contracts to support compute workflows.

## Background

`ComputeRuntime` was originally consumed as a concrete WebGPU class. Future
runtime implementations require a shared contract so pipeline code can depend
on capabilities rather than backend-specific classes.

## API/Contract

- `IComputeRuntime` must provide resource creation APIs:
  `createBuffer`, `createTexture`, `createSampler`.
- `IComputeRuntime` must provide data upload APIs:
  `writeBuffer`, `writeTexture`.
- `IComputeRuntime` must provide kernel lifecycle APIs:
  `createKernel`, `destroy`.
- `IComputeRuntime` must provide readback APIs:
  `readBuffer`, `readTexture`.
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

```ts
import type { IComputeRuntime } from "../src/renderers/IComputeRuntime";
import { ComputeRuntime } from "../src/renderers/webgpu/ComputeRuntime";

function createRuntime(webgpuSource: unknown): IComputeRuntime {
	return new ComputeRuntime(webgpuSource as any);
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

## Errors & Diagnostics

- `IComputeRuntime` implementations should throw when kernel descriptors are
  invalid (for example: duplicate binding keys or unsupported binding types).
- `dispatch` should throw when required resources are missing or type-mismatched.
- `dispatch` should throw when both `dispatch` and `dispatch2D` are set.
- Runtime methods should throw after `destroy()` is called.

## Compatibility / Breaking Changes

This change is additive and non-breaking.
Existing `ComputeRuntime` usage remains valid.
