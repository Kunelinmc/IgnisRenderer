# Compute Runtime

Use `ComputeRuntime` to create GPU resources, dispatch application-defined WGSL kernels, and read results through the WebGPU backend.

## Overview

Use `ComputeRuntime` to run application-defined WGSL compute work on the WebGPU
backend. It provides a compact API for creating GPU resources, uploading data,
dispatching kernels, and reading results back into JavaScript.

This guide covers the public `ComputeRuntime` workflow. It is intended for
WebGPU applications that need general-purpose GPU work alongside rendering.

A compute job usually follows this lifecycle:

1. Initialize a `WebGPUBackend`.
2. Create a `ComputeRuntime` from that backend.
3. Create the buffers, textures, and samplers needed by the job.
4. Compile a kernel with `createKernel()`.
5. Dispatch the kernel and await its completion ticket.
6. Read back any CPU-visible results.
7. Destroy the runtime when the compute resources are no longer needed.

`ComputeRuntime` implements `IComputeRuntime`, so application code can depend
on the interface while using the WebGPU implementation today.

WebGL applications may use `WEBGL_AUXILIARY_RASTER_EXTENSION` for scheduled
offscreen raster work. The facade supplies scope-owned resources and a safe
encoder without exposing `WebGL2RenderingContext` or native WebGL resources.

## API

### WebGL auxiliary raster

Resolve `WEBGL_AUXILIARY_RASTER_EXTENSION` from an initialized WebGL backend or
renderer. `execute()` schedules one scoped task. Resources created by the task
are valid only until its callback settles and are then destroyed automatically.

Requests default to `idle-only` frame scheduling and reject on context loss.
Use `between-passes` only for work that may safely run at an active frame
boundary, and use `retain-pending` only when replay against a restored context
generation is valid. Required WebGL extensions are revalidated immediately
before execution.

```ts
import {
	Renderer,
	WEBGL_AUXILIARY_RASTER_EXTENSION,
	WebGLBackend,
} from "ignisrenderer";

const backend = new WebGLBackend();
const renderer = new Renderer(canvas, backend);
await renderer.initialize();

const raster = renderer.requireBackendExtension(
	WEBGL_AUXILIARY_RASTER_EXTENSION,
);
await raster.execute({
	label: "offscreen-raster",
	task: async ({ encoder, resources }) => {
		// Create scoped resources, encode raster work, and return CPU data.
	},
});
```

### Compute workflow

#### Creating a runtime

Construct `ComputeRuntime` with an initialized `WebGPUBackend` or an
`IWebGPUComputeFacade`. Keep a reference to the backend when you create your
`Renderer`; the renderer itself is not a compute source.

```ts
import {
	ComputeRuntime,
	Renderer,
	WebGPUBackend,
} from "ignisrenderer";

const backend = new WebGPUBackend();
const renderer = new Renderer(canvas, backend);
await renderer.initialize();

const runtime = new ComputeRuntime(backend);
```

The backend must finish initialization before the runtime is created, because
compute submission and readback require an active WebGPU device and queue.

#### Resources and uploads

- `createBuffer()`, `createTexture()`, and `createSampler()` create resources
  owned by the runtime.
- `writeBuffer()` uploads bytes to a buffer and supports an optional byte
  offset.
- `writeTexture()` uploads texture data. Supply a positive `bytesPerRow` that
  matches the uploaded layout.

Choose usage flags for every operation you plan to perform. For example, a
texture written by a kernel and later read by JavaScript needs both
storage-binding and copy-source usage.

#### Kernels and dispatch

`createKernel()` compiles WGSL asynchronously and returns an `IComputeKernel`.
Its binding schema describes the resources managed in bind group 0:

- Every `key` and binding index is unique.
- Each binding type is `"buffer"`, `"texture"`, or `"sampler"`.
- The resources passed to `dispatch()` use the same keys and compatible types.
- The default WGSL entry point is `csMain`.
- `workgroupSize` should match the shader's `@workgroup_size`.

Choose one dispatch form:

- `dispatch: { x, y, z }` supplies workgroup counts directly.
- `dispatch2D: { width, height, depth? }` supplies logical dimensions.
  `ComputeRuntime` converts width and height to workgroup counts using the
  kernel's `workgroupSize`.

`dispatch()` returns a `ComputeDispatchTicket`. Await `ticket.done` before
reading results or releasing resources used by that dispatch.

#### Readback and cleanup

`readBuffer()` returns raw bytes and a `toFloat32()` helper.

`readTexture()` returns the raw WebGPU copy layout together with image
dimensions and conversion helpers. Prefer `toRGBAFloat32()` when you need a
tightly packed RGBA array; it removes row padding and supports
`RGBA8Unorm`, `BGRA8Unorm`, and `RGBA16Float`.

`toNormalizedRGBA8Float32()` is available for `RGBA8Unorm` and `BGRA8Unorm`
when all channels should be normalized to the 0–1 range.

Call `destroy()` on short-lived kernels, buffers, or textures when they are no
longer needed. Calling `runtime.destroy()` releases everything still owned by
the runtime, including samplers, and makes the runtime unavailable for further
work.

## Usage

### Compute workflow

The following minimal job shows runtime setup, kernel creation, binding, and
dispatch using only exports from the package entry point:

```ts
import {
	ComputeRuntime,
	Renderer,
	WebGPUBackend,
	type IComputeRuntime,
} from "ignisrenderer";

async function runKernel(runtime: IComputeRuntime): Promise<void> {
	const sampler = runtime.createSampler({
		label: "ComputeSampler",
	});

	const kernel = await runtime.createKernel({
		label: "ComputeExample",
		code: `
@group(0) @binding(0) var computeSampler: sampler;

@compute @workgroup_size(1, 1, 1)
fn csMain() {}
`,
		bindings: [
			{ key: "computeSampler", binding: 0, type: "sampler" },
		],
		workgroupSize: { x: 1, y: 1, z: 1 },
	});

	const ticket = kernel.dispatch({
		resources: { computeSampler: sampler },
		dispatch: { x: 1, y: 1, z: 1 },
	});
	await ticket.done;

	kernel.destroy();
}

const backend = new WebGPUBackend();
const renderer = new Renderer(canvas, backend);
await renderer.initialize();

const runtime = new ComputeRuntime(backend);
try {
	await runKernel(runtime);
} finally {
	runtime.destroy();
	await renderer.destroy();
}
```

## Troubleshooting

### Compute workflow

- The runtime reports that it needs an initialized device and queue: call
  `renderer.initialize()` before constructing the runtime, and pass the
  initialized backend rather than the renderer.
- Kernel creation reports a duplicate key or binding: make every schema key and
  binding index unique, and match the WGSL declarations.
- Dispatch reports a missing or mismatched resource: compare each
  `resources` entry with the kernel's binding schema.
- Dispatch rejects its dimensions: provide exactly one of `dispatch` or
  `dispatch2D`, using positive integers.
- Texture upload rejects `bytesPerRow`: provide a positive row stride that
  matches the source data and WebGPU's requirements for the copy.
- Texture conversion rejects the format: use `toRGBAFloat32()` only with a
  supported format, or inspect `readback.bytes` and `bytesPerRow` directly.
- A method reports that the runtime or kernel is destroyed: create a new
  runtime or kernel instead of reusing a released instance.

## Compatibility

### Compute workflow

`ComputeRuntime` accepts direct WebGPU compute sources. It no longer unwraps a
`Renderer` or an object shaped like `{ backend }`. Applications should keep
the `WebGPUBackend` reference used to construct the renderer and pass that
backend to `ComputeRuntime`.

Custom `IWebGPUComputeFacade` implementations return
`Promise<IComputePipeline>` from `createComputePipeline()`. Await pipeline
creation before using pipeline-dependent resources.

Existing `toNormalizedRGBA8Float32()` calls remain supported for
`RGBA8Unorm` and `BGRA8Unorm`. Use `toRGBAFloat32()` when code also needs
`RGBA16Float` readback or a single format-aware RGBA conversion path.

## Related Documents

- [Renderer](renderer.md)
- [Compute contract](../contracts/compute.md)
- [WebGPU contract](../contracts/webgpu.md)
