# ComputeRuntime Interface Usage

## Scope

This document describes how to use `ComputeRuntime` for WebGPU compute jobs.

`ComputeRuntime` must be constructed from a WebGPU compute source. Valid sources
are `WebGPUBackend`, `IWebGPUComputeFacade`, or a WebGPU backend-like object
that exposes the compute, texture, command submission, `device`, and `queue`
surface required by `WebGPUComputeFacadeSource`. `Renderer` instances and
renderer-like `{ backend }` wrappers are not valid sources.
When a source is an `IWebGPUComputeFacade`, its `device`, `queue`, and
`createComputePipeline(desc)` members must be available, and
`createComputePipeline(desc)` must return `Promise<IComputePipeline>`.

## Background

`ComputeRuntime` provides a backend-facing compute abstraction for creating GPU
resources, dispatching WGSL compute kernels, and reading results back to CPU
memory. Texture readback uses WebGPU copy alignment, so callers must decode rows
through `TextureReadbackResult` helpers instead of assuming tightly packed data.

## API/Contract

- `createBuffer(desc)` must create an `IRenderBuffer`.
- `createTexture(desc)` must create an `IRenderTexture`.
- `createSampler(desc)` must create an `ISampler`.
- WebGPU compute facade sources must implement
  `createComputePipeline(desc): Promise<IComputePipeline>`.
- WebGPU compute sources must expose an initialized WebGPU `device` and `queue`
  before `writeTexture`, `readBuffer`, `readTexture`, or dispatch completion
  tracking can be used.
- `ComputeRuntime` must not resolve `Renderer` instances, renderer-like
  `{ backend }` wrappers, or recursive source objects.
- `writeBuffer(buffer, data, offset?)` must upload buffer bytes.
- `writeTexture(texture, data, layout, size)` must upload texture bytes and
  requires positive `layout.bytesPerRow`.
- `createKernel(descriptor)` must receive unique binding keys and binding
  indices, and each binding `type` must be `"buffer"`, `"texture"`, or
  `"sampler"`.
- `dispatch(options)` must provide resources matching the kernel schema and
  exactly one of `dispatch` or `dispatch2D`.
- `readBuffer(options)` must return `BufferReadbackResult`.
- `readTexture(options)` must return `TextureReadbackResult` with `bytes`,
  `width`, `height`, `format`, `bytesPerPixel`, `bytesPerRow`, `toFloat32()`,
  `toRGBAFloat32()`, and `toNormalizedRGBA8Float32()`.
- `toRGBAFloat32()` must decode `RGBA8Unorm`, `BGRA8Unorm`, and `RGBA16Float`
  readbacks to an unpadded `Float32Array` in RGBA order.
- `toNormalizedRGBA8Float32()` must only support `RGBA8Unorm` and `BGRA8Unorm`.
- `destroy()` must release owned runtime resources.

## Usage

```ts
import {
	BufferUsage,
	ComputeRuntime,
	TextureFormat,
	TextureUsage,
	type IComputeRuntime,
	type WebGPUBackend,
} from "../src";
import { WEBGPU_COMPUTE_EXTENSION } from "../src/renderers/BackendExtensions";

declare const backend: WebGPUBackend;

async function run(runtime: IComputeRuntime): Promise<Float32Array> {
	const texture = runtime.createTexture({
		label: "OutputTexture",
		width: 16,
		height: 16,
		format: TextureFormat.RGBA16Float,
		usage: TextureUsage.StorageBinding | TextureUsage.CopySrc,
	});
	const params = runtime.createBuffer({
		label: "Params",
		size: 16,
		usage: BufferUsage.Uniform | BufferUsage.CopyDst,
	});

	runtime.writeBuffer(params, new Float32Array([16, 16, 0, 0]));

	const kernel = await runtime.createKernel({
		label: "FillTexture",
		code: `
@group(0) @binding(0) var outTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: vec4<f32>;

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) id: vec3<u32>) {
	if (id.x >= u32(params.x) || id.y >= u32(params.y)) {
		return;
	}
	textureStore(outTex, vec2<i32>(id.xy), vec4<f32>(2.0, 1.0, 0.5, 1.0));
}
`,
		bindings: [
			{ key: "outTex", binding: 0, type: "texture" },
			{ key: "params", binding: 1, type: "buffer" },
		],
		workgroupSize: { x: 8, y: 8, z: 1 },
	});

	await kernel.dispatch({
		resources: { outTex: texture, params },
		dispatch2D: { width: 16, height: 16 },
	}).done;

	const readback = await runtime.readTexture({
		texture,
		format: TextureFormat.RGBA16Float,
	});
	const pixels = readback.toRGBAFloat32();

	kernel.destroy();
	texture.destroy();
	params.destroy();
	return pixels;
}

const runtime = new ComputeRuntime(backend);
const pixels = await run(runtime);
runtime.destroy();
console.log(pixels[0]);
```

Applications that only have a `Renderer` must resolve the backend-owned compute
extension outside `ComputeRuntime`, then construct the runtime from that
extension:

```ts
const compute = renderer.requireBackendExtension(WEBGPU_COMPUTE_EXTENSION);
const runtime = new ComputeRuntime(compute);
```

## Errors & Diagnostics

- `ComputeRuntime requires a webgpuSource that exposes an initialized GPU device and queue.`
  triggers when the source cannot resolve a WebGPU `device` and `queue`.
- `Failed to resolve WebGPU compute facade from provided source.` triggers when
  a renderer-like wrapper or incomplete backend-like source is passed.
- Kernel creation must throw for duplicate binding keys, duplicate binding
  indices, or unsupported binding types.
- Dispatch must throw when required resources are missing, resource types do not
  match the schema, or both `dispatch` and `dispatch2D` are supplied.
- `toNormalizedRGBA8Float32()` must throw for non-8-bit readback formats.
- `toRGBAFloat32()` must throw for unsupported texture formats.
- Runtime methods must throw after `destroy()` is called.

## Compatibility / Breaking Changes

`IWebGPUComputeFacade.createComputePipeline(desc)` now returns
`Promise<IComputePipeline>`. Consumers that call the facade directly must
`await` pipeline creation before creating bind groups, dispatching kernels, or
passing the pipeline to WebGPU helper code.

`toRGBAFloat32()` remains additive. Existing `toNormalizedRGBA8Float32()`
callers remain valid for `RGBA8Unorm` and `BGRA8Unorm` readbacks.

`ComputeRuntime` no longer accepts `Renderer` instances or renderer-like
`{ backend }` source wrappers. Callers must pass a `WebGPUBackend`, a
backend-owned `IWebGPUComputeFacade`, or another direct
`WebGPUComputeFacadeSource`.
