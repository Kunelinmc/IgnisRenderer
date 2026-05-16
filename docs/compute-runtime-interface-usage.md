# ComputeRuntime Interface Usage

## Scope

This document describes how to use `ComputeRuntime` for WebGPU compute jobs.

`ComputeRuntime` must be constructed from a WebGPU source. Valid sources are
`Renderer` with `WebGPUBackend`, `WebGPUBackend`, `IWebGPUComputeFacade`, or an
object exposing `backend` or `getComputeFacade()`.

## Background

`ComputeRuntime` provides a backend-facing compute abstraction for creating GPU
resources, dispatching WGSL compute kernels, and reading results back to CPU
memory. Texture readback uses WebGPU copy alignment, so callers must decode rows
through `TextureReadbackResult` helpers instead of assuming tightly packed data.

## API/Contract

- `createBuffer(desc)` must create an `IRenderBuffer`.
- `createTexture(desc)` must create an `IRenderTexture`.
- `createSampler(desc)` must create an `ISampler`.
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
} from "../src";

declare const renderer: unknown;

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

const runtime = new ComputeRuntime(renderer);
const pixels = await run(runtime);
runtime.destroy();
console.log(pixels[0]);
```

## Errors & Diagnostics

- `ComputeRuntime requires a webgpuSource that exposes an initialized GPU device and queue.`
  triggers when the source cannot resolve a WebGPU `device` and `queue`.
- Kernel creation must throw for duplicate binding keys, duplicate binding
  indices, or unsupported binding types.
- Dispatch must throw when required resources are missing, resource types do not
  match the schema, or both `dispatch` and `dispatch2D` are supplied.
- `toNormalizedRGBA8Float32()` must throw for non-8-bit readback formats.
- `toRGBAFloat32()` must throw for unsupported texture formats.
- Runtime methods must throw after `destroy()` is called.

## Compatibility / Breaking Changes

`toRGBAFloat32()` is additive. Existing `toNormalizedRGBA8Float32()` callers
remain valid for `RGBA8Unorm` and `BGRA8Unorm` readbacks.
