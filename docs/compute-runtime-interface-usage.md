# ComputeRuntime Interface Usage

This document describes how to use `ComputeRuntime` for WebGPU compute jobs.

## Scope

- `ComputeRuntime` only works with a WebGPU source.
- Valid constructor sources are `Renderer` (with `WebGPUBackend`),
  `WebGPUBackend`, `IWebGPUComputeFacade`, or a resolver source that exposes
  `backend` or `getComputeFacade()`.

## Quick Start

```ts
import {
	BufferUsage,
	ComputeRuntime,
	Renderer,
} from "../src";

// renderer.backend must be WebGPUBackend
// Assume `renderer` is already created and initialized.
const runtime = new ComputeRuntime(renderer as Renderer);

const elementCount = 1024;
const byteLength = elementCount * 4;

const inputBuffer = runtime.createBuffer({
	label: "InputBuffer",
	size: byteLength,
	usage: BufferUsage.Storage | BufferUsage.CopyDst,
});

const outputBuffer = runtime.createBuffer({
	label: "OutputBuffer",
	size: byteLength,
	usage: BufferUsage.Storage | BufferUsage.CopySrc,
});

const paramsBuffer = runtime.createBuffer({
	label: "ParamsBuffer",
	size: 16, // vec4<f32> aligned payload
	usage: BufferUsage.Uniform | BufferUsage.CopyDst,
});

runtime.writeBuffer(inputBuffer, new Float32Array(elementCount).fill(2));
runtime.writeBuffer(paramsBuffer, new Float32Array([elementCount, 3, 0, 0]));

const kernel = await runtime.createKernel({
	label: "ScaleKernel",
	code: `
struct BufferData {
	values: array<f32>,
};

struct Params {
	count: f32,
	factor: f32,
	_pad: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> inputData: BufferData;
@group(0) @binding(1) var<storage, read_write> outputData: BufferData;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn csMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
	let index = globalId.x;
	if (f32(index) >= params.count) {
		return;
	}
	outputData.values[index] = inputData.values[index] * params.factor;
}
`,
	bindings: [
		{ key: "inputData", binding: 0, type: "buffer" },
		{ key: "outputData", binding: 1, type: "buffer" },
		{ key: "params", binding: 2, type: "buffer" },
	],
	workgroupSize: { x: 64, y: 1, z: 1 },
});

const ticket = kernel.dispatch({
	label: "ScaleDispatch",
	resources: {
		inputData: inputBuffer,
		outputData: outputBuffer,
		params: paramsBuffer,
	},
	dispatch: {
		x: Math.ceil(elementCount / 64),
		y: 1,
		z: 1,
	},
});

await ticket.done;

const readback = await runtime.readBuffer({
	buffer: outputBuffer,
	size: byteLength,
});
const result = readback.toFloat32().subarray(0, elementCount);
console.log(result[0]); // 6

kernel.destroy();
runtime.destroy();
```

## Kernel Descriptor Contract

`createKernel(descriptor)` expects:

- `code`: shader source (WGSL by default)
- `entryPoint`: optional, defaults to `csMain`
- `bindings`: required and non-empty
- `workgroupSize`: positive integers

Binding schema rules:

- `key` must be unique.
- `binding` index must be unique and non-negative.
- `type` must be `"buffer"`, `"texture"`, or `"sampler"`.
- `optional: true` allows omitting that resource in `dispatch()`.

## API Reference

Main runtime methods:

- `createBuffer(desc: BufferDesc): IRenderBuffer`
- `createTexture(desc: TextureDesc): IRenderTexture`
- `createSampler(desc: SamplerDesc): ISampler`
- `writeBuffer(buffer, data, offset?)`
- `writeTexture(texture, data, layout, size)`
- `createKernel(descriptor): Promise<ComputeKernel>`
- `readBuffer(options): Promise<BufferReadbackResult>`
- `readTexture(options): Promise<TextureReadbackResult>`
- `destroy(): void`

Kernel methods:

- `dispatch(options: ComputeDispatchOptions): ComputeDispatchTicket`
- `destroy(): void`

## Dispatch Contract

`kernel.dispatch(options)` expects:

- `resources`: keys must match the kernel schema.
- Exactly one dimension mode:
  `dispatch` (`x`, optional `y`, optional `z`) or `dispatch2D` (`width`,
  `height`, optional `depth`).

Notes:

- `dispatch2D` auto-converts to workgroups using `workgroupSize`.
- If both `dispatch` and `dispatch2D` are provided, dispatch throws.
- `ComputeDispatchTicket.done` resolves when submitted GPU work is complete.

## Texture IO

### writeTexture

`writeTexture(texture, data, layout, size)` requires:

- `layout.bytesPerRow` is a positive number.
- `size.width/height/depthOrArrayLayers` are positive.

### readTexture

`readTexture({ texture, ... })` returns:

- `bytes`: raw data buffer
- `width`, `height`, `format`
- `bytesPerPixel`, `bytesPerRow`
- `toFloat32()`
- `toNormalizedRGBA8Float32()`

Notes:

- `toNormalizedRGBA8Float32()` only supports `TextureFormat.RGBA8Unorm`.
- `readTexture()` aligns `bytesPerRow` to 256 for GPU copy requirements.

## Resource Lifetime

- `ComputeRuntime` tracks owned buffers/textures/samplers/modules/pipelines.
- Calling `.destroy()` on a tracked resource during in-flight dispatch is safe.
- Real GPU destruction is deferred until all related submitted work finishes.
- `runtime.destroy()` destroys all kernels and tracked resources.

## Common Errors

- Non-WebGPU source:
  `ComputeRuntime requires a webgpuSource that exposes an initialized GPU device and queue.`
- Missing dispatch dimensions:
  `Compute dispatch options require either dispatch or dispatch2D.`
- Invalid schema:
  duplicate keys, duplicate bindings, unsupported binding type.
- Resource mismatch:
  `Compute resource "<key>" expects "<type>" but received "<actual>".`
