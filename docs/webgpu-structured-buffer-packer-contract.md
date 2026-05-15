# WebGPU Structured Buffer Packer Contract

## Scope

This document defines the public contract for the WebGPU structured buffer
packer exported from `src/renderers/webgpu/index.ts`.

## Background

`StructuredBufferLayout` computes WGSL-compatible byte layout for uniform,
storage, and vertex address spaces. The structured buffer packer adds a
declarative field mapping layer on top of that layout without changing layout
or writer ownership.

## API/Contract

- `createStructuredBufferPacker<TInput, TOutput>(options)` must create a
  reusable packer for a `StructuredBufferLayout`.
- `StructuredBufferPacker<TInput, TOutput>.pack(input)` must allocate a writer,
  apply all field rules, and return the configured output view.
- `StructuredBufferPacker<TInput, TOutput>.packInto(writer, input)` must reuse
  the provided writer and return the configured output view.
- `StructuredBufferPacker<TInput, TOutput>.createWriter()` must create a writer
  sized to the packer layout.
- `StructuredBufferPackerOutput` must support `"arrayBuffer"` and
  `"float32Array"`.
- `clearBeforePack` must default to `true`.
- `output` must default to `"arrayBuffer"`.
- Field resolvers that return `null` or `undefined` must skip the write.
- `arrayVec4(path, length, resolver)` must write element values to
  `[path, index]`.
- `arrayStruct(path, length, elementResolver, fields)` must write nested fields
  below `[path, index]`.
- `custom(label, writerCallback)` may perform arbitrary writer operations.
- Packer construction must validate declared field paths when the helper can
  identify a path. Runtime type and length validation must remain owned by
  `StructuredBufferWriter`.

## Usage

```ts
import {
	StructuredBufferLayout,
	arrayOf,
	mat4x4f32,
	structOf,
	vec,
} from "../src/renderers/webgpu/StructuredBufferLayout";
import {
	arrayVec4,
	createStructuredBufferPacker,
	mat4,
	vec4,
} from "../src/renderers/webgpu";

const layout = new StructuredBufferLayout(
	structOf([
		{ name: "modelMatrix", type: mat4x4f32() },
		{ name: "baseColorFactor", type: vec(4, "f32") },
		{ name: "textureTransformA", type: arrayOf(vec(4, "f32"), 2) },
	]),
	"uniform"
);

const packer = createStructuredBufferPacker({
	label: "ExampleModelUniforms",
	layout,
	output: "float32Array",
	fields: [
		mat4("modelMatrix", (input) => input.modelMatrix),
		vec4("baseColorFactor", (input) => input.material.baseColorFactor),
		arrayVec4("textureTransformA", 2, (input, index) =>
			input.material.textureSlots[index]?.transformA
		),
	],
});

const packed = packer.pack({
	modelMatrix: [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	],
	material: {
		baseColorFactor: [1, 1, 1, 1],
		textureSlots: [{ transformA: [0, 0, 1, 1] }],
	},
});
```

## Errors & Diagnostics

- Unknown field path: triggered when a helper path does not exist in the
  `StructuredBufferLayout`.
- Invalid array length: triggered when `arrayVec4` or `arrayStruct` receives a
  negative or non-integer `length`.
- Writer byte-size mismatch: triggered when `packInto` receives a writer that
  was not created for the same layout byte size.
- Invalid scalar, vector, or matrix value: triggered by `StructuredBufferWriter`
  when a resolver returns an incompatible value.

## Compatibility / Breaking Changes

The packer API is additive. Existing `packFrameUniformData` and
`packModelUniformData` call signatures remain valid.
