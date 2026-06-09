# WebGL Early Z Pre-Pass Contract

## Scope

This document defines the WebGL backend Early Z pre-pass contract for opaque
scene rendering, built-in material alpha masking, and `ShaderMaterial`
opt-in depth fragments.

## Background

The WebGL backend renders `main-opaque` as a backend-internal sequence. When
Early Z pre-pass is enabled, WebGL first writes eligible opaque geometry to the
scene depth buffer, then runs the existing color pass against that depth buffer.
This optimization must not add a renderer-level frame stage.

## API/Contract

`WebGLBackendOptions.enableEarlyZPrepass` may be set to `false` to disable the
optimization. The default must be `true`.

`WebGLBackend.isEarlyZPrepassEnabled()` must return the resolved backend option.

The WebGL backend capability `occlusionCulling` must remain `false`.

`ShaderMaterial.chunks` may contain a WebGL depth pre-pass chunk with
`backend: "webgl"`, `language: "glsl"`, and `stage: "fragment-depth"`.

`ShaderMaterial.resolveWebGLDepthPrepassProgram(mode, options)` must return a
custom WebGL vertex and depth fragment program when both chunks are available.
It must return `null` when the material has not opted in.

Built-in material depth pre-pass fragments must only apply alpha mask discard.
The discard test must use `uBaseColor.a * texture(uBaseMap, uv).a < uAlpha.x`
when `uHasBaseMap == 1`, and `uBaseColor.a < uAlpha.x` otherwise.

WebGL Early Z pre-pass eligibility must require opaque rendering,
`depthWrite !== false`, triangle topology, no skeleton, and a finite world
matrix. Transparent objects, OIT passes, particles, and environment rendering
must not participate.

The depth pre-pass must use depth writes, `LESS`, disabled blending, and disabled
color writes. The following color pass must use `LEQUAL` and read-only depth only
for packet ids submitted by the depth pre-pass.

## Usage

```ts
import { WebGLBackend } from "../src/renderers/WebGLBackend";

const backend = new WebGLBackend({
	enableEarlyZPrepass: false,
});

console.log(backend.isEarlyZPrepassEnabled());
```

```ts
import { ShaderMaterial } from "../src/materials/ShaderMaterial";

const material = new ShaderMaterial({
	chunks: [
		{
			backend: "webgl",
			language: "glsl",
			stage: "vertex",
			code: "#version 300 es\nvoid main() { gl_Position = vec4(0.0); }",
		},
		{
			backend: "webgl",
			language: "glsl",
			stage: "fragment-depth",
			code: "#version 300 es\nprecision highp float;\nvoid main() {}",
		},
	],
});

const depthProgram = material.resolveWebGLDepthPrepassProgram("single");
```

## Errors & Diagnostics

`ShaderMaterial.resolveWebGLDepthPrepassProgram()` must return `null` when WebGL
vertex or `fragment-depth` source is missing.

A WebGL `ShaderMaterial` using `AlphaMode.Mask` without a WebGL
`fragment-depth` chunk should warn once and skip Early Z pre-pass for that
material.

Custom WebGL depth pre-pass compile failures should throw in strict shader mode.
In warn mode, the backend should warn once and skip Early Z pre-pass for that
material.

## Compatibility / Breaking Changes

N/A
