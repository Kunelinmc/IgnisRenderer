# ShaderMaterial Custom Uniform Contract
## Scope
This document defines the v1 custom numeric uniform contract for
`ShaderMaterial` custom shader chunks in `WebGPUBackend` and `WebGLBackend`.
`SoftwareBackend` must retain custom uniform data on the material but does not
execute custom shader uniform logic in v1.

## Background
`ShaderMaterial` already supports backend-specific shader chunks and custom
texture bindings. Custom numeric uniforms provide the matching scalar, vector,
and matrix value path for user-authored shader code without requiring users to
manage backend-specific bind group indices or WebGL uniform declarations.

## API/Contract
- `ShaderMaterialParams.uniformBindings` may declare custom numeric uniforms.
- `ShaderMaterial.setUniformBindings(bindings)` must replace the complete
  custom uniform schema.
- `ShaderMaterial.setUniformBinding(binding)` must add or replace one custom
  uniform binding.
- `ShaderMaterial.setUniform(name, value)` must update only the value of an
  existing binding and must not increment `shaderRevision`.
- `ShaderMaterial.getUniformBindings()` must return resolved copies in
  declaration order.
- `ShaderMaterial.removeUniformBinding(name)` and
  `ShaderMaterial.clearUniformBindings()` must remove schema entries and must
  increment `shaderRevision` when a schema entry is removed.
- `ShaderMaterialUniformBinding.type` must be one of `f32`, `i32`, `u32`,
  `vec2f`, `vec3f`, `vec4f`, `vec2i`, `vec3i`, `vec4i`, `vec2u`, `vec3u`,
  `vec4u`, or `mat4x4f`.
- `mat4x4f` values must be `Matrix4` or row-major `number[4][4]`; flat
  16-element arrays are invalid.
- `ShaderMaterialUniformBinding.stage` may be `vertex`, `fragment`, or `both`
  and must default to `both`.
- WebGPU uniform block layout must remain material-wide. Stage-restricted
  fields may be represented as internal padding fields in inactive stages to
  preserve buffer offsets.
- WebGPU shaders must access values through `ignisShaderUniforms.<wgslField>`.
- WebGL shaders must access values through the resolved `webglUniform` name.
- WebGPU must bind the custom uniform buffer at `@group(1) @binding(38)`.
- Shader source injection must use `ignis/material/uniform-block`.
- Schema changes must affect shader/program caches. Value-only changes must
  update backend uniform data without rebuilding shader modules or pipelines.

## Usage
```ts
import { ShaderMaterial } from "../src/materials/ShaderMaterial";

const material = new ShaderMaterial({
	uniformBindings: [
		{ name: "time", type: "f32", value: 0 },
		{
			name: "tint",
			type: "vec4f",
			value: [1, 0.8, 0.6, 1],
			wgslField: "tint",
			webglUniform: "uTint",
		},
	],
});

material.setUniform("time", 1.25);
```

```wgsl
@fragment
fn fsMain() -> @location(0) vec4<f32> {
	return ignisShaderUniforms.tint * ignisShaderUniforms.time;
}
```

```glsl
#version 300 es
precision highp float;
out vec4 outColor;

void main() {
	outColor = uTint * uTime;
}
```

```bash
bun tests/test_shader_material.mjs
bun tests/test_shader_runtime.mjs
bun tests/test_webgpu_bridge.mjs
bun tests/test_webgl_backend_v2.mjs
```

## Errors & Diagnostics
- `setUniform(name, value)` must throw when `name` has not been declared.
- Uniform declarations must throw when `webglUniform` or `wgslField` is not a
  shader identifier.
- Uniform declarations must throw when two bindings use the same `name`, or
  resolve to the same `webglUniform` or `wgslField`.
- Uniform value updates must throw when scalar, vector, matrix, signed integer,
  or unsigned integer constraints are violated.
- Shader compile diagnostics for custom chunks must follow the existing
  `ShaderMaterial` strict, warn, and silent runtime behavior.

## Compatibility / Breaking Changes
No existing `ShaderMaterial` texture binding API is removed. Existing custom
shader chunks remain valid unless they already declare WebGPU group `1` binding
`38` with an incompatible resource. Builtin materials and `SoftwareBackend`
behavior are unchanged in v1.
