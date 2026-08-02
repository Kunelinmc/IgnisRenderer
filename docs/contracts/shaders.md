# Shader Contract

This document defines shader source ownership, composition, diagnostics, and custom material uniform behavior.

## Contract

### Shader source

- `ShaderSource.load(key, params)` must return the source identified by `key`.
  Raw keys return `string`, composite keys return `CompositeShaderSource`, and
  WebGL scene keys return `{ vertex, fragment }`.
- `ShaderSource.prepare(key, params)` must load and cache the requested source
  without returning it.
- `ShaderSource.prepareMany(requests)` must prepare every requested source and
  may run requests concurrently.
- `ShaderSource.get(key, params)` must synchronously return a prepared source and
  must throw when the source is not prepared.
- `ShaderSource.getSync(key)` must synchronously load only keys listed by the
  `ShaderSourceSyncKey` contract. It must populate the same prepared cache used
  by `ShaderSource.get()`.
- `ShaderSource.has(key, params)` must report whether `ShaderSource.get()` can
  return the requested source without asynchronous work.
- `ShaderSource.clearCache(scope)` must clear raw, composite, assembled, and
  prepared source caches for `scope`. `scope` may be `all`, `webgpu`, or `webgl`.
- `ShaderSource.getCacheStats()` must report cache hit, miss, and size counters
  for raw files, file composites, assembled results, and prepared results.
- `webgl.scene.raw` and `webgl.scene.composite` must receive
  `params.limits` with `maxDirectionalLights`, `maxPointLights`, and
  `maxSpotLights`.
- `webgl.scene.raw` and `webgl.scene.composite` may receive an internal
  `params.variant` descriptor. Renderer and material public APIs must not
  expose this descriptor; WebGL frame execution derives it from current
  frame, light, target, and built-in material state.
- WebGL scene fragment source may be assembled from multiple internal GLSL
  parts. `webgl.scene.composite.fragment.sourceMap.segments` may therefore
  contain one segment per internal part plus generated define or fallback
  segments.
- WebGL scene sources must leave light-count placeholders in source text using
  `__WEBGL_MAX_DIRECTIONAL_LIGHTS__`, `__WEBGL_MAX_POINT_LIGHTS__`, and
  `__WEBGL_MAX_SPOT_LIGHTS__`. Clustered fragment loops must use
  `__WEBGL_MAX_CLUSTER_LIGHTS_PER_FRAGMENT__`. `ShaderRuntime` directive
  profiles own replacement with concrete backend constants.
- Composite results returned from `ShaderSource` must be cloned so callers cannot
  mutate cached source maps.

### ShaderMaterial custom uniforms

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
- WebGPU must bind the custom uniform buffer at `@group(1) @binding(36)`.
- Shader source injection must use `ignis/material/uniform-block`.
- Schema changes must affect shader/program caches. Value-only changes must
  update backend uniform data without rebuilding shader modules or pipelines.

## Usage

### Shader source

```ts
import { ShaderSource } from "./shaders/ShaderSource";

const scene = await ShaderSource.load("webgpu.scene.composite");
const ssr = await ShaderSource.load("webgpu.postprocess.ssr.raw");

await ShaderSource.prepare("webgl.scene.raw", {
	limits: {
		maxDirectionalLights: 4,
		maxPointLights: 16,
		maxSpotLights: 8,
	},
});
const webglScene = ShaderSource.get("webgl.scene.raw", {
	limits: {
		maxDirectionalLights: 4,
		maxPointLights: 16,
		maxSpotLights: 8,
	},
});

await ShaderSource.prepare("webgl.scene.composite", {
	limits: {
		maxDirectionalLights: 4,
		maxPointLights: 16,
		maxSpotLights: 8,
	},
	variant: {
		output: "single",
		oit: false,
		scene: {
			shadows: false,
			shadowTransmittance: false,
			clusteredLighting: false,
			sh: false,
			localLightProbes: false,
			irradianceProbeGrid: false,
			reflectionProbes: false,
			environmentSpecular: false,
		},
		material: {
			model: "unlit",
			baseMap: false,
			metallicRoughnessMap: false,
			normalMap: false,
			emissiveMap: false,
			occlusionMap: false,
			iridescence: false,
			iridescenceMap: false,
			iridescenceThicknessMap: false,
			anisotropy: false,
			anisotropyMap: false,
			transmission: false,
			alphaMask: false,
		},
	},
});

const mipmapBlit = ShaderSource.getSync("webgpu.utility.mipmapBlit.raw");

console.log(
	scene.sourceMap.lineCount,
	ssr.length,
	webglScene.fragment.length,
	mipmapBlit.length
);
```

### ShaderMaterial custom uniforms

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
bun tests/static/shaders/test_shader_material.mjs
bun tests/static/shaders/test_shader_runtime.mjs
bun tests/static/webgpu/test_webgpu_bridge.mjs
bun tests/static/webgl/test_webgl_backend_v2.mjs
```

## Diagnostics

### Shader source

- `ShaderSource.get()` must throw when the requested key and params have not
  been prepared.
- WebGL scene keys must throw when `params.limits` is missing.
- WebGL scene keys must normalize missing `params.variant` to the full
  compatibility variant.
- Browser loading must throw when a shader path is not bundled by the
  centralized `import.meta.glob` registry.
- `ShaderSource.getSync()` must throw when the requested key is not available
  through the synchronous source contract.
- Node loading must throw if `fs/promises` cannot read the resolved shader file.
- `getCacheStats()` should be used for source cache diagnostics only; runtime
  and GPU compilation cache diagnostics are reported by their owning systems.

### ShaderMaterial custom uniforms

- `setUniform(name, value)` must throw when `name` has not been declared.
- Uniform declarations must throw when `webglUniform` or `wgslField` is not a
  shader identifier.
- Uniform declarations must throw when two bindings use the same `name`, or
  resolve to the same `webglUniform` or `wgslField`.
- Uniform value updates must throw when scalar, vector, matrix, signed integer,
  or unsigned integer constraints are violated.
- Shader compile diagnostics for custom chunks must follow the existing
  `ShaderMaterial` strict, warn, and silent runtime behavior.

## Compatibility

### Shader source

The legacy `loadShaderSource`, WebGPU `load*Shader*` helpers, and
`WebGLShaderSourceFactory` APIs have been removed. Callers must use
`ShaderSource.load()`, `ShaderSource.prepare()`, and `ShaderSource.get()`
instead.

### ShaderMaterial custom uniforms

No existing `ShaderMaterial` texture binding API is removed. Existing custom
shader chunks remain valid unless they already declare WebGPU group `1` binding
`36` with an incompatible resource. WebGPU custom shader chunks that manually
targeted binding `38` must use `#inject <ignis/material/uniform-block>` or move
their declaration to binding `36`. Builtin materials and `SoftwareBackend`
behavior are unchanged.

## Verification

```bash
bun tests/static/shaders/test_shader_source.mjs
bunx tsc --noEmit
```

## Related Documents

- [Rendering architecture](../architecture/rendering.md)
- [Materials contract](materials.md)
- [Migration guidance](../migrations/README.md)
