# Shader Contract

This document defines shader source ownership, composition, diagnostics, and custom material uniform behavior.

## Contract

### Shader source

- `ShaderSource.load(key, params)` must return the source artifact identified by
  the canonical, suffix-free `key`.
- A module artifact must expose one `CompositeShaderSource` as `source`. A
  program artifact must expose its composite sources through `stages`.
- Every artifact must expose `kind`, `key`, stable `identity`, `language`, and
  `sourceKind` metadata.
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
- `webgl.scene` may receive an internal `params.specialization` descriptor.
  Renderer and material public APIs must not
  expose this descriptor; WebGL frame execution derives it from current
  frame, light, target, and built-in material state.
- Light and device limits must be owned by the backend directive profile and
  must not participate in shader source artifact identity.
- Optional WebGL scene shader blocks must be selected solely by the normalized
  scene variant. Device texture-unit limits must be validated by the resulting
  exact sampler layout rather than by shader-source capability flags.
- WebGL scene fragment source must be assembled from the internal GLSL assets
  declared by the WebGL shader manifest.
  `webgl.scene` fragment source-map segments may contain one segment
  per internal part plus generated define or fallback segments.
- WebGL scene sources must leave light-count placeholders in source text using
  `__WEBGL_MAX_DIRECTIONAL_LIGHTS__`, `__WEBGL_MAX_POINT_LIGHTS__`, and
  `__WEBGL_MAX_SPOT_LIGHTS__`. Clustered fragment loops must use
  `__WEBGL_MAX_CLUSTER_LIGHTS_PER_FRAGMENT__`. `ShaderRuntime` directive
  profiles own replacement with concrete backend constants.
- The WebGL directive profile must provide
  `#import <ignis/webgl/animation>`. The include is the shader-facing animation
  ABI for `ShaderMaterial` and built-in vertex shaders. It must declare the
  reserved `uAnimationPayload`, `uMorphPositionDeltas`, and
  `uMorphNormalDeltas` samplers; the `uAnimationCounts`, `uAnimationOffsets`,
  and `uAnimationTextureWidths` metadata uniforms;
  `IgnisAnimationVertex`, `ignisApplyAnimationVertex(...)`, and
  `ignisApplyAnimationPosition(...)`.
- WebGL animation helpers must expose current and previous deformation, apply
  morph before skinning, and use the fixed scene joint and weight attribute
  locations. Importing the ABI does not automatically rewrite a custom vertex
  entry point; `ShaderMaterial` authors must call the appropriate helper.
- The WebGL directive profile must provide
  `#import <ignis/webgl/constants>`. Its general numerical constants must match
  the equivalent WebGPU include.
- The WebGPU directive profile must provide
  `#import <ignis/webgpu/constants>`. The include owns general numerical
  constants, including circular, Gaussian-normalization, golden-ratio, and
  Stefan–Boltzmann constants. Effect-specific parameters, such as a blur
  kernel's sigma, must remain owned by their effect.
- Composite results returned from `ShaderSource` must be cloned so callers cannot
  mutate cached source maps.
- Built-in WebGL and WebGPU shader assets, source composition, specialization,
  preload groups, and directive-profile inputs must be declared by backend
  pure-data manifests interpreted by the shared shader manifest runtime.
- Built-in backend code must not perform ad hoc shader-text replacement.
- Manifest source expressions are closed to the built-in `asset`, `concat`,
  `when`, `defines`, and `template` nodes. Manifests must not contain callbacks
  or register custom operations.

### Directive profiles

- Each GPU backend instance must own one composed `ShaderDirectiveProfile`.
  Shader runtime code must not construct a cross-backend default profile or
  import backend constants and built-in shader assets.
- A directive profile must be composed from a prepared static base and an
  instance overlay resolved after backend capability probing. Profile
  composition must reject duplicate include-module `(language, id)` pairs,
  duplicate injection-script ids, and feature packs targeting another backend.
- `ShaderDirectiveProfile.fingerprint` must be derived from the backend, feature
  pack ids and revisions, include-module contents, injection-script schemas,
  and instance overlay contents. Shader and program cache identities must include
  this fingerprint.
- Built-in directive include modules must be stored as shader assets. Instance
  overlays may generate only short constant or define modules from resolved
  backend limits and ABI values.
- `ShaderInjectionScript` argument schemas must be validated after macro
  expansion and before `run()` executes. Invalid invocations must not execute.
  Strict mode must report an error, warn mode must report a warning, and silent
  mode must skip the invocation without publishing a diagnostic.
- Backend directive hooks may add namespaced include modules and injection
  scripts. A hook patch must not replace a profile module or script; any
  collision must disable the entire patch for that invocation context.

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
- WebGPU must bind the custom uniform buffer at `@group(1) @binding(39)`.
- Custom WebGPU shader code that declares the material uniform buffer directly
  must use binding `39`; binding `36` is reserved for morph weights.
- Shader source injection must use `ignis/material/uniform-block`.
- Schema changes must affect shader/program caches. Value-only changes must
  update backend uniform data without rebuilding shader modules or pipelines.

## Usage

### Shader source

```ts
import { ShaderSource } from "./shaders/ShaderSource";

const scene = await ShaderSource.load("webgpu.scene");
const ssr = await ShaderSource.load("webgpu.postprocess.ssr");

await ShaderSource.prepare("webgl.scene", {
	specialization: undefined,
});
const webglScene = ShaderSource.get("webgl.scene");

await ShaderSource.prepare("webgl.scene", {
	specialization: {
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

const mipmapBlit = ShaderSource.getSync("webgpu.utility.mipmapBlit");

console.log(
	scene.source.sourceMap.lineCount,
	ssr.source.code.length,
	webglScene.stages.fragment?.code.length,
	mipmapBlit.source.code.length
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
bun tests/static/webgpu/test_webgpu_bridge_shader_contracts.mjs
bun tests/static/webgl/test_webgl_backend_program_library.mjs
bun tests/static/webgl/test_webgl_backend_scene_shadow_contracts.mjs
```

## Diagnostics

### Shader source

- `ShaderSource.get()` must throw when the requested key and params have not
  been prepared.
- WebGL scene keys must normalize missing `params.specialization` to the full
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

## Verification

```bash
bun tests/static/shaders/test_shader_source.mjs
bunx tsc --noEmit
```

## Related Documents

- [Rendering architecture](../architecture/rendering.md)
- [Materials contract](materials.md)
- [Migration guidance](../migrations/README.md)
