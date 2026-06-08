# Shader Source Contract

## Scope

This document defines the `ShaderSource` contract for built-in WebGPU and WebGL
shader source loading. It applies to source text and source-map composition only.
It does not define `ShaderRuntime`, directive preprocessing, GPU shader module
caches, or WebGL program caches.

## Background

Built-in shader source loading is centralized in `ShaderSource` so WebGPU and
WebGL use one cache, one browser bundling path, and one Node file loading path.
Callers must use `ShaderSource.load()` for asynchronous loading and
`ShaderSource.get()` only after the requested source has been prepared.

## API/Contract

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
- WebGL scene sources must leave light-count placeholders in source text using
  `__WEBGL_MAX_DIRECTIONAL_LIGHTS__`, `__WEBGL_MAX_POINT_LIGHTS__`, and
  `__WEBGL_MAX_SPOT_LIGHTS__`. `ShaderRuntime` directive profiles own the
  replacement to concrete backend constants.
- Composite results returned from `ShaderSource` must be cloned so callers cannot
  mutate cached source maps.

## Usage

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

const mipmapBlit = ShaderSource.getSync("webgpu.utility.mipmapBlit.raw");

console.log(
	scene.sourceMap.lineCount,
	ssr.length,
	webglScene.fragment.length,
	mipmapBlit.length
);
```

## Errors & Diagnostics

- `ShaderSource.get()` must throw when the requested key and params have not
  been prepared.
- WebGL scene keys must throw when `params.limits` is missing.
- Browser loading must throw when a shader path is not bundled by the
  centralized `import.meta.glob` registry.
- `ShaderSource.getSync()` must throw when the requested key is not available
  through the synchronous source contract.
- Node loading must throw if `fs/promises` cannot read the resolved shader file.
- `getCacheStats()` should be used for source cache diagnostics only; runtime
  and GPU compilation cache diagnostics are reported by their owning systems.

## Compatibility / Breaking Changes

The legacy `loadShaderSource`, WebGPU `load*Shader*` helpers, and
`WebGLShaderSourceFactory` APIs have been removed. Callers must use
`ShaderSource.load()`, `ShaderSource.prepare()`, and `ShaderSource.get()`
instead.
