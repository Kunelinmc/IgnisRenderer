# WebGL Backend V2 Contract
## Scope
This document defines the Phase 1 contract for `WebGLBackend` V2 in
IgnisRenderer. `WebGLBackend` must keep the public constructor and type name
unchanged.

## Background
The previous WebGL path provided a V1-style pass execution and feature subset.
Phase 1 of V2 must align orchestration semantics with WebGPU while keeping
WebGL-specific implementation constraints. WebGL post-processing is now
backend-owned and executes through the `"postprocess"` backend pass.

## API/Contract
- `WebGLBackendSession` must report core `profile.capabilities.sh = true`.
- `WebGLBackendSession` must report `profile.capabilities.clusteredLighting = true`.
- `WebGLBackendSession` must report `profile.capabilities.postProcess = true`.
- `WebGLBackend` must not expose `postProcessCapabilities`.
- WebGL post-process support must be derived from pass-owned WebGL
  implementations.
- `WebGLBackendSession.extensions` must not expose `renderer.postprocess`.
- `WebGLBackendSession.executePass({ stage: "postprocess" }, context)` must delegate
  to backend-owned post-process runtime execution.
- `WebGLBackendSession.endFrame()` must commit post-process histories only after the
  WebGL frame executor ends successfully.
- `WebGLBackendSession.abortFrame(error)` must abort post-process runtime before
  clearing WebGL frame executor state.
- `WebGLBackendSession.warmup(context, options)` must use
  `BackendPostProcessRuntime.compileWarmupGraph(context)` to collect post-process
  pass descriptors.
- `WebGLPostProcessExecutor.createGBufferBridge(context)` must return a
  `LogicalGBufferBridge` that wraps WebGL texture handles.
- `FogPass` must provide a WebGL implementation and must support both
  post-process fog and scene-mode fog.
- WebGL G-buffer bridge channels must report actual runtime attachment formats.
- When `EXT_color_buffer_float` is unavailable, WebGL scene, motion-depth, and
  post-process color attachments must fall back to `rgba8unorm`.
- `WebGLBackend` must not expose public `postProcess`, `postProcessExecutor`,
  `postProcessAdapter`, or `createPostProcessGBufferBridge(context)` members.
- The backend must validate pass dependency order per frame and must treat
  `skipPass` as an executed stage.
- SH lighting must use 16 coefficients and must be uploaded through
  texture-backed data for shader sampling.
- Clustered lighting must be CPU-built with tile and z-slice partitioning.
- Clustered lighting must provide runtime fallback to legacy forward lighting
  when requirements are not met.
- For non-perspective cameras, clustered lighting must be disabled for the frame
  and a warning key must be emitted.
- Forward-lighting uniform budgets must clamp to `4` directional lights, `16`
  point lights, and `8` spot lights.
- PBR materials with `transmissionFactor > 0` must be treated as transparent
  pass submissions even when `alphaMode` is `OPAQUE`.
- WebGL PBR shading must consume transmission through `uPBR.w` and must modulate
  composite alpha so transmissive surfaces do not render as fully opaque.
- `ShaderMaterial` custom WebGL scene shaders must resolve `webgl` GLSL
  `fragment-single` for `single` mode.
- `ShaderMaterial` custom WebGL scene shaders must resolve `webgl` GLSL
  `fragment-mrt` for `mrt` mode and may fall back to `fragment-single` when
  `fragment-mrt` is absent.

## Usage
```ts
import { FogPass, Renderer, WebGLBackend } from "../src";

const backend = new WebGLBackend();
const renderer = new Renderer(backend, canvas, camera);
renderer.features.enableSH = true;
renderer.features.enableClusteredLighting = true;
renderer.features.clusteredLightingOptions = {
	tileSizePx: 64,
	zSlices: 24,
	maxLights: 256,
	maxLightsPerCluster: 64,
};
renderer.postProcess.registerPass(new FogPass({
	enabled: true,
	options: {
		application: "postprocess",
	},
}));

await renderer.initialize();
renderer.requestRender();
```

```bash
bun tests/static/webgl/test_webgl_backend_v2.mjs
bun tests/static/webgl/test_webgl_frame_executor_fxaa.mjs
```

## Errors & Diagnostics
- `webgl-clustered-perspective-only`: triggered when
  `enableClusteredLighting` is `true` on a non-perspective camera.
- `webgl-clustered-light-budget`: triggered when light count exceeds
  `clusteredLightingOptions.maxLights`.
- `webgl-clustered-texture-size-overflow`: triggered when clustered buffers
  cannot fit within texture capacity.
- `webgl-sh-ambient-texture-create-failed`: triggered when SH coefficient
  texture allocation fails.
- `webgl-sh-ambient-texture-upload-failed`: triggered when SH coefficient
  texture upload fails.
- `webgl-hdr-float-unsupported`: triggered when `EXT_color_buffer_float` is
  unavailable and WebGL falls back to `RGBA8` color, motion-depth, and
  post-process attachments.
- `"<backend>-postprocess-unsupported-<passId>"`: triggered when an enabled
  renderer-default built-in post-process pass has no WebGL implementation.

## Compatibility / Breaking Changes
- Public backend type name remains `WebGLBackend`.
- `WebGLBackend` is now a provider and exposes only `createSession(context)`;
  runtime state and lifecycle methods belong to `WebGLBackendSession`.
- Core capability fields `sh` and `clusteredLighting` changed from disabled to
  enabled.
- `BackendCapabilities.postProcess` is added and is `true` for `WebGLBackend`.
- Backend post-process capability maps remain removed.
- `renderer.postprocess` is no longer a WebGL backend extension.
- `resolvePostProcessBackendExtension(new WebGLBackend())` is removed.
- `WebGLBackend.registerPostProcessPass(pass)` and
  `WebGLBackend.unregisterPostProcessPass(id)` are removed.
- `WebGLBackend.postProcess` is removed.
- `WebGLBackend.postProcess.registerPass(pass)` and
  `WebGLBackend.postProcess.unregisterPass(id)` are removed.
- `WebGLBackend.postProcessAdapter`, `WebGLBackend.postProcessExecutor`, and
  `WebGLBackend.createPostProcessGBufferBridge(context)` are removed.
- Public WebGL custom post-process passes must migrate to `PostProcessPass`
  instances registered through `renderer.postProcess.registerPass(pass)`.
- Forward-lighting point-light budget changed from `4` to `16` to match the
  WebGPU backend budget.
- Test entrypoint is `tests/static/webgl/test_webgl_backend_v2.mjs`.
