# Rendering Pipeline and Shader Contracts

## Scope

This document defines the rendering, shader, coordinate system, vertex layout,
post-processing, environment lighting, and performance contracts for
contributors changing rendering behavior.

## Background

This document is the canonical rendering contract for contributor-facing
engineering decisions. Automation-specific entrypoints, such as `AGENTS.md`,
may reference this file, but the contracts here apply to all contributions that
touch renderer frame execution, backend passes, shaders, materials,
post-processing, lighting, or GPU data layout.

IgnisRenderer uses a right-handed coordinate system, linear lighting, explicit
backend pass ownership, and backend-specific shader folders. Backend-specific
contract documents in `docs/` may define additional requirements for WebGPU,
WebGL, Software, shadows, warmup, and post-processing features.

## API/Contract

### Frame Pipeline Contract

The renderer frame pipeline must preserve this logical order:

1. Feature resolution detects shadows, IBL, post-processing, and related
   requirements.
2. Warmup pre-compiles required shaders and pipelines when needed.
3. Sync-in transfers `Node` state into ECS.
4. Simulation updates animation, physics, and particles.
5. Transform update writes current world matrices.
6. Prepared scene building collects draw packets indexed by `MeshInstance`.
7. Backend dispatch performs software rasterization, GPU command encoding,
   WebGL batching, and backend-owned passes including `"postprocess"` when
   `BackendCapabilities.postProcess` is `true`.
8. Sync-out propagates ECS simulation output back to `Node` facades.

### Coordinate and Matrix Contract

- The coordinate system must be right-handed.
- `+Y` must be up.
- `-Z` must be forward in camera view direction.
- `+X` must be right.
- Camera logic must handle perspective projection as FOV-based non-linear depth.
- Camera logic must handle orthographic projection as volume-based linear depth.
- `src/maths/Matrix4.ts` must use row-major `number[row][col]` internally.
- GPU buffers for WGSL and GLSL must use column-major `Float32Array` packing.
- `A.multiply(B)` must perform `A = A * B`.
- Internal projection matrices must target the standard NDC Z range `[-1, 1]`.

### Color and Lighting Contract

- Lighting and shading calculations must run in linear space.
- Encoding and decoding must assume gamma 2.2 unless a more specific contract
  overrides it.
- Shaders must treat textures as sRGB by default and decode samples to linear
  space.
- Linear textures, such as normal maps and roughness maps, must be flagged to
  bypass sRGB decoding.
- The default PBR model must use GGX NDF, Smith-Schlick geometry, and
  Fresnel-Schlick.

### Vertex Layout Contract

- `shaderLocation 0` must be position as `vec3`.
- `shaderLocation 1` must be normal as `vec3`.
- `shaderLocation 2` must be UV0 as `vec2`.
- `shaderLocation 3` must be UV1 as `vec2`.
- `shaderLocation 4` must be tangent as `vec4`, with `w` storing handedness.

### Shader Management Contract

- TypeScript files must not embed shader code as long strings.
- Shader source must live in separate `.wgsl` or `.glsl` files.
- Shader files must be organized by backend under `src/shaders/software/`,
  `src/shaders/webgpu/`, and `src/shaders/webgl/`.
- `src/shaders/runtime/` owns rule-based shader transformation, validation,
  injection, and source mapping.

### Post-Processing Contract

- `src/postprocess/` owns logical pass descriptors, graph compilation, G-buffer
  semantic contracts, history resource policies, transient resource policies,
  and pass-owned implementations for built-in cross-backend passes.
- `Renderer` must own only the public `renderer.postProcess` registry.
- Post-processing must execute as a backend-owned `"postprocess"` backend pass.
- Software, WebGL, and WebGPU backends must hold a
  `BackendPostProcessRuntime` when they support post-processing.
- Backends may expose `IPostProcessExecutor.executePass(passId, request)`.
- Backends may expose `IPostProcessExecutor.getPassExecutionContext(request)`
  as an optional low-level helper.
- Backends may expose `LogicalGBufferBridge`.
- Backends must not expose public post-process graph registration APIs,
  `renderer.postprocess` backend extensions, or hardcoded pass kernel
  orchestration that belongs in `src/postprocess/passes/`.

### Built-In Post-Processing Pass Contract

- SSAO must provide screen-space ambient occlusion with depth-aware bilateral
  blur.
- TAA must provide temporal anti-aliasing with variance clamping and history
  rectification.
- SSR must use Hi-Z tracing for screen-space reflections.
- Volumetric lighting must support ReSTIR-style reservoir spatiotemporal
  importance resampling where implemented.
- Bloom must support HDR thresholding and soft-knee curves.
- Motion blur must use velocity data from `gMotionDepth` with shutter scale and
  sample control.
- Depth of Field must support focus distance, focus range, bokeh behavior, and
  chromatic aberration where implemented.
- FXAA must remain available for broad compatibility.

### WebGPU Deferred Lighting Contract

- WebGPU `main-opaque` may internally split into background, G-buffer, deferred
  lighting resolve, and forward fallback GPU passes.
- This split must remain WebGPU-internal and must not add global renderer
  frame-pass stages for Software or WebGL.
- Detailed WebGPU deferred requirements must follow
  `docs/webgpu-deferred-lighting-contract.md`.

### WebGL Post-Process Program Contract

- Built-in WebGL post-process implementations must own their program
  descriptors, uniform reflection, slots, warmup, and slot lifecycle.
- `WebGLProgramCompiler` owns compilation and raw WebGL resource lifecycle.
- `WebGLProgramLibrary` must not expose pass-specific post-process program APIs.
- `WebGLProgramLibrary` is reserved for backend-owned scene, presentation,
  copy, shadow, particle, environment, and OIT programs.

### Warmup and Environment IBL Contract

- `WarmupPlanner` must pre-compile required pipelines and resources before
  rendering based on scene features.
- `IBLPrefilter` owns CPU, worker, and WebGPU environment specular prefiltering
  as a standalone service.
- `Renderer` must not schedule environment IBL bake/update work or expose
  environment IBL update APIs.
- Applications and tools must invoke `IBLPrefilter` or
  `bakeEnvironmentIBLFromEnvironmentMap` explicitly and assign probe data.

### Performance and Resource Contract

- Rendering and simulation time units must be seconds.
- Time-step variables should use the `deltaTimeSeconds` suffix.
- Hot paths must use pre-allocated math objects where practical and avoid
  avoidable allocation.
- Physics must use the adapter pattern to allow backend implementation swaps.
- Backend resource ownership must use explicit `destroy()` methods.
- Specialized registries, such as `WebGPUTextureRegistry`, should manage
  backend resources and may use `FinalizationRegistry` as a safety net.

## Usage

Review this document before editing renderer passes, shader source, material
data packing, camera math, post-processing, IBL, warmup, or backend resource
lifecycle code.

Example validation commands:

```bash
bunx tsc --noEmit
bun tests/static/webgpu/test_webgpu_bridge.mjs
bun tests/static/webgpu/test_webgpu_post_graph.mjs
bun tests/static/webgl/test_webgl_backend_v2.mjs
```

## Errors & Diagnostics

- Matrix and projection bugs should report whether the input camera is
  perspective or orthographic.
- Shader packing diagnostics should name the backend, shader stage, binding,
  attribute location, or semantic that failed validation.
- Color-space regressions should identify whether a texture was treated as sRGB
  or linear.
- Post-process execution errors should identify the logical pass id, backend
  pass stage, and G-buffer dependency that failed.
- Backend resource leaks should identify the owning registry and missing
  `destroy()` path.

## Compatibility / Breaking Changes

Changes to coordinate handedness, matrix packing, vertex attributes, shader
locations, material texture decoding, post-process graph ownership, backend pass
ordering, or environment IBL ownership are breaking changes. Such changes must
update contract documents, tests, and migration notes in the same PR.
