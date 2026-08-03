# WebGL Backend Contract

This document defines the current WebGL backend lifecycle, frame graph, resource ownership, execution, and compatibility behavior.

## Contract

### Backend lifecycle and execution

- `WebGLBackend` must report core `profile.capabilities.sh = true`.
- `WebGLBackend` must report `profile.capabilities.clusteredLighting = true`.
- `WebGLBackend` must report `profile.capabilities.postProcess = true`.
- Projected decals are temporarily not applicable to `WebGLBackend`.
  `WebGLBackend` must ignore prepared `DecalPacket` instances without capability
  probing, receiver planning, resource allocation, decal passes, diagnostics,
  or changes to forward frame output.
- `WebGLBackend` must not expose `postProcessCapabilities`.
- WebGL post-process support must be derived from pass-owned WebGL
  implementations.
- `WebGLBackend.extensions` must not expose `renderer.postprocess`.
- `WebGLBackend.extensions` must expose the stable
  `IBL_PREFILTER_EXECUTOR_EXTENSION` facade for WebGL2 fragment-based IBL
  prefiltering.
- The IBL prefilter facade must remain identity-stable across WebGL context
  loss and restoration. It must reject work while uninitialized, retain an
  explicit WebGL request while context-lost, and resume delegation to restored
  context-scoped services.
- The stable facade and IBL scheduling policy must be owned by the WebGL
  extension owner rather than `WebGLBackend`. The context-generation runtime
  must be owned beside, not inside, `WebGLFrameServiceOwner`.
- WebGL fragment IBL prefiltering must require `EXT_color_buffer_float` and
  either `OES_texture_float_linear` or `OES_texture_half_float_linear`.
- `WebGLContextWorkQueue` must serialize frame lifecycle operations, backend
  passes, fragment IBL prefilter work, custom render-target readback, warmup,
  and frame-sized maintenance.
- A frame pass or frame-end operation must reserve ownership before awaiting
  pass-boundary work. A concurrent pass must reject with `active-pass`; a
  concurrent frame-end operation must reject with `active-frame`.
- The queue must permit fragment IBL prefiltering at an active frame pass
  boundary. It must reject context work requested while a backend pass callback
  is executing and must reject idle-only work while any frame is active.
- WebGL warmup requested at a safe active-frame boundary must wait for frame
  release and execute before a subsequent frame begins. Warmup requested while
  a backend pass callback is active must reject. Each released frame must fix
  the deferred warmup batch eligible before the next frame so work enqueued by
  that batch cannot starve rendering.
- WebGL resize must update desired dimensions synchronously and schedule one
  keyed, latest-wins maintenance operation. Active-frame resize must not destroy
  frame resources until frame-end or frame-abort cleanup, and that cleanup must
  await the maintenance operation before settling.
- The queue must restore the active scene framebuffer baseline after
  pass-boundary auxiliary work and the default framebuffer baseline after idle
  work. IBL execution must additionally restore pixel-pack state.
- When a frame and auxiliary work are both waiting, the queue must execute at
  most one auxiliary item between complete frames. It may drain auxiliary work
  continuously when no frame is waiting.
- Context loss must reject active context work. Pending work with a retain
  policy must remain queued and late-bind to the restored context generation;
  pending readback must reject because framebuffer contents do not survive
  restoration.
- Context loss must reject boundary waiters and pending or active warmup, and
  warmup must observe the queue abort signal between compilation and
  finalization slices. Warmup must not replay on the restored generation.
- Pending resize maintenance must not replay after context loss. Context
  restoration must initialize the restored frame services with the latest
  desired dimensions.
- WebGL fragment IBL prefiltering must use transient input, output, and
  framebuffer resources and must delete them on success, cancellation,
  context loss, or failure.
- `WebGLBackend.endFrame()` must return and settle the complete asynchronous
  present and commit operation before the context work queue releases frame
  ownership.
- `WebGLBackend.executePass({ stage: "postprocess" }, context)` must delegate
  to backend-owned post-process runtime execution.
- `WebGLBackend.endFrame()` must commit post-process histories only after the
  WebGL frame executor ends successfully.
- `WebGLBackend.abortFrame(error)` must abort post-process runtime before
  clearing WebGL frame executor state.
- `WebGLBackend.resize(size)` must invalidate post-process frame-sized
  resources before invalidating WebGL frame targets.
- WebGL context replacement must destroy the post-process resource pool before
  destroying the previous context-scoped WebGL frame services.
- `WebGLBackend.warmup(context, options)` must use
  `BackendPostProcessRuntime.planWarmup(context)` once and reuse that validated
  declaration plan while warming post-process implementations.
- `WebGLPostProcessExecutor.createGBufferBridge(context)` must return a
  `LogicalGBufferBridge` that wraps WebGL texture handles.
- `FogPass` must provide a WebGL implementation and must support both
  post-process fog and scene-mode fog.
- WebGL G-buffer bridge channels must report actual runtime attachment formats.
- TAA and motion history textures must be allocated and destroyed by the
  backend post-process resource pool. WebGL frame-target allocation must not
  create or destroy temporal history textures.
- When an enabled WebGL post-process implementation requires `albedo`,
  `roughness`, `metallic`, or `specular`, the backend must attempt to allocate
  a five-target material G-buffer. It must require both `MAX_DRAW_BUFFERS`
  and `MAX_COLOR_ATTACHMENTS` to be at least `5`.
- The material G-buffer must use `sceneColor`, `gMotionDepth`,
  `gNormalRoughMetal`, `gAlbedoAlpha`, and `gSpecular`. The latter three must
  expose `linear-rgb-alpha`, `normal-roughness-metallic.z/.w`, and
  `specular-color-factor.rgba` logical encodings through `LogicalGBufferBridge`.
- `gNormalRoughMetal` and `gSpecular` must use `rgba16float` when float color
  attachments are available and `rgba8unorm` otherwise; `gAlbedoAlpha` must
  use `rgba8unorm`.
- PBR `gSpecular` values must include the resolved `specularColorFactor` and
  `specularFactor`. Built-in opaque PBR, Phong, and Unlit draws may write the
  material payload; `ShaderMaterial` keeps its existing MRT compatibility path.
- When `EXT_color_buffer_float` is unavailable, WebGL scene, motion-depth, and
  post-process color attachments must fall back to `rgba8unorm`.
- `WebGLBackend` must not expose public `postProcess`, `postProcessExecutor`,
  `postProcessAdapter`, or `createPostProcessGBufferBridge(context)` members.
- The backend must validate pass dependency order per frame and must treat
  `skipPass` as an executed stage.
- WebGL custom color targets must support exact storage for `r8unorm`,
  `rg8unorm`, `rgba8unorm`, `rgba8unorm-srgb`, `r16float`, `rg16float`,
  `rgba16float`, `r32float`, `rg32float`, and `rgba32float`.
- Float custom color targets must require `EXT_color_buffer_float`; they must
  not fall back to normalized formats.
- WebGL custom depth targets must use sampleable depth textures. Supported
  formats are `depth16unorm`, `depth24plus`, and `depth32float`.
- WebGL custom targets must reject stencil formats, unsupported formats,
  attachment counts beyond `MAX_DRAW_BUFFERS` or `MAX_COLOR_ATTACHMENTS`, and
  incomplete framebuffers.
- WebGL custom render passes must set the viewport from their attachments,
	enable scissor testing when a scissor rect is set, and restore the
	frame-executor baseline state on success or failure.
- WebGL custom render passes must reject compute commands, texture copies,
  resolve targets, non-zero `baseVertex`, and non-zero `firstInstance`.
- WebGL custom target readback must preserve native bottom-left row order and
  must return bytes matching the exact attachment format.
- SH lighting must use 16 coefficients and must be uploaded through
  texture-backed data for shader sampling.
- Clustered lighting must be CPU-built with tile and z-slice partitioning.
- Clustered lighting must provide runtime fallback to legacy forward lighting
  when requirements are not met.
- For non-perspective cameras, clustered lighting must be disabled for the frame
  and a warning key must be emitted.
- Forward-lighting uniform budgets must clamp to `4` directional lights, `16`
  point lights, and `8` spot lights.
- `WebGLShadowRuntime` must be the sole frame-aware entry point for the WebGL
  shadow subsystem. It must own shadow metadata synchronization, reusable frame
  planning, particle-volume state, and the identity-stable sampling state read
  by scene consumers.
- `WebGLShadowRasterPass` must consume only a prepared
  `WebGLShadowRasterPlan`. It must not inspect `FrameContext`, light collection
  state, particle transients, or frame-target services, and it must exclusively
  own the shadow framebuffer, depth atlas, and transmittance atlas.
- WebGL frame preparation must synchronize shadow metadata before light
  collection, prepare shadow plans and predictable native targets after light
  collection, and compile the frame graph only after those targets are known.
  The shadow graph node must execute only the prepared plan.
- Shadow consumers must obtain atlas, transmittance, particle-volume, and
  availability data through one readonly sampling-state contract. Runtime-owned
  typed arrays must remain identity-stable and must not be mutated by consumers.
- Shadow frame abort must clear pending plans and active sampling availability
  while retaining reusable native resources. Missing shader programs must
  disable sampling for the frame without discarding prepared targets.
- PBR materials with `transmissionFactor > 0` must be treated as transparent
  pass submissions even when `alphaMode` is `OPAQUE`.
- WebGL PBR shading must consume transmission through `uPBR.w` and must modulate
  composite alpha so transmissive surfaces do not render as fully opaque.
- `ShaderMaterial` custom WebGL scene shaders must resolve `webgl` GLSL
  `fragment-single` for `single` mode.
- `ShaderMaterial` custom WebGL scene shaders must resolve `webgl` GLSL
  `fragment-mrt` for `mrt` mode and may fall back to `fragment-single` when
  `fragment-mrt` is absent.

### Internal frame graph

- `WebGLFrameGraphPlanner` must create WebGL internal nodes for every enabled
  renderer-level `FramePass` during `beginFrame()`.
- `WebGLFrameGraphRuntime` must execute synthetic `scene-clear` and optional
  `environment` nodes during `beginFrame(context)`.
- `WebGLFrameGraphRuntime` must execute a synthetic `present` node during
  `endFrame(context)`.
- `WebGLFrameServiceOwner` must construct device-scoped WebGL frame services
  and destroy them in dependency order. It must directly provide frame begin,
  finish, abort, resize, and service lifetime coordination.
- Each WebGL frame runtime must own and destroy the native handles it creates.
  Frame-sized attachments must be owned exclusively by
  `WebGLFrameTargetManager`; post-process histories must be owned exclusively
  by `BackendPostProcessRuntime` resource pools.
- `WebGLFrameNodeExecutorRegistry` must assign every WebGL graph node kind to
  exactly one executor and must reject missing or duplicate registrations.
- `WebGLFrameServiceOwner` must not own renderer-level pass orchestration.
- WebGL graph and post-process runtimes must depend on narrow internal
  contracts and must not require the concrete `WebGLFrameServiceOwner` type.
- The frame graph runtime must retain one post-process declaration plan,
  configure frame targets, finalize that plan, and pass only generic finalized
  frame requirements to frame services before scene execution.
- WebGL temporal camera state must commit only after post-process histories
  commit. Frame abort must restore the tentative jitter sequence and previous
  view-projection state.
- `WebGLFrameGraphCompiler` must compile one whole-frame definition and
  preserve planner order among retained nodes.
- `executePass()` must consume a precompiled stage slice and must not plan or
  compile the stage again.
- The WebGL resource catalog must provide complete logical descriptors when
  metadata is known and stable physical IDs without native handles.
- `WebGLFrameTargetManager` must describe only scene, OIT, and post-process
  targets that it owns. `WebGLShadowRuntime` must separately describe prepared
  shadow resources, and `WebGLFrameServiceOwner` must merge both catalogs before
  whole-frame compilation.
- A retained shadow stage must declare `shadow:atlas`,
  `shadow:transmittance`, and `shadow:particle-volume` only when their prepared
  physical resources exist. A frame without a retained shadow stage must omit
  all three descriptors and bindings.
- `shadow:particle-volume` must use `r32float`, frame residency, dimensions
  derived from the fixed WebGL particle-shadow grid constants, and a stable
  physical ID. The shadow node must optionally write it as `copy-target` after
  particle simulation has completed.
- Opaque scene nodes and every legacy or OIT transparent raster node must
  optionally read `shadow:particle-volume` as `texture-sampling`. Particle
  rendering nodes must not declare that read while their shader does not
  consume the texture.
- Render Graph shadow descriptors must describe logical extent and physical
  identity only. Native atlas and particle-volume allocation and destruction
  must remain owned by the shadow runtime and raster pass.
- `WebGLFrameGraphCompiler` must emit diagnostics for missing resources,
  reads before creation, duplicate creates, unsupported usages, and WebGL
  texture feedback loops.
- WebGL OIT graph nodes must use separate `oit-accum` and `oit-reveal` nodes
  because WebGL cannot assign different blend states per attachment.
- WebGL OIT scene-color copy and composition must use separate
  `oit-copy-scene-color` and `oit-resolve` nodes. The copy node must read
  `frame:scene-color` and write `post:color`; the resolve node must read
  `post:color`, `oit:accum`, and `oit:reveal` before writing
  `frame:scene-color`.
- WebGL frame graph debug state may be exposed through backend-internal hooks.
- Legacy barrier records must be projections of shared logical transitions;
  they must not represent native WebGL synchronization commands.
- Shared shadow diagnostics must not enter the legacy `graphDiagnostics`
  array or trigger `Logger`.
- `WebGLFrameGraphRuntime.endFrame` must seal graph analysis after successful
  presentation. `WebGLBackend` must commit analysis only after post-process
  history and backend frame cleanup succeed. Any failure must abort analysis.
- Custom render passes and particle simulation that bypass graph executors must
  have always-retained opaque placeholder nodes and mark analysis coverage as
  `"opaque"` without changing their execution path.
- The WebGL internal graph must not add public renderer graph registration APIs.

## Usage

### Backend lifecycle and execution

```ts
import { FogPass, Renderer, WebGLBackend } from "../src";

const backend = new WebGLBackend();
const renderer = new Renderer({ backend, canvas, camera });
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
bun tests/run_all.mjs tests/static/webgl
bun tests/static/webgl/test_webgl_frame_executor_fxaa.mjs
```

### Internal frame graph

```ts
const backend = new WebGLBackend();
backend.attach({
	surface: { canvas },
	events,
});
```

```bash
bun tests/static/webgl/test_webgl_frame_graph_planner.mjs
bun tests/static/webgl/test_webgl_frame_graph_compiler.mjs
bun tests/static/webgl/test_webgl_frame_graph_runtime.mjs
```

## Diagnostics

### Backend lifecycle and execution

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
- `webgl-gbuffer-material-semantics-unsupported`: triggered when an enabled
  material-semantic post-process requirement cannot use the five-target WebGL
  G-buffer because the runtime reports fewer than five draw buffers or color
  attachments.
- `"<backend>-postprocess-unsupported-<passId>"`: triggered when an enabled
  renderer-default built-in post-process pass has no WebGL implementation.
- `WebGL custom render target "<id>" requires EXT_color_buffer_float.`:
  triggered when an exact float attachment cannot be allocated.
- `WebGL custom render target "<id>" exceeds the runtime color attachment limit.`:
  triggered when MRT requirements exceed WebGL limits.
- `WebGL custom framebuffer "<id>" is incomplete`: triggered when attachment
  allocation produced an invalid framebuffer.
- `WebGL IBL prefilter acceleration requires ...`: triggered when required
  floating-point render-target or filtering extensions are unavailable.
- `WebGL IBL prefilter framebuffer is incomplete at mip <level>`: triggered
  when a transient RGBA16F output mip cannot be attached for rendering.
- `WebGL context work "<label>" cannot run while frame pass "<stage>" is
  active.`: triggered when work would wait on the pass that requested it.
- `WebGL render-target readback cannot run while a frame is active.`: triggered
  when `readColor` is requested before the current frame completes.

### Internal frame graph

- `read-before-create` must trigger when a required node read references an
  inactive resource.
- `duplicate-create` must trigger when a node creates an active resource.
- `missing-resource` must trigger when OIT or other required runtime targets are
  unavailable.
- `texture-feedback-loop` must trigger when a node samples and writes the same
  texture in one framebuffer pass.
- `unsupported-node-resource` must trigger when a node declares a usage outside
  the WebGL graph usage set.
- `missing-node-executor` must trigger when a planned node has no runtime
  executor.
- `webgl-frame-graph-stage-unsupported-{stage}` must be logged once when a
  renderer-level pass has no WebGL internal graph plan.
- `graphAnalysis` must expose grouped canonical transitions, generation-aware
  live ranges, completeness, shadow diagnostics, and successful/failed
  snapshots without native handles.

## Compatibility

### Backend lifecycle and execution

- Public backend type name remains `WebGLBackend`.
- `WebGLBackend` is now an attached backend runtime. It exposes
  `attach(context)`, profile, extensions, and frame lifecycle methods directly.
- A `WebGLBackend` instance must not be reused across multiple renderers.
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
- `WebGLBackend.postProcessRuntime` is removed. Post-process execution remains
  backend-owned and is not part of the public backend contract.
- Public WebGL custom post-process passes must migrate to `PostProcessPass`
  instances registered through `renderer.postProcess.registerPass(pass)`.
- Forward-lighting point-light budget changed from `4` to `16` to match the
  WebGPU backend budget.
- Test entrypoints use the `tests/static/webgl/test_webgl_backend_*.mjs` prefix.
- WebGL applications on runtimes with fewer than five color attachments retain
  the existing `color`, `depth`, `motion`, and `normal` bridge channels. Passes
  requiring omitted material channels are skipped through the shared
  post-process requirement diagnostics.
- Custom render targets no longer fall back to approximate color or depth
  formats. Unsupported descriptors now fail the frame during target
  configuration.
- Custom depth attachments changed from renderbuffers to sampleable textures.

### Internal frame graph

The public `WebGLBackend` constructor and `Renderer` integration remain
unchanged. The WebGL graph is an internal implementation detail. Tests and
diagnostic tools may use internal debug state, but application code must not
depend on it as a stable public API. The shared analyzer does not change
framebuffer ownership, WebGL state changes, planner order, presentation, or the
backend-owned post-process pool. Eligible post-process passes are composed as
individual `"post-process-pass"` nodes in the authoritative whole-frame graph.
`WebGLFrameExecutor` is removed; `WebGLFrameServiceOwner` now directly owns
the internal frame-service lifecycle.

## Verification

```bash
bun tests/run_all.mjs tests/static/webgl
bun tests/static/webgl/test_webgl_backend_stub.mjs
bunx tsc --noEmit
```

## Related Documents

- [Rendering architecture](../architecture/rendering.md)
- [Render Graph architecture](../architecture/render-graph.md)
- [Post-process contract](postprocess.md)
