# WebGPU Backend Contract

This document defines WebGPU frame-graph execution, deferred lighting, presentation configuration, reflections, and structured buffer packing.

## Contract

### Internal frame graph

- `WebGPUFrameGraphPlanner` must create WebGPU internal nodes for every enabled
  renderer-level `FramePass` during frame sealing through registered stage
  planners. Sealing normally occurs in `beginFrame()` and may follow
  `particle-sim` as defined below.
- Unsupported renderer-level backend pass ids must produce an empty WebGPU
  stage plan; `WebGPUFrameOrchestrator` must warn once and skip execution for
  that pass.
- `WebGPUFrameGraphNode.reads` must list resource ids and usages sampled or
  loaded by the node.
- `WebGPUFrameGraphNode.writes` must list resource ids and usages written by
  the node.
- `WebGPUFrameGraphNode.creates` and `WebGPUFrameGraphNode.destroys` may declare
  explicit resource lifetime changes when a future node owns transient targets.
- `WebGPUFrameGraphCompiler` must compile one whole-frame definition and
  preserve planner order among retained nodes.
- `executePass()` must consume a precompiled stage slice and must not plan or
  compile the stage again.
- The WebGPU resource catalog must provide complete logical descriptors when
  metadata is known and stable physical IDs without native handles.
- `WebGPUFrameGraphCompiler` must emit a diagnostic when a non-optional read or
  destroy references an inactive resource.
- `WebGPUFrameGraphCompiler` must emit barrier records for read/write or usage
  transitions between nodes.
- `WebGPUBackendOptions.frameGraphValidation` may be `"throw"` or `"warn"`.
- `WebGPUBackendOptions.frameGraphValidation` must default to `"throw"`.
- `"throw"` mode must fail frame execution on error diagnostics.
- `"warn"` mode must emit diagnostics through `Logger.warn` and continue.
- Only enforced diagnostics may participate in `"throw"` or `"warn"` policy.
  Shared shadow diagnostics must not be logged or stop execution.
- Legacy barrier records must be projected from shared logical transitions.
  They must not emit native WebGPU synchronization commands.
- `WebGPUFrameHost` must expose only the device-scoped resource, canvas, command
  recording, submission, and validation operations required by the frame
  subsystem. Frame graph services and runtimes must not depend on
  `WebGPUBackend`.
- `WebGPUFrameHost` operations must delegate directly to their owning resource,
  pipeline, binding-group, command-scheduler, or canvas-target service. They
  must not call forwarding methods on `WebGPUBackend`.
- `WebGPUFrameFeatureAnalyzer` must scan scene, particle, reflection,
  visibility, and post-process work exactly once per frame without applying
  device capability fallbacks.
- `WebGPUFrameFeatureAnalyzer` must derive post-process frame-target
  requirements from retained `PostProcessExecutionDeclaration` values. It must
  not identify resource consumers through built-in post-process pass IDs.
- WebGPU post-process declarations must be planned before frame-target
  allocation, finalized against allocated G-buffer and shared-resource
  availability, and reused for whole-frame graph composition.
- `WebGPUFrameConfigurationResolver` must consume analyzed feature work and
  resolve only capability gating, effective configuration, and fallback policy.
- Planner, compiler, orchestrator, and debug state must use the shared typed
  graph resource catalog. The catalog must derive logical descriptors and
  stable physical bindings from concrete frame targets without exposing native
  handles.
- `WebGPUFrameTargetManager` must own WebGPU offscreen frame target allocation,
  pooled texture ownership, and target debug state. It must return allocation
  retry results and must not query or mutate orchestrator state.
- The backend-internal sample-count resolver must own request normalization,
  device capability probing, capability caches, and domain-scoped persistent
  `1x` fallback state. It must not own frame textures or one backend-wide
  active sample count.
- `WebGPUFrameOrchestrator` must own a single active frame scope and orchestrate
  target retry, frame lifecycle, and node execution; it must not own texture
  pool allocation logic.
- `WebGPUFrameServiceOwner` must be the backend-private shared-service
  composition root. It owns device-lifetime scene, texture, deferred, shadow,
  and particle-render resources; consumers must receive only the corresponding
  narrow resource-provider capability.
- `WebGPUFrameServiceOwner` must own particle-render resources through a
	delegated `WebGPUParticleRenderResources` service. The service must own only
	billboard pipelines, buffers, bindings, and pass recording; frame scopes
	retain particle-shadow-volume binding ownership.
- `WebGPUParticleRenderResources` must select particle pipeline sample counts
	from the concrete particle pass-target descriptor. It must not depend on the
	backend MSAA controller or backend-wide MSAA state.
- `FramePacketContributorRegistry` is a cross-backend internal composition
	contract. A backend must register contributors before its first preparation;
	registration must be sealed once preparation begins.
- WebGPU must register its mesh-particle packet contributor with that registry.
	The contributor must prepare packets without WebGPU device resources.
- `WebGPUBackend` must report `profile.capabilities.meshParticles = true` so
	renderer-owned requirement resolution retains mesh-particle rendering passes
	without depending on the `"webgpu"` backend identifier.
- The registry must compose baseline prepared-scene packets and contributor
	packets into one view-local `PreparedFramePacketSet`. Frame analysis, resource
	preparation, pass recording, and capture recording must consume that set
	instead of particle-specific packet accessors.
- Packet-set cache identity must include the registry, prepared scene, view
	camera, and view purpose. Cloned transients must not reuse a main-view set for
	a secondary capture view.
- Billboard-pass consumers must receive `WebGPUParticleBillboardRenderer`
	separately from scene, frame-scope, and shadow capabilities. The frame
	orchestrator may pass this capability to leaf recording runtimes during
	construction, but must not retain it as orchestrator state or resolve it
	through the concrete `WebGPUFrameServiceOwner`.
- `WebGPUFrameServiceOwner` must receive `WebGPUDeviceResourceHost` and
  `IWebGPUComputeFacade` dependencies explicitly. It must not resolve compute
  capabilities from a concrete `WebGPUBackend`.
- `WebGPUFrameResourceScope` must own frame bindings and clustered-lighting
  state. The orchestrator owns the main scope, each planar target owns one
  persistent capture scope, and probe capture must destroy its temporary scope
  in a `finally` block. Prepared-frame data must not expose a string scope key.
- Scope callers must use `WebGPUFrameScopePrepareOptions`; the frame service
  adapter must inject its private scope key through
  `WebGPUFrameServicePrepareOptions`. Distinct option shapes must not share one
  interface name.
- `WebGPUFrameBindingCache` must delegate temporal camera transaction state to
  the cross-backend `TemporalFrameState`. It may retain only the current
  snapshot needed by same-frame reuse and GPU uniform packing.
- `WebGPUDeviceResourceHost` must be the dependency boundary for registries and
  device runtimes. Such modules must not import the complete `WebGPUBackend`.
- Shared shader invalidation and destruction must be routed through
  `WebGPUFrameServiceOwner`; frame-node runtimes may invalidate only their
  pass-local resources.
- Scene, shadow, deferred, transparency, reflection, visibility, post-process,
  and presentation runtimes must own their node executors and feature-local
  pipeline/binding lifecycle. The orchestrator must not provide callback-only
  runtime wrappers for those features.
- Transparency graph nodes must separately represent OIT preparation, target
  clear, mesh accumulation, particle accumulation, resolve, transmission, and
  additive particle work. The OIT scene-color copy must occur in the prepare
  node before any accumulation node.
- `WebGPUTransparencyRuntime` owns OIT resolve shader, pipeline, sampler, and
  binding lifecycle. `WebGPUFrameTargetManager` exclusively owns OIT frame
  textures.
- Post-process and presentation must be explicit internal graph nodes.
- Planar reflection composite must be an explicit graph node after opaque or
  deferred output and before transparency.
- `WebGPUFrameSession` must own the mutable state for one frame and must expose
  a lifecycle state of `"preparing"`, `"recording"`, `"committing"`, or
  `"skipped"`.
- When the renderer frame plan includes `particle-sim`,
  `WebGPUFrameOrchestrator.beginFrame()` must create a `"preparing"` session
  without analyzing features, allocating frame targets, preparing frame
  resources, or compiling the whole-frame graph.
- After `particle-sim` emits current-frame render batches, the WebGPU backend
	must prepare the main view's composed frame packet set and seal the session
	before any graph-owned backend pass executes. Sealing must perform feature
	analysis, target configuration, resource preparation, and whole-frame
	compilation exactly once.
- Sealing a `particle-sim` session must update the main frame scope's particle
  shadow-volume bindings after resource preparation, so billboard and mesh
  particle shadow volumes consume the current frame's emitted batches.
- Frames without an enabled `particle-sim` pass must seal during
  `beginFrame()`.
- Zero-sized frames must use a `"skipped"` session without allocating an
  encoder. They must still preserve the `beginFrame`/`endFrame` lifecycle.
- `WebGPUFrameOrchestrator.beginFrame` must reject while another session is
  active. `executePass` and `endFrame` must reject when no session is active.
- `WebGPUFrameOrchestrator.executePass` must receive the same `FrameContext`
  object passed to `beginFrame`. A `"skipped"` session must ignore pass
  execution without recording commands.
- `WebGPUFrameOrchestrator.abortFrame` must remain idempotent when no session is
  active.
- `WebGPUFrameOrchestrator.endFrame` must seal graph analysis after final graph
  recording. `WebGPUFrameTransaction` must coordinate submission and
  post-submit logical commits. It must commit post-process history, temporal
  state, custom target publication, and graph analysis before deferred
  lifecycle invalidation runs. Any failure must abort unpublished state.
- Custom render-target and particle-simulation paths that bypass graph
  executors must have always-retained opaque placeholder nodes and must mark
  shared analysis coverage as `"opaque"` without changing execution.
- `WebGPUFrameCommitter` must retain labeled command buffers until `endFrame()`,
  submit them one at a time in recording order, and discard all retained work
  when recording is aborted.
- No frame runtime may submit a command buffer directly. Planar reflection
  captures must enqueue their buffers in the frame committer.
- A failure after at least one successful submission must throw
  `WebGPUFramePartialSubmitError` with submitted and pending command metadata.
- `WebGPUFrameOrchestrator` must execute graph nodes through
  `WebGPUFrameNodeExecutorRegistry`, keyed by `WebGPUFrameGraphNode.kind`.
- The node executor table must exhaustively cover `WebGPUFrameGraphNodeKind` so
  adding a node kind produces a TypeScript error until an executor is supplied.
- A planned graph node with no runtime executor must throw because it indicates
  an internal planner/runtime mismatch.
- `WebGPUPresentationRuntime` must own the presentation node executor and the
  `WebGPUPresentPass` resource lifecycle. `WebGPUFrameOrchestrator` must only
  compose that runtime and provide its narrow recording context.
- The internal WebGPU graph must not add global renderer-level stages for
  Software or WebGL.
- The frame graph may allocate a shared full-chain `frame:hiz` target when
  occlusion culling or a declared required Hi-Z consumer is active. A
  `hiz-build` node
  must run after opaque depth is available and before `occlusion-test`.
- `WebGPUHiZBuilder` owns Hi-Z shader, pipeline, mip-view, and binding caches.
  `WebGPUFrameTargetManager` owns the `frame:hiz` texture lifetime.

### Deferred lighting

- Public control contract:
	- `WebGPUBackendOptions.enableDeferredLighting` must default to `true`.
	- When `enableDeferredLighting === false`, `WebGPUBackend` must not run the
	  deferred lighting resolve and must route opaque materials through the
	  legacy MRT forward path when MRT is available.
	- When `enableDeferredLighting !== false`, `WebGPUBackend` must attempt to
	  enable deferred lighting and must warn once if any runtime requirement
	  prevents correct activation.
	- `WebGPUBackend.isDeferredLightingEnabled()` must return the configured
	  public switch value.
- Runtime gating contract:
	- Deferred lighting must require `sampleCount === 1`.
	- Base deferred lighting must require `maxColorAttachments >= 4` and
	  `maxColorAttachmentBytesPerSample >= 32`.
	- Extended deferred lighting must require `maxColorAttachments >= 7` and
	  `maxColorAttachmentBytesPerSample >= 56`.
	- Deferred lighting must require
	  `maxStorageTexturesPerShaderStage >= 2`.
	- Color-attachment byte requirements must be derived through the WebGPU-owned
	  `getWebGPURenderTargetPixelByteCost()` lookup rather than texel storage size
	  or duplicated literals. Backend-agnostic `TextureFormatInfo` metadata must
	  not contain WebGPU-specific attachment costs.
	- If any requirement is not met, `WebGPUBackend` must use the legacy MRT
	  forward path when available, otherwise the single-target forward path,
	  and must warn once.
- Pass ordering contract:
	- `main-opaque` must render environment/background into `sceneColorMain`
	  before lighting resolve when background rendering is enabled.
	- Builtin opaque and mask deferred materials must render surface payloads
	  into the G-buffer.
	- Deferred lighting must run as a fullscreen render pass that reads the
	  G-buffer, shadow data, environment data, and clustered-light buffers, then
	  writes `sceneColorMain`.
	- Deferred lighting must discard pixels with `gMotionDepth.z <= 0`, preserving
	  prior background color.
	- Non-deferred opaque fallback materials must render after deferred lighting
	  through the legacy MRT forward shader.
	- Transparent, OIT, transmission, and particles must render after opaque
	  lighting resolve through existing forward paths.
- G-buffer contract:
	- The frame analyzer must select `base` or `extended` deferred payload mode.
	  `ShaderMaterial` deferred chunks and deferred decals must conservatively
	  select `extended` mode.
	- `base` mode must expose only `gAlbedoAlpha`, `gNormalRoughMetal`,
	  `gEmissiveOcclusion`, and `gMotionDepth`. It must use no more than
	  `24` color bytes per pixel, excluding depth.
	- `extended` mode may additionally expose `gSpecular`, `gCoatSheen`,
	  `gSheenReflectance`, `gMaterialExt0`, and `gMaterialExt3`. Its complete
	  deferred payload must use no more than `60` bytes per pixel, excluding
	  depth.
	- Color MRTs must be:
	  `gAlbedoAlpha`, `gNormalRoughMetal`, `gEmissiveOcclusion`,
	  `gMotionDepth`, `gSpecular`, `gCoatSheen`, and `gSheenReflectance`.
	- `gAlbedoAlpha`, `gNormalRoughMetal`, and `gSheenReflectance` must use
	  `rgba8unorm`. `gEmissiveOcclusion`, `gMotionDepth`, `gSpecular`, and
	  `gCoatSheen` must use `rgba16float`.
	- `gNormalRoughMetal.xy` must store the encoded world-space normal,
	  `gNormalRoughMetal.z` must store roughness, and
	  `gNormalRoughMetal.w` must store metallic.
	- `gSpecular.rgb` must store the resolved specular color. `gSpecular.a` must
	  store the PBR specular factor or the unclamped Phong/Flat shininess,
	  according to the packed shading model.
	- Deferred storage payload textures must be `gMaterialExt0` as
	  `rgba16float` and `gMaterialExt3` as `rgba16uint`. Deferred opaque shading
	  must not store transmission volume or attenuation payloads.
	- `gMaterialExt0.xy` must store the encoded clearcoat normal,
	  `gMaterialExt0.z` the iridescence factor, and `gMaterialExt0.w` the
	  iridescence thickness.
	- `gMaterialExt3.xy` must store the encoded world-space anisotropy tangent,
	  `gMaterialExt3.z` must store the resolved anisotropy strength, and
	  `gMaterialExt3.w` must store the 11-bit receiver render-layer mask.
	- `gMotionDepth.w` must store an exactly representable packed material word.
	  Bits `0..1` contain the shading model; higher bits identify clearcoat,
	  sheen, iridescence, anisotropy, and non-default specular/reflectance data.
	- Deferred lighting must read the base payload first and must not load an
	  extension texture when the packed material word says that extension is
	  absent.
	- Forward and deferred opaque lighting must use the shared Phong and PBR
	  direct-light evaluators. Forward-only transmission background and volume
	  terms must remain outside those shared opaque evaluators.
	- When no deferred packet or decal needs `gMaterialExt0`, the target manager
	  must bind a device-lifetime placeholder instead of allocating a full-size
	  texture.
	- The deferred lighting shader must branch on `PBR`, `Phong`, `Flat`, and
	  `Unlit` shading models inside the same fullscreen pass.
	- `gSpecular.a` must decode as unclamped Phong shininess for `Phong` and
	  `Flat`, and as a `[0, 1]` specular factor only for `PBR`.
	- Opaque and mask `PBRMaterial` instances with `anisotropyStrength > 0.0`
	  or `anisotropyMap` may enter deferred lighting when all runtime gates pass.
	- `transmissionFactor > 0.0`, `AlphaMode.Blend`, OIT, and transparent
	  particles must remain on forward transparent paths.
- `ShaderMaterial` contract:
	- `ShaderTargetMode` must include `"deferred"`.
	- `ShaderStageKind` must include `"fragment-deferred"`.
	- `ShaderMaterialParams.deferredLighting` must opt a shader material into
	  deferred routing.
	- `ShaderMaterialParams.fragmentDeferredEntryPoint` must select the deferred
	  fragment entry point and must default to `fsMainDeferred`.
	- A `ShaderMaterial` must enter the G-buffer path only when
	  `deferredLighting === true` and a WebGPU deferred fragment chunk exists.
	- Non-opt-in `ShaderMaterial` instances must render after lighting through
	  the legacy MRT forward fallback.
- Camera reconstruction contract:
	- Perspective camera packing must keep `environmentBasisRight.w` as
	  `tanHalfFov` and `environmentBasisUp.w` as `aspect`.
	- Orthographic camera packing must store `halfWidth` in
	  `environmentBasisRight.w` and `halfHeight` in `environmentBasisUp.w`.
	- `environmentBasisBackward.w` must remain the orthographic flag.

### Sample-count configuration

- `WebGPUBackendOptions.sampleCount?: number`
	- Input contract: accepts a finite number. The value is floored and clamped
	  to at least `1`.
	- Behavior contract: omitted values request the default `1x` sample count.
	  Values greater than `1` request main-scene multisampling. The active count
	  may be lower than requested when device capabilities do not support the
	  requested count. Custom render targets use their own descriptor count.
	- Error contract: non-finite values must throw a configuration error.
- Sample-count runtime control is internal. `WebGPUBackend` must not expose
	`getMSAASampleCount()`, `setMSAAEnabled()`, or `setMSAASampleCount()`.
- The legacy `enableMSAA`, `msaaSampleCount`, and `sceneSampleCount` options are
	removed. JavaScript callers that supply any of them must receive a deterministic
	error directing them to `sampleCount`.

### Custom render-target sample counts

- Each custom render target must use an independent sample-count domain named
	`custom-target:<id>`.
- Selection must use the normalized request and the complete color/depth format
	set. Every attachment in one target must use the same effective sample count.
- When the effective count is greater than `1`, each color attachment must own
	a multisampled render texture and a single-sample resolve texture. Depth must
	remain multisampled and must not receive an automatic resolve texture.
- Allocation failure must destroy every partially allocated render, resolve,
	and depth texture, pin only that target signature to `1x`, and retry during
	the same synchronization operation. The pin must survive resize and clear
	only when the device runtime is reset.
- Custom pass pipelines must use `context.target.sampleCount`, and color
	readback must use `resolveTexture` when it is present.

### Planar reflections

- WebGPU must honor `Material.reflectivity` when `Material.mirrorPlane` is set.
- WebGPU must report
  `WebGPUBackend.profile.capabilities.reflection === true`.
- WebGL must continue to report
  `WebGLBackend.profile.capabilities.reflection === false`.
- WebGPU must support at most `WEBGPU_PLANAR_REFLECTION_MAX_PLANES = 2` active
  mirror planes per frame.
- WebGPU must capture planar reflection targets at
  `WEBGPU_PLANAR_REFLECTION_RESOLUTION_SCALE = 0.5` of the frame size.
- Reflection capture must render into a single offscreen color target.
- Reflection capture color target must use `rgba16float`.
- Reflection capture must include environment, opaque packets, and transparent
  packets.
- Reflection capture must exclude receivers that use the same normalized mirror
  plane as the active capture plane.
- Reflection capture must set `enableReflection` to `false`.
- Reflection capture must set SSR post-processing to disabled.
- Reflection capture must reuse current-frame shadow resources and must not
  schedule a dedicated shadow recapture pass.
- Reflection capture command buffers must remain pending in the frame committer
  until `endFrame()` and must be discarded without submission when the frame is
  aborted before commit.
- Reflection capture must use a `reflection-capture` pipeline variant with
  flipped front-face winding.
- Main scene rendering must render the base material first.
- Composite must run after opaque or deferred lighting output and before
  transparent scene rendering.
- Capture and composite must be explicit internal frame graph nodes.
- Composite must write a planar reflection mask target for mirror receiver
  pixels.
- SSR compose must skip pixels marked by the planar reflection mask.
- MSAA rendering must composite into the MSAA scene color and resolve into the
  main scene color target.
- Non-MSAA rendering may composite directly into the main scene color target.
- Orthographic cameras must skip planar reflection and emit a once-only
  diagnostic.
- Roughness blur, recursive planar reflection, per-material resolution,
  particle reflection, and WebGL parity are not supported.

### Structured buffer packing

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

The frame packers share `WebGPUFrameUniformInput` and target these
`sceneFrameBindGroupLayout` resources:

| Packer | Binding | Layout | Size |
| --- | --- | --- | --- |
| `packFrameCameraUniformData` | `0` | `FrameCameraUniforms` | 288 bytes |
| `packFrameLightUniformData` | `14` | `FrameLightUniforms` | 1,680 bytes |
| `packFrameShadowUniformData` | `15` | `FrameShadowUniforms` | 5,760 bytes |
| `packFrameEnvironmentUniformData` | `16` | `FrameEnvironmentUniforms` | 4,208 bytes |

`FrameCameraUniforms.options` must preserve its four-lane layout. The lanes
must contain lighting enablement in `x`, zero in the reserved `y` lane, shadow
enablement in `z`, and zero in the reserved `w` lane.
`WebGPUFrameUniformInput` must not contain a gamma pass enablement field;
backend post-process runtime state owns that decision.
`WebGPUFrameUniformInput` must only expose values written by at least one frame
packer. Backend resource-presence state that is not represented in a frame
uniform layout must remain outside this input contract.

## Usage

### Internal frame graph

```ts
const backend = new WebGPUBackend({
	frameGraphValidation: "throw",
});
```

```bash
bun tests/static/webgpu/test_webgpu_frame_graph_compiler.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_planner.mjs
bun tests/static/webgpu/test_webgpu_frame_node_executor_registry.mjs
bun tests/static/webgpu/test_webgpu_frame_executor_whole_frame_planning.mjs
```

### Deferred lighting

```ts
import { WebGPUBackend } from "../src/backends/webgpu/WebGPUBackend";

const backend = new WebGPUBackend({
	enableDeferredLighting: false,
});
```

```ts
import { ShaderMaterial } from "../src/materials/ShaderMaterial";

const material = new ShaderMaterial({
	deferredLighting: true,
	fragmentDeferredEntryPoint: "fsDeferred",
	chunks: [
		{
			language: "wgsl",
			stage: "fragment",
			mode: "deferred",
			code: deferredFragmentWGSL,
		},
	],
});
```

```bash
bun tests/static/shaders/test_shader_material.mjs
bun tests/static/webgpu/test_webgpu_bridge_material_pipelines.mjs
bun tests/static/webgpu/test_webgpu_frame_executor_transparency_deferred.mjs
```

### Sample-count configuration

```ts
import { WebGPUBackend } from "../src/backends/webgpu/WebGPUBackend";

const backend = new WebGPUBackend({ sampleCount: 1 });
```

```bash
bun tests/static/webgpu/test_webgpu_backend_cache_and_dependency.mjs
```

### Planar reflections

```ts
import { Material } from "../src/materials/Material";
import { Plane } from "../src/maths/Plane";

const mirror = new Material({
	name: "water-mirror",
	reflectivity: 0.8,
	mirrorPlane: new Plane({ x: 0, y: 1, z: 0 }, 0),
});
```

```bash
bun tests/static/renderer/test_backend_capabilities.mjs
bun tests/static/webgpu/test_webgpu_frame_executor_reflection_refraction.mjs
```

The WebGPU frame planner should schedule the `reflection` frame pass when
`enableReflection` is true and the prepared scene contains reflective packets.

### Structured buffer packing

```ts
import {
	StructuredBufferLayout,
	arrayOf,
	mat4x4f32,
	structOf,
	vec,
} from "../src/backends/webgpu/StructuredBufferLayout";
import {
	arrayVec4,
	createStructuredBufferPacker,
	mat4,
	vec4,
} from "../src/backends/webgpu";

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

## Diagnostics

### Internal frame graph

- `read-before-create` must trigger when a required node read references an
  inactive resource.
- `destroy-before-create` must trigger when a required node destroy references
  an inactive resource.
- `duplicate-create` must trigger when a required node creates an active
  resource.
- `webgpu-frame-graph-validation` must be logged when validation mode is
  `"warn"` and error diagnostics exist.
- `webgpu-pass-unsupported-{stage}` must be logged once when a renderer-level
  backend pass has no WebGPU frame graph stage planner.
- `WebGPUFrameOrchestrator has no active frame session.` must report pass or
  frame completion outside an active frame lifecycle.
- `WebGPU frame pass context must match the context passed to beginFrame().`
  must report a mismatched per-pass `FrameContext`.
- `webgpu-hiz-build-failed` must leave opaque rendering active, make occlusion
  candidates visible, and prevent Hi-Z-dependent post-process passes from
  running for the affected frame.
- `WebGPUFramePartialSubmitError` must identify the failure phase, original
  cause, submitted count, total count, submitted labels, and pending labels.
- `getFrameGraphDebugState().graphAnalysis` must group `current`,
  `lastAttempt`, and `lastSuccessful` snapshots. Shadow diagnostics such as
  `read-content-unknown` and `opaque-stage-effects` must appear only there.

### Deferred lighting

- `webgpu-deferred-disabled-msaa`:
  emitted when deferred lighting is unavailable because `sampleCount !== 1`.
- `webgpu-deferred-disabled-attachments`:
  emitted when the active base or extended layout exceeds
  `maxColorAttachments`.
- `webgpu-deferred-disabled-bytes`:
  emitted when the active base or extended layout exceeds
  `maxColorAttachmentBytesPerSample`.
- `webgpu-deferred-disabled-storage-textures`:
  emitted when `maxStorageTexturesPerShaderStage < 2`.
- `webgpu-deferred-runtime-fallback`:
  emitted when deferred frame target allocation, binding creation, or pipeline
  creation fails after limits passed. Pipeline and binding creation must be
  preflighted before the first G-buffer command so the frame can atomically use
  legacy forward rendering.
- Shader compile diagnostics for opt-in `ShaderMaterial` deferred chunks must
  follow the existing `ShaderMaterial` strict/warn/silent runtime behavior.

### Sample-count configuration

- Requested MSAA counts that are unsupported must be clamped to the highest
  supported count that does not exceed the request, and may end at `1x`.
- A frame-target allocation failure may log
  `webgpu-scene-sample-count-runtime-fallback-1x`
  and retry at `1x` before recording render commands. The fallback remains
  active until the device runtime is reinitialized.
- A custom-target allocation failure may log
  `webgpu-custom-target-sample-count-runtime-fallback-1x`. Its fallback applies
  only to the matching device, target domain, normalized request, and attachment
  signature, and remains active until the device runtime is reinitialized.

### Planar reflections

- `[webgpu-planar-reflection-orthographic-disabled]` is emitted once when the
  active camera is orthographic.
- `[webgpu-planar-reflection-mask-unavailable]` is emitted once when composite
  is requested without a planar reflection mask target.
- If mirror pixels receive SSR over planar reflection, verify the SSR compose
  binding includes `planarReflectionMask`.
- If mirrored geometry appears culled, verify the draw uses the
  `reflection-capture` pipeline variant.
- If the reflection capture target fails WebGPU pipeline validation, verify the
  capture draw uses the offscreen single-target `color` mode instead of the
  canvas `single` mode or the scene MRT mode.
- If same-plane receivers appear recursively in the reflection, verify capture
  filtering removes packets whose normalized `mirrorPlane` key matches the
  active capture plane.
- If reflection target memory grows after mirror removal, verify stale capture
  targets are destroyed when planes are no longer active.

### Structured buffer packing

- Unknown field path: triggered when a helper path does not exist in the
  `StructuredBufferLayout`.
- Invalid array length: triggered when `arrayVec4` or `arrayStruct` receives a
  negative or non-integer `length`.
- Writer byte-size mismatch: triggered when `packInto` receives a writer that
  was not created for the same layout byte size.
- Invalid scalar, vector, or matrix value: triggered by `StructuredBufferWriter`
  when a resolver returns an incompatible value.

## Compatibility

### Internal frame graph

`getFrameGraphDebugState()` may expose structured internal graph diagnostics,
barriers, resources, and target-manager state. Tests and diagnostic tooling must
not depend on private runtime fields when equivalent graph debug data exists.
The planner and runtime use internal registries instead of switch statements;
this does not add public WebGPU frame graph registration APIs. Rejecting
duplicate frame begins, missing active sessions, and mismatched frame-context
identity strengthens internal lifecycle validation without changing the public
renderer API. WebGPU composes each eligible post-process pass into the
whole-frame graph under the `"postprocess"` namespace. The shared analyzer does
not transfer native resource ownership, allocate pool textures, emit native
barriers, or change planner and executor order.

### Deferred lighting

No public renderer frame-pass stage is added. Builtin opaque and mask WebGPU
materials may use deferred lighting by default when runtime limits allow it.
Transparent, OIT, transmission, particles, `SoftwareBackend`, `WebGLBackend`,
and reflection probe capture keep their existing paths.

### Sample-count configuration

This change is breaking. Runtime MSAA methods, `enableMSAA`,
`msaaSampleCount`, and `sceneSampleCount` are removed; use `sampleCount` when
creating `WebGPUBackend`. The default scene sample count is `1x`; applications
that require multisampled main-scene rendering must request it explicitly.

### Planar reflections

- Behavior change: WebGPU now renders planar reflection for materials that set
  `reflectivity` and `mirrorPlane`.
- Behavior change: WebGPU no longer emits unsupported-material warnings for
  `reflectivity` or `mirrorPlane`.
- Behavior change: WebGPU planar reflection capture uses a single offscreen
  `rgba16float` color target instead of writing scene MRT G-buffer targets.
- Backend compatibility: Software keeps its existing reflection behavior.
- Backend compatibility: WebGL remains without planar reflection support.
- Capability inspection must use `Renderer.backendProfile` or an explicit
  attached backend.

### Structured buffer packing

`packFrameUniformData` has been replaced by
`packFrameCameraUniformData`, `packFrameLightUniformData`,
`packFrameShadowUniformData`, and `packFrameEnvironmentUniformData`. Consumers
must select the packer matching the target frame uniform binding.

This is a breaking WebGPU shader contract. Custom WGSL must retain camera and
global settings through `frame` at binding `0`, and must read lights, shadows,
and probe data through `frameLights`, `frameShadows`, and `frameEnvironment`
at bindings `14`, `15`, and `16` respectively.

## Verification

```bash
bun tests/run_all.mjs tests/static/webgpu
bun tests/static/webgpu/test_webgpu_post_graph.mjs
bunx tsc --noEmit
```

## Related Documents

- [WebGPU architecture](../architecture/webgpu.md)
- [Render Graph architecture](../architecture/render-graph.md)
- [WebGPU bindings reference](../reference/webgpu-bindings.md)
