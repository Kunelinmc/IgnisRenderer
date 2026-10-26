# WebGPU Backend Contract

This document defines WebGPU frame-graph execution, geometry packing, deferred
lighting, presentation configuration, reflections, and structured buffer packing.

## Contract

### Shader directive profile lifecycle

- `WebGPUBackend` must prepare its static directive profile assets during
  initialization, after an adapter is available and before shader compilation.
- The backend must compose an instance overlay from resolved logical limits,
  binding ABI values, and validated device capabilities after device creation.
- Device restoration must rebuild the instance overlay, directive profile,
  compile stage, and dependent shader caches before the restored runtime is
  published.
- WebGPU custom-material source augmentation must generate WGSL uniform,
  texture, sampler, and sampling-helper declarations from resolved material
  bindings. It must pass generated blocks through `ShaderBackendCompileStage`
  rather than emitting engine-owned `#inject` directives.
- WebGPU material ABI revision must participate in custom shader and pipeline
  cache identity. Generated material blocks must be created only during shader
  program resolution or cache miss, never during frame draw iteration.

### Internal frame graph

- `WebGPUCommandEncoder.setBindingGroup()` must omit the native `setBindGroup()`
  third argument when `dynamicOffsets` is `undefined`, for both render and
  compute passes. Supplied offset sequences, including empty sequences, must
  be forwarded unchanged.
- WebGPU canvas configurations for SDR and HDR must use premultiplied alpha.
  Opaque presentation must force final alpha `1.0`; transparent presentation
  must preserve premultiplied alpha from the final color resource.
- Main scene and dirty-region clears must use alpha `0.0` only for transparent
  presentation. Environment background rendering must write alpha `1.0`.
- Presentation must safely unpremultiply for output-range handling and
  premultiply the resulting RGB again. It must follow the shared
  incomplete-color-domain warning contract in `docs/contracts/rendering.md`
  and must not apply missing display conversions.
- Registered frame modules must create WebGPU internal nodes for every enabled
  renderer-level `FramePass` during frame sealing through their stage
  contributions. Sealing normally occurs in `beginFrame()` and may follow
  `particle-sim` as defined below.
- `WebGPUFrameGraphModuleRegistry` must be backend-private. The WebGPU backend
  must register every built-in frame module during runtime composition and
  seal the registry before the first frame begins. Applications must not
  receive a frame-module or frame-graph registration API.
- A sealed frame-module registry must reject later registration. Sealing must
  reject duplicate module ids, duplicate owner-and-kind executors, and missing
  node-kind executors.
  The sealed module list must be immutable at runtime.
  Stage planning must reject duplicate planned node ids, unordered same-lane
  contributors, lane cycles, and multiple composed subgraphs for one renderer
  stage before graph compilation.
- Frame modules must select a general stage lane from `setup`, `geometry`,
  `lighting`, `composite`, `visibility`, `transparent`, `postprocess`, and
  `present`. Contributors in the same lane must declare static `before` or
  `after` edges. Registration order must not affect planned graph order.
- Planned node identity must use stage, owner id, and module-local id. Node
  execution must route by owner id and kind, so multiple modules may own the
  same kind without ambiguity.
- Frame-module analysis, configuration, and planning must use backend-private
  typed message descriptors. Descriptors must declare a unique id, owner,
  phase, cardinality, and handler inputs and outputs.
- The frame-message registry must validate descriptor identity, required
  producers, single-value producer cardinality, forward-only phase edges, and
  an acyclic same-phase handler graph before the first frame.
- Same-wave message handlers may execute concurrently, but their isolated
  outputs must merge deterministically by wave, module id, handler id, and
  publication order. A handler failure must discard all provisional output for
  the phase before the frame transaction aborts.
- Analysis output must seal before configuration begins. Allocation fallback
  retries may rerun configuration against the same sealed analysis snapshot and
  must not rescan scene packets.
- Post-process planning must publish the final color resource and color domain
  as a typed per-frame output message. Presentation must consume the sealed
  publication and must not retain a post-process module callback.
- Feature modules must publish capability policy and resource demands from
  their retained analysis. The configuration reducer may merge generic target
  classes, resource demands, diagnostics, and conflicts; it must not contain
  feature-specific packet scans.
- Scene, shadow, deferred, transparency, reflection, visibility, post-process,
  presentation, and custom-render-target modules must own their feature-local
  analysis, graph contribution, node execution, warmup, and lifecycle work.
- WebGPU device initialization must construct resource owners without compiling
  scene, environment, deferred, or reflection shader modules and pipelines.
  Those resources must compile lazily on first use or explicit warmup.
- Scene, environment, deferred, and reflection owners must enumerate their own
  warmup variants from the current frame inputs, resolved backend options,
  device capabilities, and sample count. `WebGPUFrameRuntimeComposition` must
  expose one backend-private warmup entrypoint that plans warmup work, invokes
  post-process/presentation, frame services, and reflection in that order, and
  aggregates their phase reports and progress notifications. It must not
  reproduce feature pipeline-selection policy or execute a normal frame.
- `WebGPUBackend.warmup()` must validate runtime availability and manage
  compilation logging, resetting logging in a `finally` block. One warmup call
  must use one runtime composition throughout its work. Warmup planning failures
  must reject; compilation failures must remain in `WarmupReport.errors`.
  The composition must retain cooperative yielding and must not expose
  separate post-process or reflection warmup ports.
- Deferred warmup must reuse the same analysis and capability policy as frame
  configuration. Reflection warmup must own its temporary frame scope and
  release it in a `finally` block.
  Modules must receive narrow runtime capabilities and must not receive the
  concrete `WebGPUFrameOrchestrator`.
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
- Registered frame modules must collectively scan scene, particle, reflection,
  visibility, and post-process work exactly once per frame without applying
  device capability fallbacks.
- The post-process frame module must derive frame-target requirements from
  retained `PostProcessExecutionDeclaration` values. It must not identify
  resource consumers through built-in post-process pass IDs.
- WebGPU post-process declarations must be planned before frame-target
  allocation, finalized against allocated G-buffer and shared-resource
  availability, and reused for whole-frame graph composition.
- Post-process finalization must receive the allocated frame-target view during
  frame sealing. It must not depend on the later recording-session execution
  binding to discover G-buffer channels or shared-resource allocation.
- Feature-owned configuration handlers must apply capability and fallback
  policy, then publish typed demands. `WebGPUFrameConfigurationModule` must
  reduce only generic target classes, logical resource requirements, feature
  states, diagnostics, and incompatible exclusive demands.
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
- `WebGPUFrameOrchestrator` must own a single active-session slot and orchestrate
  target retry, frame lifecycle, graph compilation, and node execution. It must
  not own texture-pool allocation, command-stream state, feature-local frame
  state, or runtime feature capabilities.
- `WebGPUFrameServiceOwner` must be the backend-private shared-service
  composition root. It owns device-lifetime scene, texture, deferred, shadow,
  and particle-render resources; consumers must receive only the corresponding
  narrow resource-provider capability.
- Each frame-scope preparation must normalize scene lights into one
  `WebGPULightingCatalog` and derive lighting data through stateless functions
  in `src/backends/webgpu/lights.ts`. Clustered-light preparation must preserve
  camera eligibility, light budgets, and diagnostics. Volumetric lighting must
  use the effective surface-light selection without duplicating legacy and
  clustered lights.
- Clustered-light data and the intermediate surface-light view must remain
  local to scope preparation. `WebGPUPreparedFrameResources` must expose
  `volumetricLightingData` as a typed nullable field, populated only while the
  volumetric pass is enabled. Preparing another scope or disabling the pass
  must not overwrite a previously prepared scope's lighting data.
- `WebGPUScenePipelineResources` must own only built-in and custom scene shader
  modules together with forward, G-buffer, Early-Z, and scene-capture pipeline
  variants. It must not own environment, deferred-lighting, or planar-reflection
  composite pipelines, and it must consume immutable material pipeline state.
- `WebGPUEnvironmentResources` must own the environment shader and render
  pipeline handles shared by main-scene, planar-reflection, and reflection-probe
  capture paths. Capture runtimes must not create private copies.
- `WebGPUDeferredResources` must directly own deferred-lighting and decal shader
  and pipeline handles. It must not recover deferred-lighting pipelines through
  a callback into a shared scene pipeline owner.
- `WebGPUTextureRegistry` must explicitly release registry-owned texture and
  sampler handles when `Texture.dispose()` is called. It must also release them
  after an unreachable `Texture` is finalized when `FinalizationRegistry` is
  available. Externally owned texture handles must never be destroyed by this
  cleanup.
- `WebGPUFrameServiceOwner` must own particle-render resources through a
	delegated `WebGPUParticleRenderResources` service. The service must own only
	billboard pipelines, buffers, bindings, and pass recording; frame scopes
	retain camera binding ownership. Billboard particle passes must not
  bind shadow textures, comparison samplers, or shadow uniforms.
  Billboard particle frame-graph nodes must not declare shadow-resource reads.
- `WebGPUParticleRenderResources` must select particle pipeline sample counts
	from the concrete particle pass-target descriptor. It must not depend on the
	backend MSAA controller or backend-wide MSAA state.
- `prepareFramePackets` is a stateless pipeline composition function. Frame
	analysis, resource preparation, pass recording, and capture recording must
	consume the returned `PreparedFramePacketSet` instead of particle-specific
	packet accessors.
- `WebGPUBackend` must report `profile.capabilities.meshParticles = true` so
	renderer-owned requirement resolution retains mesh-particle rendering passes
	without depending on the `"webgpu"` backend identifier.
- Packet-set cache identity must include the prepared scene, view camera, and view
	purpose in `context.transient`. Cloned transients must not reuse a main-view
	set for a secondary capture view.
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
  persistent reflection scope, and each render-target view job owns one
  temporary scope. A recorded view scope must remain alive through
  `queue.submit()` and must be released by the module `afterSubmit` hook; an
  aborted transaction must discard it through `abortFrameState`. Prepared-frame
  data must not expose a string scope key.
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
  presentation, and custom-render-target modules must own their node executors
  and feature-local pipeline/binding lifecycle. The orchestrator must not
  construct concrete feature runtimes, retain feature passes, or provide
  callback-only runtime wrappers for those features.
- `WebGPUShadowRuntime` must be the sole frame-facing entry point for WebGPU
  shadows. It must consume the atlas-oriented `FrameContext.shadowPlan`, directly
  own the atlas allocator and caster renderer, coordinate atlas preparation and
  frame execution, and publish backend-private sampling textures. It must not
  mutate the shared plan.
- `WebGPUShadowAtlasAllocator` must own depth and transmittance atlas allocation,
  growth, device-limit clamping, and texture destruction.
- `WebGPUShadowCasterRenderer` must own shadow pipelines, transient draw
  resources, and depth/transmittance command encoding. It must borrow atlas
  resources from the runtime-owned allocator without retaining or destroying
  that allocator. It must not own capability decisions or graph topology.
- `WebGPUShadowFrameModule` must own shadow graph nodes and their logical
  resource writes, and execute them through the runtime's recording capability.
- Frame consumers must obtain sampling resources through
  `WebGPUShadowRuntime` instead of retaining its allocator or caster renderer.
- WebGPU atlas sampling must implement the shared PCF and PCSS quality presets.
  One logical PCF tap must use bilinear comparison sampling.
- Transparency graph nodes must separately represent OIT preparation, target
  clear, mesh accumulation, particle accumulation, resolve, transmission, and
  additive particle work. The OIT scene-color copy must occur in the prepare
  node before any accumulation node.
- Shared transparency attachment builders must take an explicit shadow-read
  option. Mesh and transmission paths must enable shadow-resource reads;
  billboard particle paths must disable them so those nodes declare attachment
  reads only.
- `WebGPUTransparencyRuntime` owns transparency graph planning and node
	execution together with the OIT resolve shader, pipeline, sampler, and binding
	lifecycle. It must not delegate planning to a separate transparency planner.
	`WebGPUFrameTargetManager` exclusively owns OIT frame textures.
- Post-process and presentation must be explicit internal graph nodes.
- Planar reflection composite must be an explicit graph node after opaque or
  deferred output and before transparency.
- `WebGPUFrameSession` must be a discriminated lifecycle state of
  `"preparing"`, `"recording"`, `"committing"`, or `"skipped"`. A recording or
  committing session must contain non-null configuration, prepared resources,
  frame packets, sealed messages, target view, and command stream.
- Frame sealing must use provisional local data and must publish a recording
  session only after analysis, target configuration, graph compilation, and
  main-scope resource preparation all succeed.
- `WebGPUFrameCommandStream` must exclusively own the current encoder, labeled
  encoder rotation, committer integration, ordered submission, abort, and
  partial-submit diagnostics. Feature modules must not assign an encoder or
  committer field on `WebGPUFrameSession`.
- `WebGPUFrameTargetManager` must publish a stable frame-scoped target view with
  attachments, MSAA attachments, dimensions, sample count, and scene-target
  mode. Publishing the view must not transfer resource ownership.
- An optional `WebGPUFrameDiagnosticsObserver` may receive immutable session,
  target, graph, node-execution, and commit snapshots. The observer must not be
  exposed as a runtime capability, and absent observers must not cause
  per-event allocation.
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
  instance bindings after resource preparation, so billboard and mesh particle
  rendering consumes the current frame's emitted batches.
- Frames without an enabled `particle-sim` pass must seal during
  `beginFrame()`. `WebGPUBackend.beginFrame()` must return a promise and callers
  must await asynchronous frame-message dispatch and graph sealing before pass
  execution.
- Zero-sized frames must use a `"skipped"` session without allocating an
  encoder. They must still preserve the `beginFrame`/`endFrame` lifecycle.
- `WebGPUFrameOrchestrator.beginFrame` must reject while another session is
  active. `executePass` and `endFrame` must reject when no session is active.
- `WebGPUFrameOrchestrator.executePass` must receive the same `FrameContext`
  object passed to `beginFrame`. A `"skipped"` session must ignore pass
  execution without recording commands.
- `WebGPUFrameTransaction` must use `abortRecording()` and `abortFrameState()`
  for recording and logical-state rollback. `WebGPUFrameOrchestrator` must not
  expose compatibility frame or graph-analysis aliases.
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
  `WebGPUFrameNodeExecutorRegistry`, keyed by node owner and kind. Multiple
  modules may own the same generic node kind without sharing an executor.
- `WebGPUFrameOrchestrator` must depend only on the sealed frame-module registry
  for feature execution. Runtime composition must be backend-owned and must
  expose separate narrow post-process, visibility, custom-target, lifecycle,
  and warmup ports. Adding a built-in frame feature must not require an
  orchestrator change.
- `WebGPUFrameNodeExecutor` must receive an explicit complete per-frame
  execution context. Frame modules and leaf runtimes must not resolve current
  session data through a callback-based recording context.
- Hi-Z readiness and build diagnostics must be owned by the visibility module;
  motion-history write state by the post-process module; presentation status by
  the presentation runtime; deferred opaque state by a deferred state port; and
  transparency runtime fallback by the transparency runtime.
- The node executor table must exhaustively cover `WebGPUFrameGraphNodeKind` so
  adding a node kind produces a TypeScript error until an executor is supplied.
- A planned graph node with no runtime executor must throw because it indicates
  an internal planner/runtime mismatch.
- `WebGPUPresentationRuntime` must own the presentation node executor,
  per-frame presentation status, and the `WebGPUPresentPass` resource
  lifecycle. It must consume the explicit frame execution context.
- The internal WebGPU graph must not add global renderer-level stages for
  Software or WebGL.
- The frame graph may allocate a shared full-chain `frame:hiz` target when
  occlusion culling or a declared required Hi-Z consumer is active. A
  `hiz-build` node
  must run after opaque depth is available and before `occlusion-test`.
- `WebGPUHiZBuilder` owns Hi-Z shader, pipeline, mip-view, and binding caches.
  `WebGPUFrameTargetManager` owns the `frame:hiz` texture lifetime.

### Animation payload resources

- Scene, planar-reflection composite, and shadow vertex shaders must import
  shared morph and skinning helpers from `ignis/webgpu/animation`, backed by
  `src/shaders/webgpu/common/animation.wgsl`. Consumers must apply morph before
  skinning and supply explicit pose offsets; morph delta offsets are measured
  in three-component vertices, not scalar storage elements.
- The animation include must leave resource bindings and `AnimationParams`
  layouts to each consumer. Shadow shaders must define
  `IGNIS_WEBGPU_ANIMATION_POSITION_ONLY` as `1` before importing it so the
  compiled helpers require neither morph-normal storage nor direction
  normalization. Scene consumers must retain normal and tangent deformation.
- `WebGPUFrameServiceOwner` must own one device-lifetime
  `WebGPUAnimationPayloadPool` shared by scene-material and shadow consumers.
  The pool must own packet animation parameter, joint-matrix, morph-weight, and
  zero-storage fallback buffers. Scene-material and shadow runtimes must retain
  ownership of their feature-local bind groups.
- The pool must key payload state by `DrawSubmission.id` and must resolve and upload
  joint and morph storage at most once per logical backend frame. Runtime joint
  matrices and morph weights must be consumed directly without recomputing
  skeleton matrices or allocating another morph snapshot in WebGPU consumers.
- Animation consumers must resolve runtime payloads only through the
  submission deformation binding keys. They must not recover skeleton or morph
  state from scene, mesh, or primitive authoring objects. When an active
  deformation binding lacks a required current-frame payload, every scene
  consumer must skip that packet, every shadow consumer must skip that
  submission, and each must emit one deduplicated diagnostic.
- Packets without active skinning or morph targets must use device-lifetime
  zero-value fallback buffers and must not allocate packet-owned animation
  buffers. Scene and shadow animation parameter layouts may remain distinct.
- Joint and morph storage must retain current and previous payloads. A changed
  deformation must publish the last current payload as previous; the next
  unchanged logical frame must settle previous to current before later
  unchanged frames skip the upload.
- Packet-owned animation buffers must grow only when active capacity increases.
  Inactive packets must bind fallbacks immediately, retain dormant allocations
  for 60 logical frames, then release their packet-owned GPU and CPU payloads.
  Re-activation during the grace period must initialize previous data from the
  new current pose so it does not create a false motion vector.
- Shadow atlas or page instances of one caster must share the caster's current
  joint and morph payload. Instance indices used for repeated shadow projection
  must not advance into previous-pose storage.
- `WebGPUAnimationPayloadPool.getDebugStats()` must expose backend-internal live
  entry, buffer, and byte counts together with cumulative allocation, release,
  upload, skipped-upload, rebuild, and grace-release counters.

### CPU draw preparation and static batching

- `WebGPUFrameServiceOwner` must own one `WebGPUDrawResourceCache` for object
  transform history, scene and depth model bindings, material snapshots, and
  shared material buffers. It must advance resource history once per logical frame
  and destroy them through one lifecycle entrypoint after static bindings release
  their material references.
- The draw resource cache must index draw records and material snapshots. Each
  draw record must own its object transform history and one aligned slot in a
  shared object-uniform arena, without a separate per-draw object buffer.
  The draw's frame marker must also govern object history advancement; object
  state must not maintain a second frame marker or initialization flag.
  Using either pass must retain the draw record. Eviction must dispose the whole
  inactive record; active-frame records must remain live through submission.
- The shared object arena must expose each slot through a dynamic uniform
  offset. Slot stride must use the device `minUniformBufferOffsetAlignment`, and
  each object binding must declare the `ObjectUniforms` byte size so the arena
  remains within `maxUniformBufferBindingSize`.
- A draw record must retain at most one solid and one wireframe scene binding
  reference. Compatible draws may share a scene binding keyed by material
  snapshot, animation payload, morph buffers, and object arena. Replacing a
  slot must release its previous reference and retire the binding through the
  next frame boundary when the last reference is gone. Scene binding slots must
  share the draw record's lifetime rather than maintain independent pipeline-key
  caches or expiry frames.
- Static depth records must use instance-arena buffer identity directly. Depth
  binding reuse must depend on the bound native buffer identities and sizes;
  animation generation alone must not trigger a rebuild when those are unchanged.
  Depth model bindings must bind the same object-arena slot with the same
  dynamic offset used by the draw's scene binding; static depth records must
  bind their control buffer at offset `0`.
- Material revision refresh must be tracked on the material record, shared by
  solid and wireframe snapshots. Built-in materials must refresh at most once
  per logical frame; `ShaderMaterial` must continue to observe same-frame edits.
- Each material resource record must contain immutable snapshot inputs and lazily
  created common, lighting-model, and custom shader uniform buffers. Scene and
  static bindings must hold that record directly. `WebGPUDrawResourceCache` must
  own reference counting and buffer disposal; bindings and snapshots must not
  allocate leases or per-reference retain/release callbacks. Depth model bindings
  must not carry material resource fields. Material revision changes must produce
  a new record, without a second material-data-to-buffer cache.
- Depth model preparation must not resolve scene material snapshots or allocate
  scene material buffers. Borrowed animation and static instance buffers must
  remain owned by their existing resource owners.
- WebGPU material uniform data, texture resources, sampler resources, and
  pipeline keys must be resolved once per material render revision and reused
  by every compatible draw in the frame.
- `WebGPUMaterialPipelineResolver` must translate `Material` and
  `ShaderMaterial` state into immutable WebGPU pipeline policy without creating
  shader modules, pipelines, bindings, or other GPU resources.
- Resolved material pipeline identity must include the material revision,
  shader target mode, wireframe state, and shader-runtime fingerprint. Shared
  scene pipeline resources must consume the resolved policy rather than inspect
  concrete material subclasses.
- Frame shadow state must not participate in built-in scene or deferred shader,
  pipeline, or static preparation cache identity. Draw resource assembly, warmup,
  and deferred pass orchestration must not select a shadow compilation variant.
  Built-in lighting and custom-program fallback must read the current runtime
  shadow uniforms. Ready built-in static preparation must remain synchronous
  across shadow-state changes.
- WebGPU device resource hosts must expose shader-runtime revision, diagnostic
  mode, directive cache identity, and runtime-injection support through one
  read-only view. Pipeline resource owners must not recover those values by
  casting a concrete backend.
- Pipeline lookup must consume the resolved material snapshot and must not
  rebuild material uniform data solely to obtain a cache key.
- `WebGPUDrawResourceAssembler` must own shared geometry, material snapshot,
  animation payload, binding, wireframe, and static-batch preparation. It must
  select pipelines only through an injected `WebGPUDrawPipelineProvider` and
  must not branch on feature runtime identity.
- Scene and planar-reflection composite draws must use distinct pipeline
  providers. Planar composite must not enter scene pipeline resources through a
  special draw-mode branch.
- Every enabled scene stage must prepare draw resources before beginning its
  render pass. Recording must not await resource discovery or resolve the same
  packet again for each incremental dirty rectangle.
- Unique model binding groups must not enter the shared binding-group
  deduplication cache. Shared cache insertion and eviction must remain
  amortized constant time and must not sort every entry after reaching its
  capacity.
- Scene draw bindings must separate object, common material, and
  lighting-model uniform buffers. `ObjectUniforms` data must remain draw-owned
  through one non-overlapping arena slot selected by a dynamic offset; immutable
  `MaterialCommonUniforms` and the active lighting-model uniform buffer must be
  shared by compatible dynamic and static draws.
- WebGPU material shading families are `pbr`, `phong`, `flat`, and `unlit`.
  `GouraudMaterial` must continue to use the `phong` family until Gouraud is
  removed by a separate public API change. An unlit draw must not bind an
  inactive lighting-model uniform buffer.
- Scene color, G-buffer, and reflection draws must use the model bind-group and
  pipeline layouts of their shading family. Their pipeline, shader-module,
  warmup, and binding cache identity must include the resolved shading family.
  WebGPU-private PBR feature and texture flags must remain runtime uniform
  data and must not enter pipeline identity.
- Built-in Early-Z draws must use one family-independent pipeline layout with
  camera bindings in group `0`, object and current-position deformation
  bindings in group `1`, and alpha-mask material bindings in group `2`.
  They must not resolve or bind lighting buffers, clustered lighting, normal
  maps, or unrelated material textures. Opaque draws may share neutral
  alpha-mask bindings required by the common depth layout.
- Built-in depth material snapshots must contain only alpha-mask coverage and
  base-map sampling inputs. Unrelated material revisions must not replace depth
  material resources. Native texture or sampler replacement must invalidate the
  corresponding depth binding even when authored material values are unchanged.
  Depth preparation must reject results when relevant draw policy or alpha
  coverage inputs change while shader compilation or texture resolution awaits.
- Scene pipelines must implement the opaque equal-depth and invariant-position
  rules in `docs/contracts/rendering.md`. Early-Z must remain enabled by default
  and must not require an additional attachment or change the depth format.
- Object and animation buffers must be shared through their resource owners;
  scene and depth binding lifetimes must not destroy each other's live buffers.
  Static depth preparation must reuse the instance arena without first creating
  scene material bindings. Prepared bindings must stay live through submission.
- The shared static object control buffer must be cached only after its initial
  upload succeeds. A failed upload must release the allocation and leave scene
  and depth preparation retryable without reusing uninitialized control data.
- Built-in depth pipeline identity must depend on the actual shader, vertex
  layout, primitive, depth, sample-count, and shader-runtime state, not the
  shading family. Concurrent equivalent requests must share pipeline creation.
  Custom `ShaderMaterial` depth programs must retain the existing scene binding
  ABI and use the corresponding scene pipeline layout.
- Material GPU buffers must use reference-counted ownership. Each successful
  retain must be balanced by one release through the owning draw resource cache.
  Failed binding creation must release its retained reference. Retired bindings
  must retain their reference until disposal; a shared buffer must not be
  destroyed while any dynamic or static binding still references it. Cache
  destruction must release remaining allocations and make later releases harmless.
- Static material bindings referenced by prepared draws in the active frame
  must not be evicted. Ready preparation reuse must refresh their usage.
  Inactive bindings must be preferred for eviction; any temporary active-frame
  overflow must be trimmed at the next frame boundary.
- Shader-runtime invalidation and backend destruction must explicitly destroy
  each unique managed shader-module and pipeline handle before clearing cache
  maps. Alias caches must not cause duplicate destruction.
- Concurrent requests for one Early-Z pipeline key must share one high-level
  in-flight promise. Cache population must not overwrite managed wrappers from
  other waiters without explicitly destroying the losing wrapper.
- Render-pass recording must suppress redundant pipeline, bind-group, vertex
  buffer, and index-buffer commands when native state is unchanged.
- Static built-in opaque and mask packets may be grouped into instanced draws
  when pipeline, geometry layout, material bindings, and pass behavior match.
  Transparent, transmission, wireframe, reflection, `ShaderMaterial`, skinning,
  and morph packets must retain the legacy path.
- Static instance records must contain current and previous transforms, normal
  data, render layers, and shadow flags in a frame-owned arena. Arena growth
  must be geometric and uploads must be consolidated.
- Unchanged static instance records may reuse their existing arena contents.
  Reuse must observe current float32 matrix, normal, layer, and shadow data,
  including direct matrix edits, packet reordering, and previous-pose changes.
  Uploads must cover changed records; arena replacement must upload fresh data.
- Prepared static draws must reuse material policy and geometry preparation
  within an owning frame scope, pipeline provider, pass, and sample count.
  Ready immutable preparation may be retained across frames. Ready cache
  hits must return synchronously while preserving a distinct draw result and
  instance index for each packet. Callers must await only pending preparation.
- Shared preparation must be invalidated by material or geometry revision,
  geometry resource replacement or release,
  provider or shader-runtime invalidation, instance-arena binding replacement,
  scope replacement, and backend destruction. A new frame must invalidate
  pending work and refresh instance indices and frame bindings while retaining
  valid ready preparation. Failed or stale in-flight work
  must not populate a current preparation cache. Unsupported packets must
  retain the legacy preparation path.
- Incremental dirty-rectangle execution must build rect-local instance ranges;
  instances outside the rectangle must not be emitted by a batched draw.
- Dirty-rectangle packet selection must intersect indexed candidates with the
  spatial query even when a caller supplies a copied, reordered, or filtered
  packet array. It must preserve candidate order and must not reintroduce
  packets omitted by the caller. Candidates absent from the prepared scene's
  indexed packet lists must remain eligible conservatively, including transient
  particle packets. Without a spatial index, selection must retain candidates.
- `submitWebGPUDraws()` must report `staticBatching.nonAdjacentKeySplits` and
  `staticBatching.nonContiguousInstanceSplits` without changing draw order or
  the adjacent-key and contiguous-instance merge requirements.
- Split accounting must restart for each submission and dirty rectangle. A
  key's first emitted batch in that scope must not count as a split. Each later
  batch for that key must count once: `nonAdjacentKeySplits` when another
  command intervenes, otherwise `nonContiguousInstanceSplits` when its
  `firstInstance` does not follow the preceding batch's instance range.
  Non-adjacency must take precedence when both conditions apply. Skipped draws
  must not act as intervening commands; non-static emitted draws must.
- Frame resource diagnostics must accumulate both counters from successful
  submissions under `getDebugStats().staticBatching` and reset them in
  `beginFrameResourceLifecycle()`. Counters must describe recorded batch
  fragmentation, not a guaranteed number of safely removable draw calls.

### Geometry packing

- `WebGPUGeometryRegistry` must store scene geometry in semantic vertex streams.
- Geometry registry entries must be keyed by the submission's opaque geometry
  resource key and invalidated by its captured geometry version. The registry
  must not inspect primitive authoring state through that key.
  Position data must use an independent `float32x3` stream. Surface data may
  contain normal, tangent, and UV attributes, while skinning data may contain
  four- or eight-influence joint and weight groups.
- Missing optional vertex attributes must be supplied by one device-lifetime
  default vertex buffer with `arrayStride: 0`. Default normal, tangent, UV,
  joint, and weight values must remain zero. Geometry uploads must not allocate
  per-vertex padding for a missing semantic.
- Normal and joint attributes must retain their existing floating-point shader
  input types. Tangents should use `snorm16x4`, UVs should use `float16x2`, and
  weights should use `unorm16x4` when the source data is safely representable.
- A UV semantic may use `float16x2` only when every finite source component
  round-trips with absolute error no greater than `1 / 4096`. A tangent may use
  `snorm16x4` only when every component is finite and in `[-1, 1]`. A weight
  group may use `unorm16x4` only when every component is finite and in `[0, 1]`.
  The complete semantic must fall back to its corresponding `float32` format
  when any component fails these requirements.
- Scene, early-Z, wireframe, reflection, and shadow pipeline identity must
  include the complete geometry layout signature. The signature must describe
  stream slots, attributes, formats, offsets, strides, and the skinning profile.
- Shadow draws must bind only the position, skinning, and default streams.
  Surface attributes must not be fetched by shadow-depth pipelines.
- Geometry whose largest index is at most `65535` must use `uint16` index
  storage. Other geometry must use `uint32`. Every scene and shadow draw must
  bind the index buffer with the format recorded by its geometry handle.
- Triangle wireframe indices must be created only when a wireframe draw is
  requested. Generation must retain the first occurrence of each canonical
  undirected edge, cache the resulting buffer until geometry invalidation, and
  use the smallest valid index format. Non-triangle geometry must not allocate
  a wireframe buffer.
- Morph position and normal deltas must use separate target-major flat XYZ
  `array<f32>` storage. A semantic buffer must exist only when at least one
  active target provides that semantic. Animation metadata must provide vertex
  count and semantic-presence flags so shaders never infer layout from fallback
  storage length.
- Geometry-version invalidation and registry destruction must release every
  semantic stream, index buffer, lazily created wireframe buffer, and morph
  buffer owned by the invalidated handle.

### Deferred lighting

- Public control contract:
	- `WebGPUBackendOptions.enableDeferredLighting` must default to `true`.
	- When `enableDeferredLighting === false`, `WebGPUBackend` must not run the
	  deferred lighting resolve and must route opaque materials through the
	  legacy MRT forward path when MRT is available.
	- When `enableDeferredLighting !== false`, `WebGPUBackend` must attempt to
	  enable deferred lighting and must warn once if any runtime requirement
	  prevents correct activation.
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
- Clustered-lighting compute contract:
	- Gather culling and overflow resolution must dispatch the cluster grid across
	  `tilesX`, `tilesY`, and `zSlices`. They must not flatten the complete cluster
	  count into one dispatch dimension.
	- Gather and overflow shaders must derive the row-major cluster index from the
	  three-dimensional workgroup id so large valid frame grids do not exceed
	  `maxComputeWorkgroupsPerDimension` in a single dimension.
- G-buffer contract:
	- Built-in PBR draws, including generic materials using PBR shading, must
	  upload WebGPU-private feature and texture flags in a draw-uniform
	  `pbrMasks` field. WebGPU must construct these flags from
	  `resolvePBRMaterialFeatures` semantics. They must not participate in
	  shader-module or render-pipeline cache keys.
	- Built-in WGSL asset composition and directive-profile inputs must be
	  declared by the WebGPU shader manifest. The shared manifest mechanism must
	  not turn material feature flags into source specializations.
	- Built-in PBR shaders must use uniform `if` control flow to sample a material
	  texture only when its texture-presence bit is set. Extension texture samples
	  must additionally require an active parent feature; `select` must not be
	  used to guard a texture operation.
	- Missing texture samples must resolve to the same neutral values as the
	  device-lifetime fallback textures. Fixed bind-group layouts and fallback
	  bindings must remain available even when shader control flow skips the
	  texture operation.
	- Deferred packet preflight must retain one frame-local recording snapshot per
	  packet. The snapshot must contain the resolved pipeline, material uniform
	  data, texture and sampler resources, model binding, geometry resources, and
	  the selected early-Z/G-buffer draw variants.
	- G-buffer, extension, and early-Z recording must consume the retained packet snapshots
	  and must not call `getDrawResources()` again. A packet selected by multiple
	  incremental dirty rectangles must reuse the same snapshot for every
	  rectangle.
	- Deferred scene warmup must compile the base G-buffer variant and, for an
	  `extended` layout, the extension replay variant. It must report extension
	  compilation failures with their original compiler diagnostics. Warmup and
	  frame preflight must share the deferred
	  early-Z eligibility policy; custom deferred programs must not warm unused
	  vertex-only early-Z variants. A `base` layout must not warm extension replay.
	- Forward scene, base G-buffer, and extension replay pipelines must reuse the
	  same scene pipeline layout for each shading family.
	- Before an incremental partial scene draw, WebGPU must shader-clear scene
	  color and every active raster G-buffer attachment inside all dirty
	  rectangles. The clear must use the attachment layout's canonical empty
	  values, must preserve pixels outside the rectangles, and must precede
	  depth clear, early-Z, and geometry recording.
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
	- `gNormalRoughMetal.xy` must store the encoded view-space normal,
	  `gNormalRoughMetal.z` must store roughness, and
	  `gNormalRoughMetal.w` must store metallic.
	- `gSpecular.rgb` must store the resolved specular color. `gSpecular.a` must
	  store the PBR specular factor or the unclamped Phong/Flat shininess,
	  according to the packed shading model.
	- Extension payload textures must be `gMaterialExt0` as `rgba16float` and
	  `gMaterialExt3` as `rgba16uint`. They must support render attachments and
	  retain storage usage for visibility-filtered decal composition. Deferred
	  opaque shading must not store transmission volume or attenuation payloads.
	- Opaque geometry must write extensions through a second raster pass after
	  the base G-buffer has selected the final visible depth. Extension outputs
	  must be color attachments, not fragment storage writes. The extension pass
	  must use read-only depth with `equal` comparison, replay the same packet and
	  primitive order, and preserve the original alpha discard, fragment depth,
	  and sample mask. Equal-depth ties must retain the base pass's last-draw winner.
	  This rule must hold with early-Z enabled or disabled and for incremental
	  dirty rectangles. Base-only frames must not record an extension pass.
	- `gMaterialExt0.xy` must store the encoded view-space clearcoat normal,
	  `gMaterialExt0.z` the iridescence factor, and `gMaterialExt0.w` the
	  iridescence thickness.
	- `gMaterialExt3.xy` must store the encoded view-space anisotropy tangent,
	  `gMaterialExt3.z` must store the resolved anisotropy strength, and
	  `gMaterialExt3.w` must store the 11-bit receiver render-layer mask.
	- WebGPU must expose `LogicalGBufferBridge.normalSpace` as `"view"`.
	  Consumers that combine G-buffer normals with reconstructed world-space
	  positions or directions must transform decoded normals through the packed
	  camera basis first.
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
	- Every WebGPU shader chunk must explicitly set `backend: "webgpu"`.
	- `ShaderTargetMode` must include `"deferred"`.
	- `ShaderStageKind` must include `"fragment-deferred"`.
	- `ShaderMaterialParams.deferredLighting` must opt a shader material into
	  deferred routing.
	- `ShaderMaterialParams.fragmentDeferredBaseEntryPoint` must select the base
	  G-buffer fragment entry point and must default to `fsMainDeferredBase`.
	- `ShaderMaterialParams.fragmentDeferredExtensionsEntryPoint` must select the
	  extension fragment entry point and must default to `fsMainDeferredExtensions`.
	  Both entries must exist in the same WebGPU deferred fragment chunk. The base
	  entry must write the seven standard G-buffer color outputs. The extension
	  entry must write `gMaterialExt0` at location `0` as `vec4<f32>` and
	  `gMaterialExt3` at location `1` as `vec4<u32>`.
	- When MRT rendering falls back from a missing `fragment-mrt` chunk to a
	  `fragment-single` chunk, the resolved WebGPU program must use
	  `fragmentSingleEntryPoint`. The pipeline must preserve the MRT attachment
	  layout while setting every color target after location `0` to a zero write
	  mask.
	- A `ShaderMaterial` must enter the G-buffer path only when
	  `deferredLighting === true` and a WebGPU deferred fragment chunk exists.
	  Deferred packets must also be opaque or masked and write depth.
	- Custom shader authors must preserve the same discard, fragment depth, and
	  sample mask semantics in both entries, preferably through a shared WGSL
	  surface function. Their shared vertex entry must mark its `position` output
	  with `@invariant` and use identical clip-position inputs and arithmetic in
	  both pipelines; sharing a module alone is not an invariance guarantee.
	  The engine must select the supplied entries directly, without rewriting
	  their interfaces or generating compatibility entry points. Direct fragment
	  storage writes to extension G-buffer textures are unsupported.
	- Custom deferred programs must select final depth without a vertex-only
	  early-Z prepass that omits their fragment depth or discard logic. Missing or
	  invalid deferred entries must reject frame preflight before recording draw
	  commands; they must not silently select built-in or forward shaders.
	- Non-opt-in `ShaderMaterial` instances must render after lighting through
	  the legacy MRT forward fallback.
- Camera reconstruction contract:
	- Perspective camera packing must keep `environmentBasisRight.w` as
	  `tanHalfFov` and `environmentBasisUp.w` as `aspect`.
	- Orthographic camera packing must store `halfWidth` in
	  `environmentBasisRight.w` and `halfHeight` in `environmentBasisUp.w`.
	- `environmentBasisBackward.w` must remain the orthographic flag.

### Screen-space global illumination

- `ScreenSpaceGlobalIlluminationPass` must transform decoded view-space
  G-buffer normals into world space before constructing hemisphere rays,
  offsetting ray origins, or evaluating hit-facing terms.
- A Hi-Z depth candidate whose normal does not face the incoming ray must be
  treated as a traversal miss. SSGI must continue marching instead of ending
  the ray with a rejected self-hit or back-facing hit.
- Binary hit refinement must apply the same facing test as the coarse trace so
  refinement cannot move a valid hit onto a rejected surface.

### Direct PBR transmission

- Forward PBR directional, point, spot, and sampled area lighting, including
  clustered lighting, must evaluate direct transmission as a thin-surface GGX
  BTDF. A Lambertian back-light term must not stand in for specular transmission.
- The transmission half vector must use the light direction mirrored across the
  oriented surface normal. Only light arriving on the opposite hemisphere may
  contribute; invalid microfacet-facing configurations must contribute zero.
- The lobe must narrow as roughness decreases. Its GGX alpha must use squared
  perceptual roughness scaled by `clamp(2 * ior - 2, 0, 1)`, with a numerical
  floor of `0.0016` matching the renderer's minimum perceptual roughness `0.04`.
  Distribution normalization must preserve the narrow aligned-light peak.
- Direct transmission must retain base-color tint, dielectric Fresnel loss,
  metallic suppression, volume attenuation, clearcoat attenuation, light
  attenuation, and shadow visibility. Fresnel must use the transmission half
  vector rather than only the macroscopic surface normal.
- This correction must not change alpha coverage, transparent-pass routing,
  environment transmission, or screen-space refraction composition. Direct
  transmission remains an isotropic thin-surface approximation; exact curved
  volume exits and anisotropic transmission are outside this contract.

### Screen-space refractions

- `ScreenSpaceRefractionsPass` must capture the nearest transmissive surface
  separately from the opaque scene color and depth used by the trace.
- Transmission capture must keep geometric coverage separate from optical
  Fresnel reflectance. Fresnel reflectance must not be multiplied into the
  geometric coverage or applied twice to captured surface lighting.
- Double-sided transmission capture must use the same camera-facing normal
  convention as double-sided PBR lighting for IOR side tests and refraction.
- The transmitted background weight must be
  `coverage * transmission * (1 - metallic) * (1 - Fresnel)`. The composed
  surface lighting, transmitted background, and uncovered scene contribution
  must not allocate more than the captured incident radiance before emissive
  and analytical-light contributions.
- A non-zero material thickness must use a parallel-interface volume
  approximation: trace through the entry interface, advance by the refracted
  path length, and refract through the exit interface before screen-space
  intersection. Zero thickness must behave as a thin interface without an
  artificial material-space travel distance.
- Beer-Lambert attenuation must use the refracted path length through the
  volume, not the normal-axis thickness alone.
- Total internal reflection must contribute no transmitted background. Its
  remaining transmission-lobe energy must be redirected to a reflected
  screen-space ray when available.
- A missing screen-space hit may fall back to the undisplaced captured
  background, but must not change the optical energy partition. Screen-edge
  and distance confidence may blend between the hit and fallback samples; it
  must not delete transmitted energy.
- Transmission roughness must widen the sampled background footprint. The
  implementation may use a generated mip chain, cone filtering, or stochastic
  microfacet sampling, but it must not request unavailable mip levels from a
  single-level texture.
- The transmission denoiser must use the captured transmission depth and normal
  as geometry guides. Opaque surfaces behind the refractor must not define the
  refractor's denoising boundary.
- When MSAA is active, transmission capture must fall back to forward
  transmission rendering because WebGPU does not resolve depth attachments.
- The capture remains a nearest-layer screen-space approximation. Nested
  transmissive layers, unavailable offscreen radiance, and exact curved-volume
  exit geometry are outside the current contract and must use documented
  fallbacks instead of pretending that an opaque-depth hit is a second material
  interface.

### Sample-count configuration

- `WebGPUBackendOptions.sampleCount?: number`
	- Input contract: accepts a finite number. The value is floored and clamped
	  to at least `1`.
	- Behavior contract: omitted values request the default `1x` sample count.
	  Values greater than `1` request main-scene multisampling. The active count
	  may be lower than requested when device capabilities do not support the
	  requested count. Custom render targets use their own descriptor count.
	- Error contract: non-finite values must throw a configuration error.
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

### Render-target view jobs

- WebGPU must execute render-target jobs through the active frame command stream
  and must not submit a dedicated probe command buffer.
- `WebGPURenderTargetViewExecutor` must prepare an isolated frame-resource scope
  for the job camera and must reuse backend-owned scene, environment, lighting,
  particle, and shadow services.
- The public target must expose only final `rgba16float` scene color and optional
  `depth32float`. Legacy MRT or deferred intermediates must be private scratch
  resources.
- Readback copies must follow their writer job in command order. Ticket results
  must not publish until the frame transaction commits.

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

The frame packers must accept separate camera, light, shadow, and environment
inputs and target these `sceneFrameBindGroupLayout` resources:

| Packer | Binding | Layout | Size |
| --- | --- | --- | --- |
| `packFrameCameraUniformData` | `0` | `FrameCameraUniforms` | 272 bytes |
| `packFrameLightUniformData` | `10` | `FrameLightUniforms` | 1,712 bytes |
| `packFrameShadowUniformData` | `11` | `FrameShadowUniforms` | 6,544 bytes |
| `packFrameEnvironmentUniformData` | `12` | `FrameEnvironmentUniforms` | 4,224 bytes |

`FrameCameraUniforms` must contain unjittered `viewProjection`,
`prevViewProjection`, `inverseView`, and `inverseProjection` matrices at byte
offsets `0`, `64`, `128`, and `192`, followed by `jitterNdc` and `prevJitterNdc`
as `vec2<f32>` at offsets `256` and `264`. Camera position and axes must be
read from `inverseView`. Shader consumers must import the shared declaration.

Screen UVs must use a top-left origin. Reconstruction must subtract current
NDC jitter; projection into the current rasterized view must add it. Hardware
depth must be mapped from `[0, 1]` to `[-1, 1]` before inverse projection.
Linear-depth reconstruction must intersect the inverse-projected view ray
with `viewZ = -depth`, preserving asymmetric orthographic bounds. Camera
helpers must distinguish hardware depth from linear view depth. Motion vectors
must retain the difference between current and previous jittered NDC positions.
Oblique clipping must not reverse camera rays or make linear-depth
reconstruction non-finite at the clip-plane horizon. Ray construction must
preserve homogeneous coordinates until the required view-depth intersection.
Shared helpers must accept only their required matrices and jitter, rather than
copying the entire camera uniform value. Planar-reflection composite sampling
must use unjittered capture UVs while its main-view raster position uses jitter.
Clustered lighting must retain its perspective-only fallback policy.

`FrameLightUniforms` must prepend a 48-byte header containing `ambientColor`,
`lightCounts: vec4<u32>`, `lightingEnabled`, and `clusteredLightingEnabled`.
`FrameShadowUniforms` must prepend a 16-byte `shadowsEnabled` header.
`FrameEnvironmentUniforms` must prepend a 16-byte header containing
`shAmbientEnabled`, `hasEnvSpecular`, and `hasBRDFLUT` as `u32` values `0` or `1`
at offsets `0`, `4`, and `8`, followed by non-negative `envSpecularMaxMipLevel`
as `f32` at offset `12`. `shAmbientCoeffs` must begin at offset `16`.
Environment uniforms must not retain padding for removed fields. Required
alignment padding must follow the final active member of a packed header.
SH enablement must combine configured enablement and available SH data.
Background parameters must contain only the 16-byte `tintExposureStrength`
vector, without padding. Background and IBL sources must use native formats that
return linear RGB, preserving the linear probe-atlas path without shader decoding flags.

Built-in standalone parameter uniforms must use their active WGSL payload size
rather than an unconditional 16-byte size rounding. Buffer offsets and members
must still meet their own alignment requirements. These parameter layouts must
use the following sizes:

| Uniform | Size | Layout requirement |
| --- | --- | --- |
| `DenoiseParams` | 48 bytes | Twelve scalar words, without a trailing vector |
| SSGI `TraceParams` | 64 bytes | Sixteen scalar words |
| SSRF `TraceParams` | 48 bytes | Twelve scalar words |
| SSR `ComposeParams` | 8 bytes | One inverse-size `vec2<f32>` |
| `GammaParams` | 8 bytes | HDR enablement and headroom |
| `ToneMappingParams` | 12 bytes | Exposure, headroom, and HDR enablement |
| `PresentParams` | 8 bytes | HDR enablement and transparent-output state |
| `OcclusionParams` | 4 bytes | Depth bias only |
| Scene and shadow `AnimationParams` | 24 bytes | Six scalar words |
| `PrefilterParams` | 32 bytes | Source mip count at offset `24`; zero tail padding at `28` |

CPU writers, allocations, and every matching shader declaration must agree on
these layouts. Alignment gaps preceding arrays or vectors must remain in place
unless the active fields are reordered together; moving padding alone must not
change the alignment of later members.

Frame packers must return `ArrayBuffer` and accept optional reusable writers.
Each view scope's `WebGPUFrameBindingCache` must own its writers and validate
camera matrices and the view and projection inverses for finite `f32`
representation. It must consume the inverse matrices owned by `Camera` instead
of retaining a separate inverse cache. Camera matrix ownership follows the
rendering contract.
View-projection validation must check finite `f32` values without computing an
additional inverse solely for validation.
Non-finite camera matrices and singular view or projection matrices must fail
before temporal advancement or GPU writes. `TemporalFrameState` must remain
the history owner; aborted frames and secondary captures must not advance
main-view history.
Post-process enablement must remain owned by the post-process runtime.

PBR material packing must consume the named semantics returned by
`resolvePBRMaterialFeatures`. `pbrMasks` and the matching WGSL constants are
private WebGPU shader ABI encodings; they must be constructed by WebGPU from
those semantics and must not depend on material-owned bit assignments.
Generic PBR materials must use the same semantic resolution as `PBRMaterial`.
Extended G-buffer selection must use resolved lobe activation; an inactive
extension texture alone must not require the extended layout. Non-default
reflectance must continue to require the extended layout.

Scene material packing must use these `group(1)` uniform bindings:

| Binding | Layout | Ownership |
| --- | --- | --- |
| `0` | `ObjectUniforms` | One dynamic-offset arena slot per draw, or one static-batch control buffer |
| `41` | `MaterialCommonUniforms` | Shared immutable material snapshot |
| `42` | `PBRMaterialUniforms` | Shared PBR snapshot only |
| `43` | `PhongMaterialUniforms` | Shared Phong or Gouraud snapshot only |
| `44` | `FlatMaterialUniforms` | Shared Flat snapshot only |

`ObjectUniforms` must contain current and previous model transforms, the normal
matrix, and one instance metadata vector. Binding `0` must use a dynamic
uniform offset; dynamic draws select one arena slot, while static-batch bindings
use offset `0`. `MaterialCommonUniforms` must contain
base color, emissive, alpha/render state, planar reflectivity, and material
texture transforms. Lighting-model buffers must contain only properties
consumed by that lighting model. The former combined `ModelUniforms` layout is
not part of the current WebGPU shader ABI.

## Usage

### Geometry packing

Geometry packing is automatic and has no public configuration switch. Asset
authors should keep UV values near their sampled range when half precision is
desired. Values outside the accepted precision envelope remain supported and
use a floating-point fallback for that semantic.

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
	fragmentDeferredBaseEntryPoint: "fsDeferredBase",
	fragmentDeferredExtensionsEntryPoint: "fsDeferredExtensions",
	chunks: [
		{ backend: "webgpu", language: "wgsl", stage: "vertex", code: vertexWGSL },
		{
			backend: "webgpu",
			language: "wgsl",
			stage: "fragment",
			mode: "deferred",
			code: deferredFragmentWGSL,
		},
	],
});
```

The shared vertex output must declare invariant position for depth-equal replay:

```wgsl
struct VertexOutput {
	@invariant @builtin(position) position: vec4<f32>,
	@location(0) normal: vec3<f32>,
}
```

`tests/browser/fixtures/webgpu_gbuffer_custom_probe.wgsl` contains a complete
two-entry example with shared surface evaluation, discard, fragment depth, and
sample mask.

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
	structOf,
	vec,
} from "../src/backends/webgpu/StructuredBufferLayout";
import {
	arrayVec4,
	createStructuredBufferPacker,
	vec4,
} from "../src/backends/webgpu";

const layout = new StructuredBufferLayout(
	structOf([
		{ name: "baseColorFactor", type: vec(4, "f32") },
		{ name: "textureTransformA", type: arrayOf(vec(4, "f32"), 2) },
	]),
	"uniform"
);

const packer = createStructuredBufferPacker({
	label: "ExampleMaterialCommonUniforms",
	layout,
	output: "float32Array",
	fields: [
		vec4("baseColorFactor", (input) => input.material.baseColorFactor),
		arrayVec4("textureTransformA", 2, (input, index) =>
			input.material.textureSlots[index]?.transformA
		),
	],
});

const packed = packer.pack({
	material: {
		baseColorFactor: [1, 1, 1, 1],
		textureSlots: [{ transformA: [0, 0, 1, 1] }],
	},
});
```

## Diagnostics

### Geometry packing

- A packed semantic that exceeds its finite range or error threshold must use
  the float32 fallback without rejecting the geometry.
- If one material produces pipelines for multiple geometry layouts, verify the
  pipeline cache key includes the complete geometry layout signature.
- If a wireframe allocation exists before the first wireframe draw or warmup,
  verify the registry does not eagerly request wireframe indices.

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
  legacy forward rendering. A valid custom deferred material must not disable
  this fallback for failures in shared deferred resources or built-in draws.
- Custom deferred program resolution, shader compilation, and render-pipeline
  creation failures must preserve their error origin and reject frame preflight,
  including in `warn` mode. They must not select built-in or forward shaders.
  Other deferred preflight failures must retain the runtime fallback policy.

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

## Verification

```bash
bun tests/run_all.mjs tests/static/webgpu
bun tests/static/geometry/test_geometry_registry_versioning.mjs
bun tests/static/webgpu/test_webgpu_post_graph.mjs
bun run test:browser:webgpu
bunx tsc --noEmit
```

## Related Documents

- [WebGPU architecture](../architecture/webgpu.md)
- [Render Graph architecture](../architecture/render-graph.md)
- [WebGPU bindings reference](../reference/webgpu-bindings.md)
