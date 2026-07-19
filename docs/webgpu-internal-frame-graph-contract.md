# WebGPU Internal Frame Graph Contract

## Scope

This document defines the internal WebGPU frame graph used behind
`WebGPUFrameOrchestrator`. It does not define renderer-level frame stages or public
post-process registration APIs. Shared logical analysis is defined by
`docs/rendergraph/internal-render-graph-architecture.md`.

## Background

`WebGPUBackend` exposes the renderer-level backend lifecycle. Internally,
`WebGPUFrameOrchestrator` compiles WebGPU-specific nodes for renderer stages and
validates resource usage before recording commands. `WebGPUFrameFeatureAnalyzer`
derives desired frame work and `WebGPUFrameConfigurationResolver` applies
capability and fallback policy before target allocation. WebGPU implementation
details such as deferred lighting, OIT, MSAA frame targets, and presentation
must remain backend-internal.

## API/Contract
- `WebGPUFrameGraphPlanner` must create WebGPU internal nodes for one
  renderer-level `FramePass` through registered stage planners.
- Unsupported renderer-level backend pass ids must produce an empty WebGPU
  stage plan; `WebGPUFrameOrchestrator` must warn once and skip execution for
  that pass.
- `WebGPUFrameGraphNode.reads` must list resource ids and usages sampled or
  loaded by the node.
- `WebGPUFrameGraphNode.writes` must list resource ids and usages written by
  the node.
- `WebGPUFrameGraphNode.creates` and `WebGPUFrameGraphNode.destroys` may declare
  explicit resource lifetime changes when a future node owns transient targets.
- `WebGPUFrameGraphCompiler` must preserve node order supplied by the planner.
- `WebGPUFrameGraphCompiler` must remain a compatibility facade over the
  shared `RenderGraphStateTracker`. Backend-specific node types, planning, and
  execution must remain WebGPU-private.
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
- `WebGPUFrameFeatureAnalyzer` must scan scene, particle, reflection,
  visibility, and post-process work exactly once per frame without applying
  device capability fallbacks.
- `WebGPUFrameConfigurationResolver` must consume analyzed feature work and
  resolve only capability gating, effective configuration, and fallback policy.
- Planner, compiler, orchestrator, and debug state must use the shared typed
  graph resource catalog. The catalog must collect initial active resources
  from concrete frame targets.
- `WebGPUFrameTargetManager` must own WebGPU offscreen frame target allocation,
  pooled texture ownership, and target debug state. It must return allocation
  retry results and must not query or mutate orchestrator state.
- The backend-internal MSAA controller must own sample-count configuration,
  device capability probing, resolved runtime state, and persistent `1x`
  fallback state. It must not own frame textures.
- `WebGPUFrameOrchestrator` must own a single active frame scope and orchestrate
  target retry, frame lifecycle, and node execution; it must not own texture
  pool allocation logic.
- `WebGPUFrameServiceOwner` must be the backend-private shared-service
  composition root. It owns device-lifetime scene, texture, deferred, shadow,
  and particle-render resources; consumers must receive only the corresponding
  narrow resource-provider capability.
- `WebGPUFrameResourceScope` must own frame bindings and clustered-lighting
  state. The orchestrator owns the main scope, each planar target owns one
  persistent capture scope, and probe capture must destroy its temporary scope
  in a `finally` block. Prepared-frame data must not expose a string scope key.
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
  a lifecycle state of `"recording"`, `"committing"`, or `"skipped"`.
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
  recording. `WebGPUBackend` must commit analysis only after submission,
  presentation, deferred lifecycle work, custom target publication, and
  post-process history commit succeed. Any failure must abort analysis.
- Custom render-target and particle-simulation paths that bypass graph nodes
  must mark shared analysis coverage as `"opaque"` without changing execution.
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
- The internal WebGPU graph must not add global renderer-level stages for
  Software or WebGL.
- The frame graph may allocate a shared full-chain `frame:hiz` target when
  occlusion culling or a built-in Hi-Z consumer is active. A `hiz-build` node
  must run after opaque depth is available and before `occlusion-test`.
- `WebGPUHiZBuilder` owns Hi-Z shader, pipeline, mip-view, and binding caches.
  `WebGPUFrameTargetManager` owns the `frame:hiz` texture lifetime.

## Usage

```ts
const backend = new WebGPUBackend({
	frameGraphValidation: "throw",
});
```

```bash
bun tests/static/webgpu/test_webgpu_frame_graph_compiler.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_planner.mjs
bun tests/static/webgpu/test_webgpu_frame_node_executor_registry.mjs
bun tests/static/webgpu/test_webgpu_frame_executor_resilience.mjs
```

## Errors & Diagnostics

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

## Compatibility / Breaking Changes

`getFrameGraphDebugState()` may expose structured internal graph diagnostics,
barriers, resources, and target-manager state. Tests and diagnostic tooling must
not depend on private runtime fields when equivalent graph debug data exists.
The planner and runtime use internal registries instead of switch statements;
this does not add public WebGPU frame graph registration APIs. Rejecting
duplicate frame begins, missing active sessions, and mismatched frame-context
identity strengthens internal lifecycle validation without changing the public
renderer API. The shared analyzer does not transfer native resource ownership,
flatten the nested post-process graph, or change planner and executor order.
