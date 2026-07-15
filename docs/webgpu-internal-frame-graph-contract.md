# WebGPU Internal Frame Graph Contract
## Scope
This document defines the internal WebGPU frame graph used behind
`WebGPUFrameExecutor`. It does not define renderer-level frame stages or public
post-process registration APIs.

## Background
`WebGPUBackend` exposes the renderer-level backend lifecycle. Internally,
`WebGPUFrameGraphRuntime` compiles WebGPU-specific nodes for renderer stages and
validates resource usage before recording commands. WebGPU implementation
details such as deferred lighting, OIT, MSAA frame targets, and presentation
must remain backend-internal.

## API/Contract
- `WebGPUFrameGraphPlanner` must create WebGPU internal nodes for one
  renderer-level `FramePass` through registered stage planners.
- Unsupported renderer-level backend pass ids must produce an empty WebGPU
  stage plan; `WebGPUFrameGraphRuntime` must warn once and skip execution for
  that pass.
- `WebGPUFrameGraphNode.reads` must list resource ids and usages sampled or
  loaded by the node.
- `WebGPUFrameGraphNode.writes` must list resource ids and usages written by
  the node.
- `WebGPUFrameGraphNode.creates` and `WebGPUFrameGraphNode.destroys` may declare
  explicit resource lifetime changes when a future node owns transient targets.
- `WebGPUFrameGraphCompiler` must preserve node order supplied by the planner.
- `WebGPUFrameGraphCompiler` must emit a diagnostic when a non-optional read or
  destroy references an inactive resource.
- `WebGPUFrameGraphCompiler` must emit barrier records for read/write or usage
  transitions between nodes.
- `WebGPUBackendOptions.frameGraphValidation` may be `"throw"` or `"warn"`.
- `WebGPUBackendOptions.frameGraphValidation` must default to `"throw"`.
- `"throw"` mode must fail frame execution on error diagnostics.
- `"warn"` mode must emit diagnostics through `Logger.warn` and continue.
- `WebGPUFrameTargetManager` must own WebGPU offscreen frame target allocation,
  pooled texture ownership, and target debug state. It must request an MSAA
  fallback from the backend-internal MSAA controller when allocation fails.
- The backend-internal MSAA controller must own sample-count configuration,
  device capability probing, resolved runtime state, and persistent `1x`
  fallback state. It must not own frame textures.
- `WebGPUFrameGraphRuntime` must orchestrate frame lifecycle and node execution;
  it must not own texture pool allocation logic.
- `WebGPUFrameGraphRuntime` must execute graph nodes through a node executor
  registry keyed by `WebGPUFrameGraphNode.kind`.
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
- `webgpu-hiz-build-failed` must leave opaque rendering active, make occlusion
  candidates visible, and prevent Hi-Z-dependent post-process passes from
  running for the affected frame.

## Compatibility / Breaking Changes
`getFrameGraphDebugState()` may expose structured internal graph diagnostics,
barriers, resources, and target-manager state. Tests and diagnostic tooling must
not depend on private runtime fields when equivalent graph debug data exists.
The planner and runtime use internal registries instead of switch statements;
this does not add public WebGPU frame graph registration APIs.
