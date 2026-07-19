# WebGL Internal Frame Graph Contract

## Scope

This document defines the backend-internal WebGL frame graph used behind
`WebGLBackend`. It does not define renderer-level stages, public
post-process registration APIs, or public Render Graph APIs. Shared logical
analysis is defined by `docs/rendergraph/internal-render-graph-architecture.md`.

## Background

`WebGLBackend` exposes renderer-owned frame lifecycle methods. Internally,
`WebGLFrameGraphRuntime` expands renderer-level `FramePass` entries into WebGL
nodes so framebuffer, texture, OIT, post-process, and presentation work can be
validated without keeping pass orchestration inside `WebGLFrameExecutor`.

## API/Contract
- `WebGLFrameGraphPlanner` must create WebGL internal nodes for one
  renderer-level `FramePass`.
- `WebGLFrameGraphRuntime` must execute synthetic `scene-clear` and optional
  `environment` nodes during `beginFrame(context)`.
- `WebGLFrameGraphRuntime` must execute a synthetic `present` node during
  `endFrame(context)`.
- `WebGLFrameExecutor` must remain a thin aggregate facade for frame begin,
  finish, abort, resize, and service lifetime coordination.
- `WebGLFrameServiceOwner` must construct device-scoped WebGL frame services
  and destroy them in dependency order.
- Each WebGL frame runtime must own and destroy the native handles it creates.
  Frame-sized attachments must be owned exclusively by
  `WebGLFrameTargetManager`; post-process histories must be owned exclusively
  by `BackendPostProcessRuntime` resource pools.
- `WebGLFrameNodeExecutorRegistry` must assign every WebGL graph node kind to
  exactly one executor and must reject missing or duplicate registrations.
- `WebGLFrameExecutor` must not own renderer-level pass orchestration.
- WebGL graph and post-process runtimes must depend on narrow internal
  contracts and must not require the concrete `WebGLFrameExecutor` type.
- `WebGLFrameGraphCompiler` must preserve planner node order.
- `WebGLFrameGraphCompiler` must remain a compatibility facade over the shared
  `RenderGraphStateTracker`. WebGL planning, validation rules, and execution
  must remain backend-private.
- `WebGLFrameGraphCompiler` must emit diagnostics for missing resources,
  reads before creation, duplicate creates, unsupported usages, and WebGL
  texture feedback loops.
- WebGL OIT graph nodes must use separate `oit-accum` and `oit-reveal` nodes
  because WebGL cannot assign different blend states per attachment.
- WebGL frame graph debug state may be exposed through backend-internal hooks.
- Legacy barrier records must be projections of shared logical transitions;
  they must not represent native WebGL synchronization commands.
- Shared shadow diagnostics must not enter the legacy `graphDiagnostics`
  array or trigger `Logger`.
- `WebGLFrameGraphRuntime.endFrame` must seal graph analysis after successful
  presentation. `WebGLBackend` must commit analysis only after post-process
  history and backend frame cleanup succeed. Any failure must abort analysis.
- Custom render passes and particle simulation that bypass graph nodes must
  mark analysis coverage as `"opaque"` without changing their execution path.
- The WebGL internal graph must not add public renderer graph registration APIs.

## Usage

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

## Errors & Diagnostics

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

## Compatibility / Breaking Changes

The public `WebGLBackend` constructor and `Renderer` integration remain
unchanged. The WebGL graph is an internal implementation detail. Tests and
diagnostic tools may use internal debug state, but application code must not
depend on it as a stable public API. The shared analyzer does not change
framebuffer ownership, WebGL state changes, planner order, presentation, or the
nested post-process execution model.
