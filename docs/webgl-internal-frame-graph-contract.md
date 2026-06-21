# WebGL Internal Frame Graph Contract
## Scope
This document defines the backend-internal WebGL frame graph used behind
`WebGLBackendSession`. It does not define renderer-level stages, public
post-process registration APIs, or a cross-backend render graph runtime.

## Background
`WebGLBackendSession` exposes renderer-owned frame lifecycle methods. Internally,
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
- `WebGLFrameExecutor` must own low-level WebGL services, resource handles,
  frame target lifecycle, post-process bridge callbacks, and draw helpers.
- `WebGLFrameExecutor` must not own renderer-level pass orchestration.
- `WebGLFrameGraphCompiler` must preserve planner node order.
- `WebGLFrameGraphCompiler` must emit diagnostics for missing resources,
  reads before creation, duplicate creates, unsupported usages, and WebGL
  texture feedback loops.
- WebGL OIT graph nodes must use separate `oit-accum` and `oit-reveal` nodes
  because WebGL cannot assign different blend states per attachment.
- WebGL frame graph debug state may be exposed through backend-internal hooks.
- The WebGL internal graph must not add public renderer graph registration APIs.

## Usage
```ts
const backend = new WebGLBackend();
const session = backend.createSession({
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

## Compatibility / Breaking Changes
The public `WebGLBackend` constructor and `Renderer` integration remain
unchanged. The WebGL graph is an internal implementation detail. Tests and
diagnostic tools may use internal debug state, but application code must not
depend on it as a stable public API.
