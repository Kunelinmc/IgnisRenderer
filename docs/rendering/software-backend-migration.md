# Software Backend Lifecycle Migration

## Scope

This document defines the migration contract for the Software backend lifecycle,
frame transaction behavior, planar reflection view isolation, and the removal of
the package-root `Rasterizer` export.

## Background

The Software backend now treats a frame as a transaction. Temporal history,
post-process history, completed coverage, and presentation become visible only
after a successful `endFrame()`. This prevents an aborted or failed presentation
from contaminating the next frame.

Planar reflections render through a Software-owned mirrored projection view. The
capture must not mutate the application `Camera` instance used by the main view.

## API/Contract

- A `SoftwareBackend` instance must transition from `detached` to `attached`,
  then `initializing`, and finally `ready` before `beginFrame()` may be called.
- `beginFrame()` may only be called in `ready`. `executePass()`, `skipPass()`,
  and `endFrame()` must operate on the active frame. `executePass()` must reject
  a foreign `FrameContext`.
- `abortFrame()` must be idempotent. It must roll back temporal jitter and retain
  the last successfully committed temporal and post-process history.
- `resize()` during an active frame must defer the latest requested size until
  `endFrame()` succeeds or `abortFrame()` completes.
- `restore()` may only be called in `ready`. A successful restore must clear
  temporal state and emit `device-restored`. A failed restore must keep the
  previous usable runtime.
- `destroy()` must be synchronous, idempotent, terminal, and must prevent a
  later `attach()` or `initialize()` call.
- `Rasterizer` is no longer exported from the package root. Applications must
  use `SoftwareBackend`; direct deep imports are unsupported internal APIs.
- Software opaque rasterization must apply prepared decals after base material
  evaluation and before lighting. Candidate filtering may occur per draw packet,
  but the projector box, receiver mask, coverage, and channel blend checks must
  occur at fragment rate.
- Software decal shading must avoid per-fragment allocation and must write the
  resolved decal normal to the normal buffer used by post-processing.
- Software decal material colors must be normalized to linear space before
  blending. Legacy encoded Phong surface storage must be adapted at the
  material-surface boundary.

## Usage

```ts
const backend = new SoftwareBackend();
renderer.initialize();

// The renderer owns begin/execute/end or abort coordination.
await renderer.renderFrame(performance.now());
```

Applications that previously imported `Rasterizer` from the package root must
remove that import and render through `Renderer` configured with
`SoftwareBackend`.

## Errors & Diagnostics

- Calling `initialize()` before `attach()`, twice, during a frame, or after
  `destroy()` throws a lifecycle error.
- Calling `beginFrame()` before initialization or while another frame is active
  throws a lifecycle error.
- A presentation or pre-commit failure leaves temporal history and completed
  coverage unchanged; the renderer must invoke `abortFrame()`.
- `device-restored` is emitted only after replacement runtime resources are
  ready for use.

## Compatibility / Breaking Changes

- This is a breaking change for consumers of the package-root `Rasterizer`
  export; no public replacement is provided.
- Illegal Software lifecycle calls that previously appeared to work now throw.
- `SoftwareBackendOptions`, scanline raster semantics, and post-process registry
  registration remain compatible.
