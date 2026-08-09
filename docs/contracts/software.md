# Software Backend Contract

This document defines the Software backend's CPU frame ownership, execution
services, attachment layout, and transaction boundaries. Cross-backend frame
ordering remains defined by the [renderer contract](renderer.md).

## Lifecycle and Frame Ownership

`SoftwareBackend` must remain a one-shot renderer runtime. Its public facade
must own attach, initialize, restore, resize, frame entrypoints, diagnostics,
and destruction while delegating active-frame work to one backend-owned frame
session.

Only one Software frame may be active. Resize requests received during a frame
must be deferred until that frame commits or aborts. Restore must rebuild all
Software execution services and reset temporal state without making the
backend reusable by another renderer.

## CPU Attachments

The Software color attachment must use one `Uint8ClampedArray` with four RGBA
entries per pixel. Depth must use one `Float32Array` entry per pixel. Normal
and motion attachments, when present, must use three and four `Float32Array`
entries per pixel respectively.

Frame attachment dimensions and storage lengths must be validated before any
frame work mutates the attachments. Invalid input must fail with the
`Software frame attachments invalid:` diagnostic prefix.

## Frame Session

The frame session must derive one immutable Software frame view from
`FrameContext`. Camera, environment, feature, scene-work, shadow-plan,
incremental-region, temporal, and attachment state must be resolved once and
shared by Software passes.

Dirty regions must use clamped half-open pixel bounds. Full-frame work must be
represented by one region covering the complete attachment extent. Every
Software raster, particle, reflection-composite, and built-in post-process
implementation must consume the same normalized regions.

Frame presentation must succeed before post-process histories, temporal
camera state, previous world matrices, and completed incremental coverage are
committed. Abort must discard all pending transaction state and must not
advance those histories.

## Execution Services

Software render passes must receive explicit backend-owned services. Shadow,
reflection, particle, raster/material, and post-process runtime objects must
not be published through the shared transient store. The transient store may
carry pipeline-owned animation and particle payloads only at the adapter
boundary where the frame session imports them into Software-owned state.

The Software shader layer must not import Software backend implementation
modules. CPU lighting evaluation used by Software shader strategies belongs to
`src/shaders/software/`.

Software must execute the logical `PostProcessPlan` directly and must not
construct a GPU Render Graph. Post-process, camera jitter, and previous
transform state must participate in the same Software frame transaction.

## Pass Ownership

The Software pass executor owns renderer-stage dispatch and unsupported-pass
diagnostics. `SoftwareBackend` must not directly select scene packets, invoke
concrete raster passes, or implement pixel loops.

The shadow runtime must separately own CPU targets, rasterization, and
sampling. The reflection runtime must separately own plane selection, target
storage, mirrored-view rendering, and main-surface compositing. Secondary
reflection views must not mutate the application camera or commit main-view
temporal history.

## Compatibility

`SoftwareBackend`, `SoftwareBackendOptions`, backend capabilities, pass order,
diagnostic keys, and output pixels remain compatible. Software-only internal
pass interfaces and transient keys are not compatibility surfaces.

## Verification

```bash
bun tests/run_all.mjs tests/static/software
bun run test:lighting
bun tests/static/foundation/test_layer_boundaries.mjs
bunx tsc --noEmit
```

## Related Documents

- [Engine architecture](../architecture/engine.md)
- [Rendering architecture](../architecture/rendering.md)
- [Renderer contract](renderer.md)
- [Shadow contract](shadows.md)
- [Post-process contract](postprocess.md)
