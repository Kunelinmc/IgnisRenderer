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

The Software backend must own one normalized linear HDR scene-color target as
a `Float32Array` with four RGBA entries per pixel. This RGBA32F target is the
authoritative color resource for rasterization, reflection, particles, and
post-processing. It must be allocated only when the surface size changes or
the backend is restored, and released on destroy.

`FrameAttachments.pixels` must remain a `Uint8ClampedArray` with four RGBA
entries per pixel for presentation compatibility and diagnostics. Software
passes must not use it as scene color. Custom Software post-process
implementations must access authoritative color through
`context.resources.color`. Depth must use one `Float32Array` entry per pixel.
Normal and motion attachments, when present, must use three and four
`Float32Array` entries per pixel respectively.

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

The frame session must track the color domain actually produced by completed
passes. A skipped or failed pass must not advance the domain. Presentation must
complete missing tone mapping, gamut conversion, and transfer encoding from
that actual domain before frame histories and temporal state are committed.

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

Software raster and material shader output must be normalized linear radiance.
RGB radiance must not be multiplied by `255` or clamped to `1.0`. Physical
material inputs such as albedo, roughness, metalness, opacity, and alpha-test
coverage must retain their declared ranges. Environment, emissive, reflection,
transparent, and additive-particle results may exceed `1.0`.

Built-in SSAO, TAA, FXAA, and color-filter implementations must operate on
float scene color. TAA history must preserve radiance above `1.0`. HDR color
filter output may be limited by the requested `hdrHeadroom`, but must not be
clamped to `1.0`.

## Display Output

The Software authoritative scene target must use premultiplied RGBA when the
surface requests transparent presentation. Scene and dirty-region clears must
use alpha zero in that mode. Built-in opaque coverage, transparent blending,
particles, environment output, post-processing, and temporal histories must
follow the shared transparent-presentation contract.

Canvas `ImageData` contains straight RGBA. Software presentation must safely
unpremultiply the authoritative color after display conversion before writing
SDR or HDR `ImageData`; it must write zero RGB when alpha is zero.

`SoftwareBackend.profile.capabilities.displayHDR` must be `true`, indicating
that the backend can attempt Display HDR. The active state must still be
resolved from runtime capability probes.

Software Display HDR must use a Display-P3 Canvas 2D context with
`colorType: "float16"` and `rgba-float16` `ImageData`. Initialization must
probe a detached canvas for Display-P3 preservation, Float16 put/read support,
and preservation of RGB values above `1.0`. `SoftwareDisplayOutputManager` must
not request or own the visible canvas context. `SoftwareBackend` must select,
request, and own that context based on the probe result, then
provide the context to the display-output manager for verification and to
`SoftwareSurfaceRuntime` for presentation. A successful probe must select the
highest supported Display-P3 Float16 configuration so display-output changes
can be applied at frame boundaries.

HDR activation additionally requires `(dynamic-range: high)`. Media-query
changes must re-resolve display output, emit `display-output-change` when the
observable state changes, and invalidate the complete frame. Probe or
configuration failure must not throw on unsupported browsers. It must fall
back to SDR, using `canvas-hdr-output-unsupported` for a failed Canvas 2D HDR
capability probe and preserving the existing display and configuration
failure reasons where applicable.

HDR presentation must submit Float16 Display-P3 pixels and may preserve
encoded values above `1.0`. In HDR mode, `FrameAttachments.pixels` must contain
only a clipped RGBA8 diagnostic preview. SDR presentation must submit RGBA8
sRGB pixels. Chromium is the required browser validation target; other browsers
must use the same probes and fall back to SDR without throwing when unsupported.

## Pass Ownership

The Software pass executor owns renderer-stage dispatch and unsupported-pass
diagnostics. `SoftwareBackend` must not directly select scene packets, invoke
concrete raster passes, or implement pixel loops.

The shadow runtime must separately own CPU targets, rasterization, and
sampling. The reflection runtime must separately own plane selection, target
storage, mirrored-view rendering, and main-surface compositing. Secondary
reflection views must not mutate the application camera or commit main-view
temporal history.

Software shadow sampling must implement the shared PCF and PCSS quality
presets. It must reconstruct each logical bilinear comparison tap from four
depth-buffer comparisons and use the selected slice projection coefficients
for PCSS blocker-depth linearization.

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
