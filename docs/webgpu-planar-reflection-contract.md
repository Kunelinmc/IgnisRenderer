# WebGPU Planar Reflection Contract
## Scope
This document defines the v1 contract for WebGPU planar reflections backed by
the existing `Material.reflectivity` and `Material.mirrorPlane` API.
The contract applies only to `WebGPUBackend`.

## Background
Software rendering already supports planar reflection from mirror materials.
WebGPU v1 adds a bounded hardware path that captures mirrored scene content into
offscreen textures and composites it back onto mirror receiver pixels.
WebGL remains outside this contract.

## API/Contract
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
- Reflection capture must use a `reflection-capture` pipeline variant with
  flipped front-face winding.
- Main scene rendering must render the base material first.
- Composite must run after opaque or deferred lighting output and before
  transparent scene rendering.
- Composite must write a planar reflection mask target for mirror receiver
  pixels.
- SSR compose must skip pixels marked by the planar reflection mask.
- MSAA rendering must composite into the MSAA scene color and resolve into the
  main scene color target.
- Non-MSAA rendering may composite directly into the main scene color target.
- Orthographic cameras must skip planar reflection in v1 and emit a once-only
  diagnostic.
- Roughness blur, recursive planar reflection, per-material resolution,
  particle reflection, and WebGL parity are not part of v1.

## Usage
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
bun tests/static/webgpu/test_webgpu_frame_executor_resilience.mjs
```

The WebGPU frame planner should schedule the `reflection` frame pass when
`enableReflection` is true and the prepared scene contains reflective packets.

## Errors & Diagnostics
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

## Compatibility / Breaking Changes
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
