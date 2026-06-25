# Paged Shadow V1 Contract

## Scope
This document defines the v1 contract for `PagedShadowMap`, shared shadow
layout metadata, and the first WebGPU paged shadow runtime.

## Background
Virtual shadow maps require page tables, physical pages, and residency tracking.
V1 keeps the public `PagedShadowMap` API stable and makes WebGPU the first
backend that renders physical pages. The implementation is intentionally
conservative: it supports directional lights, CPU page requests, CPU allocation,
and lit fallback for missing pages.

## API/Contract
- `scene.shadows.createPaged(options)` must create a `PagedShadowMap`.
- `PagedShadowMap.kind` must be `"paged-shadow"`.
- `ShadowRenderSet.storageMode` must be `"paged"` when the active backend
  advertises `supportsPagedShadowRendering: true`.
- `ShadowRenderSet.storageMode` must be `"atlas"` when the active backend does
  not support paged shadow rendering.
- `ShadowRenderSet.layout.regions` must mirror the active `ShadowRenderSet`
  slices in v1.
- `ShadowLayout.storageMode` must match `ShadowRenderSet.storageMode`.
- `RenderBackendProfile.shadow.supportsPagedShadows` may advertise backend
  support for paged shadow scheduling metadata.
- `RenderBackendProfile.shadow.supportsPagedShadowRendering` must be `true`
  before a backend may keep `PagedShadowMap` in `"paged"` rendering mode.
- WebGPU v1 must support directional paged shadow render sets only. Spot and
  point lights must use atlas fallback.
- WebGPU v1 must use CPU conservative residency. It must project shadow caster
  bounds into each directional cascade page grid and must allocate at most
  `maxPagesPerFrame` new physical pages each frame.
- WebGPU v1 must store `0xffffffff` in `pageTableBuffer` for non-resident
  virtual pages.
- WebGPU v1 must write directional physical depth pages into a `depth32float`
  2D atlas whose side length is `ceil(sqrt(physicalPageCount)) * pageSize`.
- WebGPU v1 shader sampling must treat non-resident, out-of-range, or invalid
  paged resources as fully lit visibility.
- WebGPU v1 must not require a compute shader. The `paged-shadow-page-mark`
  and `paged-shadow-page-allocate` frame graph nodes are CPU hooks in v1.
- GPU screen-feedback, compute allocation, parent clipmap fallback,
  transparent transmittance pages, and spot/point paged shadows are out of
  scope for v1.

## Usage
```ts
import { DirectionalLight, Scene } from "../src";

const scene = new Scene();
const sun = scene.add(new DirectionalLight({ intensity: 2 }));
const shadow = scene.shadows.createPaged({
	size: 2048,
	virtualResolution: 8192,
	pageSize: 128,
	physicalPageCount: 1024,
	maxPagesPerFrame: 128,
});

scene.shadows.bind(sun, shadow);
```

Verification:

```bash
bun tests/static/lighting/test_shadow_manager.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_planner.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_compiler.mjs
bun tests/static/webgpu/test_webgpu_paged_shadow_runtime.mjs
bunx tsc --noEmit
```

## Errors & Diagnostics
- Backends without `supportsPagedShadowRendering` must use atlas fallback
  metadata and must not fail a frame because a scene requested
  `PagedShadowMap`.
- WebGPU must return lit shadow visibility for a missing page table entry,
  invalid physical page index, invalid atlas coordinate, or unavailable paged
  resource.
- WebGPU must keep non-paged shadows on the existing shadow atlas and CSM
  sampling path.
- Implementations must not use `"variance"` to identify paged shadows because
  `"variance"` identifies `VarianceShadowMap`.

## Compatibility / Breaking Changes
The feature is additive. Existing `single`, `variance`, and `cascaded` shadow
maps must retain their current behavior. Backends that only advertise
`supportsPagedShadows` but not `supportsPagedShadowRendering` now resolve
`PagedShadowMap` to atlas fallback rendering.
