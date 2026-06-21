# Paged Shadow V1 Contract

## Scope
This document defines the v1 contract for `PagedShadowMap` and shared shadow
layout metadata.

## Background
Virtual shadow maps require page tables, physical pages, and residency tracking.
V1 introduces the public map kind and backend-facing layout shape without
requiring a backend to render real virtual pages.

## API/Contract
- `scene.shadows.createPaged(options)` must create a `PagedShadowMap`.
- `PagedShadowMap.kind` must be `"paged-shadow"`.
- `ShadowRenderSet.storageMode` must be `"paged"` when the active backend
  advertises `supportsPagedShadows: true`.
- `ShadowRenderSet.storageMode` must be `"atlas"` when the active backend does
  not support paged shadows.
- `ShadowRenderSet.layout.regions` must mirror the active `ShadowRenderSet`
  slices in v1.
- `ShadowLayout.storageMode` must match `ShadowRenderSet.storageMode`.
- `RenderBackendProfile.shadow.supportsPagedShadows` may advertise backend
  support for paged shadow scheduling.
- WebGPU may expose paged shadow frame graph nodes in v1, but those nodes must
  not require page-table shader bindings or physical page rendering.

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
```

## Errors & Diagnostics
- Backends without `supportsPagedShadows` must use atlas fallback metadata and
  must not fail a frame because a scene requested `PagedShadowMap`.
- WebGPU paged shadow graph nodes must be no-op stubs until physical page
  resources are implemented.
- Implementations must not use `"variance"` to identify paged shadows because `"variance"`
  identifies `VarianceShadowMap`.

## Compatibility / Breaking Changes
The feature is additive. Existing `single`, `variance`, and `cascaded` shadow maps must
retain their current behavior.
