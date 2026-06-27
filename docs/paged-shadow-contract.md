# Paged Shadow Contract

## Scope

This document defines the contract for `PagedShadowMap` and its associated GPU residency, dirty-page rendering, and frame graph scheduling in IgnisRenderer.

## Background

Virtual shadow maps require page tables, physical pages, and residency tracking. The paged shadow implementation maps virtual shadow maps to physical pages using GPU-driven allocation, residency tracking, and dirty page rendering.

## API/Contract

- `scene.shadows.createPaged(options)` must create a `PagedShadowMap` with the `kind` property set to `"paged-shadow"`.
- `ShadowRenderSet.storageMode` must be `"paged"` when the active rendering backend supports paged shadow rendering (i.e. `RenderBackendProfile.shadow.supportsPagedShadowRendering` is `true`).
- `ShadowRenderSet.storageMode` must be `"atlas"` when the active backend does not support paged shadow rendering.
- `ShadowRenderSet.layout.regions` must mirror the active slices of the `ShadowRenderSet`.
- `ShadowLayout.storageMode` must match `ShadowRenderSet.storageMode`.
- `RenderBackendProfile.shadow.supportsPagedShadows` may advertise backend support for paged shadow scheduling metadata.
- `RenderBackendProfile.shadow.supportsPagedShadowRendering` must be `true` before a backend may keep `PagedShadowMap` in `"paged"` rendering mode.
- The rendering backend must support directional paged shadow render sets only. Spot and point lights must use shadow atlas fallback.
- The GPU residency and dirty-page allocation (including request flags, request compaction, residency allocation, LRU metadata, and dirty physical page compaction) must execute in GPU compute passes. The CPU may upload frame-local caster bounds and issue grouped draw calls, but the GPU must be the authoritative owner of page-table allocation.
- The GPU page table and residency buffers are authoritative after initialization. CPU mirrors may exist for diagnostics but must not decide page residency.
- The paged shadow runtime must implement the following frame graph nodes:
  - The `paged-shadow-page-mark` node: Must write `pageRequestFlags`, `compactedRequests`, and request counters on the GPU.
  - The `paged-shadow-page-allocate` node: Must update `pageTable`, `residencyState`, `freeList`, `dirtyPhysicalPages`, and allocation counters on the GPU.
  - The `paged-shadow-depth` node: Must read `dirtyPhysicalPages` and render depth information into the physical depth atlas.
  - The `paged-shadow-feedback` node: Must write next-frame feedback flags after main scene depth is available, to be consumed on the subsequent frame.
- The `paged-shadow-depth` node must build paged shadow draw instance buffers and `drawIndexedIndirect` argument records on the GPU before the render pass. The CPU may bind each draw candidate's geometry and animation state, but it must not enumerate dirty pages or build per-page MVP instances.
- If feedback flags are unavailable or empty (such as on the initial frame), the backend must seed page requests from conservative caster bounds.
- CPU frame uploads may include one-frame tombstone caster bounds for removed shadow casters. The GPU must use those bounds to mark affected resident pages dirty, but tombstones must not produce depth draw instances.
- WebGPU must write physical depth pages into a `depth32float` 2D texture atlas whose side length is `ceil(sqrt(physicalPageCount)) * pageSize`.
- Paged shadow shader sampling must treat non-resident, out-of-range, or invalid page-table entries as fully lit visibility.

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
	feedbackMode: "screen-feedback",
});

scene.shadows.bind(sun, shadow);
```

### Verification Commands

```bash
bun tests/static/lighting/test_shadow_manager.mjs
bun tests/static/webgpu/test_webgpu_paged_shadow_runtime.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_planner.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_compiler.mjs
bun tests/static/shaders/test_shader_source.mjs
bunx tsc --noEmit
```

## Errors & Diagnostics

- Backends without `supportsPagedShadowRendering` must use atlas fallback metadata and must not fail a frame because a scene requested `PagedShadowMap`.
- If GPU compute resources cannot be encoded, the runtime may keep conservative CPU mirror metadata for diagnostics and fallback, but the GPU page-table buffers must remain valid resources.
- If feedback flags are missing or empty, conservative caster-bound requests must still seed residency without making the CPU the residency owner.
- If the count of dirty pages is zero, the depth pass may be skipped.
- If a physical page is evicted, the previous virtual page-table entry must be reset to `0xffffffff`.
- Debug state may report buffer capacities and last uploaded candidate counts. Resident and dirty counts must come from GPU counters or explicit readback, not from CPU-owned page maps.
- Implementations must not use `"variance"` to identify paged shadows because `"variance"` identifies `VarianceShadowMap`.

## Compatibility / Breaking Changes

N/A
