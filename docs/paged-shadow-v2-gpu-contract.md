# Paged Shadow V2 GPU Contract

## Scope
This document defines the WebGPU-only GPU residency and dirty-page rendering
contract for `PagedShadowMap` v2. The public `PagedShadowMap` API remains
unchanged.

## Background
Paged shadow v1 used CPU conservative page requests and CPU page-table
allocation. V2 moves request flags, request compaction, residency allocation,
LRU metadata, and dirty physical page compaction into WebGPU compute passes.
The CPU may still upload frame-local caster bounds and issue grouped draw calls,
but it must not be the authoritative owner of WebGPU page-table allocation.

## API/Contract
- `scene.shadows.createPaged(options)` must continue to create a
  `PagedShadowMap` with kind `"paged-shadow"`.
- WebGPU must keep directional paged shadow rendering as the v2 target. Spot
  and point lights must continue to use atlas fallback.
- The `paged-shadow-page-mark` node must write `pageRequestFlags`,
  `compactedRequests`, and request counters.
- The `paged-shadow-page-allocate` node must update `pageTable`,
  `residencyState`, `freeList`, `dirtyPhysicalPages`, and allocation counters.
- The `paged-shadow-depth` node must read `dirtyPhysicalPages` and render dirty
  physical pages into the physical depth atlas.
- The `paged-shadow-feedback` node must write next-frame feedback flags after
  main scene depth is available. Feedback is consumed one frame later.
- Paged shadow sampling must continue to return fully lit visibility for
  non-resident, out-of-range, or invalid page-table entries.
- Backends without `supportsPagedShadowRendering` must keep atlas fallback
  behavior.

## Usage
```ts
const shadow = scene.shadows.createPaged({
	virtualResolution: 8192,
	pageSize: 128,
	physicalPageCount: 1024,
	maxPagesPerFrame: 128,
	feedbackMode: "screen-feedback",
});

scene.shadows.bind(sun, shadow);
```

Verification:

```bash
bun tests/static/webgpu/test_webgpu_paged_shadow_runtime.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_planner.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_compiler.mjs
bun tests/static/shaders/test_shader_source.mjs
bunx tsc --noEmit
```

## Errors & Diagnostics
- If GPU compute resources cannot be encoded, the runtime may keep conservative
  CPU mirror metadata for diagnostics and fallback, but WebGPU page-table
  buffers must remain valid resources.
- If feedback flags are missing or empty, conservative caster-bound requests
  must still seed residency.
- If dirty page count is zero, the depth pass may be skipped.
- If a physical page is evicted, the previous virtual page-table entry must be
  reset to `0xffffffff`.

## Compatibility / Breaking Changes
N/A
