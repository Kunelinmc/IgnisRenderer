# WebGPU Clustered Direct Light Contract

## Scope

This document defines the WebGPU clustered direct-light contract for
`PointLight`, `SpotLight`, and `AreaLight`, including culling mode selection,
storage bindings, hard limits, and cache invalidation.

## Background

WebGPU clustered lighting uses cluster-centric gather culling by default. Each
compute workgroup owns one cluster and tests the active light set in deterministic
light-index order. The retained scatter implementation exists for performance and
image A/B comparisons.

Direct-light data uses semantic structure-of-arrays storage. Culling reads compact
bounds and only loads direction or area payload data when the light type requires
it. Shading reads the same semantic buffers and scans each cluster list once.

## API/Contract

- `ClusteredLightingOptions.cullingMode` may be `"gather"` or `"scatter"` and
  defaults to `"gather"`.
- WebGL must accept `cullingMode` and ignore it. This option must not change the
  WebGL clustered-light data format or culling algorithm.
- WebGPU must clamp `ClusteredLightingOptions.maxLights` to `1024` and must emit
  `webgpu-clustered-max-lights-limit` once when the requested value exceeds the
  limit.
- WebGPU must clamp `ClusteredLightingOptions.maxLightsPerCluster` to `128` and
  must emit `webgpu-clustered-max-per-cluster-limit` once when the requested value
  exceeds the limit.
- Gather culling must use one 128-thread workgroup per cluster and a fixed index
  span of `maxLightsPerCluster` entries.
- Gather culling must write cluster headers directly and must not execute clear or
  finalize passes.
- Overflow clusters must be resolved by deterministic estimated-contribution
  ordering. Equal scores must prefer the smaller light index.
- Scatter culling must remain available through `cullingMode: "scatter"` and must
  use the same SoA light buffers as gather culling.
- The scene bind group must contain `params` plus seven read-only storage bindings:
  `positionRange`, `directionOuter`, `colorInner`, `areaPayload`, `metadata`,
  `headers`, and `indices`.
- `metadata` must store packed flags and shadow index as two `u32` values.
- `areaPayload` must store `rightWidth`, `upHeight`, and `normalAreaScale` as three
  `vec4<f32>` values.
- The culling bind group must use `cullData` records containing position, cull
  radius, range, score inputs, and inner cone cosine. It must also use a CPU-built
  Z-slice depth buffer.
- Point and spot cull radius must equal light range. Area cull radius must equal
  range plus the rectangle half-diagonal.
- Unchanged grid and culling signatures must reuse cluster headers and indices.
  Unchanged semantic signatures must skip the corresponding `writeBuffer` call.
- Camera, viewport, grid option, culling mode, shader runtime, or buffer identity
  changes must invalidate the affected cached culling or binding state.
- `DirectionalLight` must remain a global frame-uniform light and must not be
  appended to per-cluster indices.

## Usage

```ts
renderer.features.enableClusteredLighting = true;
renderer.features.clusteredLightingOptions = {
	tileSizePx: 64,
	zSlices: 24,
	maxLights: 1024,
	maxLightsPerCluster: 128,
	cullingMode: "gather",
};
```

For an A/B comparison, the application may retain all other options and select
the scatter implementation:

```ts
renderer.features.clusteredLightingOptions = {
	...renderer.features.clusteredLightingOptions,
	cullingMode: "scatter",
};
```

## Errors & Diagnostics

- `webgpu-clustered-perspective-only` is emitted once when clustered lighting is
  enabled for a non-perspective camera.
- `webgpu-clustered-light-budget` is emitted once when the scene light count
  exceeds the effective `maxLights` budget.
- `webgpu-clustered-max-lights-limit` is emitted once when `maxLights` exceeds
  `1024`.
- `webgpu-clustered-max-per-cluster-limit` is emitted once when
  `maxLightsPerCluster` exceeds `128`.

## Compatibility / Breaking Changes

This change breaks the WebGPU clustered-light shader and backend binding contract.
Code that supplied or inspected the previous 28-float AoS light record must migrate
to the semantic SoA bindings. Values above `1024` lights or `128` lights per cluster
are no longer accepted by the WebGPU backend and are clamped with a warning.

The public option object remains source-compatible because `cullingMode` is
optional. Existing applications that do not specify it now use gather culling
instead of scatter culling.
