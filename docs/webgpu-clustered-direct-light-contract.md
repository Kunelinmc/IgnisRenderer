# WebGPU Clustered Direct Light Contract

## Scope

This document defines the WebGPU clustered direct-light contract for `PointLight`,
`SpotLight`, and `AreaLight`.

## Background

WebGPU clustered lighting previously used clustered storage only for `PointLight`
and `SpotLight`. `AreaLight` used the legacy frame uniform array and was capped by
`WEBGPU_MAX_AREA_LIGHTS`.

## API/Contract

- `renderer.features.enableClusteredLighting` must enable clustered culling for
  `PointLight`, `SpotLight`, and `AreaLight` on perspective cameras.
- `ClusteredLightingOptions.maxLights` must cap the total clustered direct-light
  records across `PointLight`, `SpotLight`, and `AreaLight`.
- `AreaLight` clustered records must carry `positionRange`, `rightWidth`,
  `upHeight`, `normalAreaScale`, and `colorInner` data in storage buffers.
- `DirectionalLight` must remain a global frame-uniform light and must not be
  appended to per-cluster light indices.
- `SpotLight` instances beyond the WebGPU spot shadow budget must keep direct
  lighting and must disable clustered shadow sampling for those extra records.
- Legacy point, spot, and area frame uniform arrays may remain populated up to
  their existing limits for clustered fallback paths.

## Usage

```ts
renderer.features.enableClusteredLighting = true;
renderer.features.clusteredLightingOptions = {
	tileSizePx: 64,
	zSlices: 24,
	maxLights: 512,
	maxLightsPerCluster: 96,
};
```

## Errors & Diagnostics

- `webgpu-clustered-perspective-only`: triggered when clustered lighting is
  enabled for a non-perspective camera.
- `webgpu-clustered-light-budget`: triggered when clustered direct-light count
  exceeds `ClusteredLightingOptions.maxLights`.
- `webgpu-clustered-overflow`: triggered when a cluster may exceed
  `ClusteredLightingOptions.maxLightsPerCluster`.
- `webgpu-clustered-spot-shadow-budget`: triggered when extra shadowed spot
  lights keep lighting but lose clustered shadow sampling.

## Compatibility / Breaking Changes

No public API is removed. In clustered WebGPU frames, `AreaLight` contribution is
resolved from clustered storage instead of the legacy frame uniform array.
