# Clustered Lighting Contract

## Scope

This document defines the cross-backend clustered lighting contract for `PointLight`, `SpotLight`, and `AreaLight`. It specifies culling modes, storage configurations, texture packings, limits, culling radius rules, and cache invalidation policies for both `WebGLBackend` and `WebGPUBackend`.

## Background

Clustered lighting divides the camera frustum into a 3D grid of clusters (tiles along the X/Y screen coordinates, and log-spaced slices along the Z/depth axis). Dynamic lights are culled against these clusters to build per-cluster light lists, allowing forward and deferred shaders to evaluate only the lights overlapping with each screen fragment.

- **`WebGLBackend`**: Leverages CPU-based light-to-cluster culling and packs the resulting data into dynamic floating-point textures to bypass WebGL 2's storage buffer limitations.
- **`WebGPUBackend`**: Leverages GPU-based compute shaders to cull lights and writes the resulting indices to storage buffers.

## API/Contract

### Options
Clustered lighting is configured via `ClusteredLightingOptions` on `RendererFeatureFlags.clusteredLightingOptions`.
- `tileSizePx` (number, optional): Tile size in pixels (must be $\ge 8$, defaults to `64`).
- `zSlices` (number, optional): Number of depth slices (must be $\ge 1$, defaults to `24`).
- `maxLights` (number, optional): Maximum number of active clustered lights per frame (must be $\ge 1$, defaults to `256`).
- `maxLightsPerCluster` (number, optional): Maximum lights stored in any single cluster (must be $\ge 1$, defaults to `64`).
- `cullingMode` (`"gather"` | `"scatter"`, optional): WebGPU-specific culling strategy. Defaults to `"gather"`. WebGL must accept and ignore this option.

---

### WebGL Backend Specification

1. **Camera Limit**: WebGL clustered lighting must only support perspective cameras. Clustered lighting must be disabled if the camera is orthographic or has an invalid depth range (where $\log(\text{far}) - \log(\text{near}) \le 10^{-6}$).
2. **Cluster Limits**: `maxLightsPerCluster` must be clamped to `MAX_CLUSTER_LIGHTS_PER_FRAGMENT` (defined as `512`).
3. **CPU Culling**: The culling process is run on the CPU. Light index assignments to clusters must be determined by bounding boxes constructed in screen-and-slice space.
4. **WebGL Textures**:
   Clustered data must be uploaded to three float textures:
   - **Header Texture (`uClusterHeaderTexture`)**: Formatted as `RGBA32F`. Each texel corresponds to a cluster index (linear index $x + y \cdot \text{tilesX} + z \cdot \text{tilesX} \cdot \text{tilesY}$).
     - `x`: Light index offset within the index texture.
     - `y`: Count of active lights overlapping this cluster.
     - `z`: Cluster flags (e.g., `1` if cluster light count exceeds `maxLightsPerCluster`, signaling overflow).
     - `w`: Reserved/unused (`0.0`).
   - **Index Texture (`uClusterIndexTexture`)**: Formatted as `RGBA32F`. Contains packed light indices for the clusters. Each texel stores up to four light indices in its components (`x`, `y`, `z`, `w`).
   - **Light Texture (`uClusterLightTexture`)**: Formatted as `RGBA32F`. Stores structured light parameters. Each light is represented by a sequence of 4 texels (16 floats):
     - **Texel 0**: `position.x`, `position.y`, `position.z`, `range`
     - **Texel 1**: `direction.x`, `direction.y`, `direction.z`, `outerCos`
     - **Texel 2**: `color.r`, `color.g`, `color.b`, `innerCos`
     - **Texel 3**: `type` (`0` for point, `1` for spot), `castsShadow` (`0` or `1`), `shadowIndex`, `0.0` (reserved)

---

### WebGPU Backend Specification

1. **Light Range & Culling Limits**:
   - `maxLights` must be clamped to a maximum of `1024`.
   - `maxLightsPerCluster` must be clamped to a maximum of `128`.
2. **GPU Culling Modes**:
   - **Gather Mode (`"gather"`)**: Uses one 128-thread workgroup per cluster. Workgroups test the active light list against their cluster boundaries in parallel. Out-of-bounds or overflow lights are discarded based on a deterministic estimated-luminance contribution score.
   - **Scatter Mode (`"scatter"`)**: Incurs three sequential compute passes:
     - `ClearPass`: Clears atomic counters and cluster headers.
     - `ScatterPass`: Evaluates each light's influence sphere and appends the light's index to intersecting clusters.
     - `FinalizePass`: Resolves cluster index ranges.
3. **Structured Storage Layout (`clusteredSceneBindGroupLayout` / `group(2)`)**:
   Shaders reading cluster data must declare the following bindings:
   - **Binding 0 (`uniform`)**: `clusterGrid` storing `ClusterGridParams`:
     ```wgsl
     struct ClusterGridParams {
         screenWidth: u32,
         screenHeight: u32,
         tilesX: u32,
         tilesY: u32,
         zSlices: u32,
         clusterCount: u32,
         near: f32,
         far: f32,
         logScale: f32,
         logBias: f32,
         lightCount: u32,
         maxLightsPerCluster: u32,
     }
     ```
   - **Binding 1 (`read-only-storage`)**: `positionRange` storing `vec4<f32>` (XYZ = position, W = range).
   - **Binding 2 (`read-only-storage`)**: `directionOuter` storing `vec4<f32>` (XYZ = direction, W = outerCos).
   - **Binding 3 (`read-only-storage`)**: `colorInner` storing `vec4<f32>` (XYZ = color, W = innerCos).
   - **Binding 4 (`read-only-storage`)**: `areaPayload` storing `ClusterAreaPayload`:
     ```wgsl
     struct ClusterAreaPayload {
         rightWidth: vec4<f32>,
         upHeight: vec4<f32>,
         normalAreaScale: vec4<f32>,
     }
     ```
   - **Binding 5 (`read-only-storage`)**: `metadata` storing `ClusterMetadata`:
     ```wgsl
     struct ClusterMetadata {
         packedFlags: u32,
         shadowIndex: u32,
     }
     ```
   - **Binding 6 (`read-only-storage`)**: `headers` storing `ClusterHeader`:
     ```wgsl
     struct ClusterHeader {
         offset: u32,
         count: u32,
         flags: u32,
         reserved: u32,
     }
     ```
   - **Binding 7 (`read-only-storage`)**: `indices` storing `array<u32>` of light indices.

4. **Compute Culling Layout (`group(0)` entries for compute passes)**:
   - **Binding 0 (`uniform`)**: `params` (`ClusterGridParams`).
   - **Binding 1 (`read-only-storage`)**: `cullData` (`ClusterCullData` containing `positionCullRadius` `vec4<f32>` and `scoreParams` `vec4<f32>`).
   - **Binding 2 (`read-only-storage`)**: `directionOuter` (`array<vec4<f32>>`).
   - **Binding 3 (`read-only-storage`)**: `areaPayload` (`array<ClusterAreaPayload>`).
   - **Binding 4 (`read-only-storage`)**: `metadata` (`array<ClusterMetadata>`).
   - **Binding 5 (`storage`)**: `headers` (`array<ClusterHeader>` with atomic counts/flags in culling shaders).
   - **Binding 6 (`storage`)**: `indices` (`array<u32>`).
   - **Binding 7 (`read-only-storage`)**: `sliceDepths` (`array<f32>`).

---

### Math and Cull Radius
- **Point and Spot Lights**: The cull radius must equal the light's `range`.
- **Area Lights**: The cull radius must equal the light's `range` plus the rectangle's half-diagonal:
  $$\text{cullRadius} = \text{range} + \sqrt{(\text{width} \cdot 0.5)^2 + (\text{height} \cdot 0.5)^2}$$
- **Directional Lights**: Directional lights are global uniforms and must never be appended to per-cluster lists.

### Invalidation
The cached culling state and bindings must be invalidated when:
- The camera transform, projection matrices, or viewport size changes.
- Tile size, slice count, or culling mode configuration changes.
- Shader runtimes or backing buffer/texture identities are recreated.

## Usage

### TypeScript Setup
```ts
// Enable clustered lighting globally
renderer.features.enableClusteredLighting = true;
renderer.features.clusteredLightingOptions = {
	tileSizePx: 64,
	zSlices: 24,
	maxLights: 512,
	maxLightsPerCluster: 64,
	cullingMode: "gather", // WebGPU-only strategy, WebGL ignores this
};
```

---

### WebGL GLSL Shader Integration

The WebGL scene fragment shaders fetch light indexes and structures from bound textures.
```glsl
#version 300 es
precision highp float;

uniform int uEnableClusteredLighting;
uniform vec4 uClusterParams0; // screenWidth, screenHeight, tilesX, tilesY
uniform vec4 uClusterParams1; // zSlices, maxLightsPerCluster, logScale, logBias

uniform sampler2D uClusterHeaderTexture;
uniform sampler2D uClusterIndexTexture;
uniform sampler2D uClusterLightTexture;

uniform vec2 uClusterHeaderTexSize;
uniform vec2 uClusterIndexTexSize;
uniform vec2 uClusterLightTexSize;

// Fetch and evaluate lights for the current fragment
void main() {
    float width = max(uClusterParams0.x, 1.0);
    float height = max(uClusterParams0.y, 1.0);
    int tilesX = max(int(floor(uClusterParams0.z + 0.5)), 1);
    int tilesY = max(int(floor(uClusterParams0.w + 0.5)), 1);
    int zSlices = max(int(floor(uClusterParams1.x + 0.5)), 1);
    
    int tileX = clamp(int(floor((gl_FragCoord.x / width) * float(tilesX))), 0, tilesX - 1);
    int tileY = clamp(int(floor((gl_FragCoord.y / height) * float(tilesY))), 0, tilesY - 1);
    
    // Logarithmic Z distribution
    float viewDepth = max(vViewDepth, 1e-4);
    int slice = clamp(int(floor(log(viewDepth) * uClusterParams1.z + uClusterParams1.w)), 0, zSlices - 1);
    
    int clusterIndex = tileX + tileY * tilesX + slice * tilesX * tilesY;
    
    // Fetch header
    ivec2 headerTexel = linearIndexToTexel(clusterIndex, uClusterHeaderTexSize);
    vec4 header = texelFetch(uClusterHeaderTexture, headerTexel, 0);
    int offset = int(floor(header.x + 0.5));
    int count = int(floor(header.y + 0.5));
    
    for (int i = 0; i < count; ++i) {
        int listIndex = offset + i;
        int lightIndex = fetchClusterListLightIndex(listIndex);
        
        // Fetch light row properties
        vec4 row0 = fetchClusterLightRow(lightIndex, 0); // Position & Range
        vec4 row1 = fetchClusterLightRow(lightIndex, 1); // Direction & outerCos
        vec4 row2 = fetchClusterLightRow(lightIndex, 2); // Color & innerCos
        vec4 row3 = fetchClusterLightRow(lightIndex, 3); // Type & Shadow metadata
        
        // Shading calculations...
    }
}
```

---

### WebGPU WGSL Shader Integration

The WebGPU scene fragment shaders use storage buffers to load light variables.
```wgsl
@group(2) @binding(0) var<uniform> clusterGrid: ClusterGridParams;
@group(2) @binding(1) var<storage, read> positionRange: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read> directionOuter: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read> colorInner: array<vec4<f32>>;
@group(2) @binding(4) var<storage, read> areaPayload: array<ClusterAreaPayload>;
@group(2) @binding(5) var<storage, read> metadata: array<ClusterMetadata>;
@group(2) @binding(6) var<storage, read> headers: array<ClusterHeader>;
@group(2) @binding(7) var<storage, read> indices: array<u32>;

fn getClusterIndex(fragCoord: vec4<f32>, viewDepth: f32) -> u32 {
    let tileX = u32(fragCoord.x / f32(clusterGrid.screenWidth) * f32(clusterGrid.tilesX));
    let tileY = u32(fragCoord.y / f32(clusterGrid.screenHeight) * f32(clusterGrid.tilesY));
    let slice = u32(log(max(viewDepth, 1e-4)) * clusterGrid.logScale + clusterGrid.logBias);
    return clamp(tileX, 0u, clusterGrid.tilesX - 1u) + 
           clamp(tileY, 0u, clusterGrid.tilesY - 1u) * clusterGrid.tilesX + 
           clamp(slice, 0u, clusterGrid.zSlices - 1u) * clusterGrid.tilesX * clusterGrid.tilesY;
}
```

## Errors & Diagnostics

- `webgpu-point-limit`: Emitted once per logger lifecycle when WebGPU forward
  shading skips point lights beyond `MAX_POINT_LIGHTS`. Repeated frames that
  remain over budget must not emit the warning again.
- `webgl-clustered-fragment-light-budget` / `webgpu-clustered-light-budget`: Emitted once if the active scene light count exceeds the configured `maxLights` budget.
- `webgl-clustered-perspective-only` / `webgpu-clustered-perspective-only`: Emitted once when clustered lighting is enabled on a non-perspective camera.
- `webgl-clustered-invalid-depth-range`: Emitted once on the WebGL backend if the camera near/far depth range is negative, zero, or too small for log slicing.
- `webgpu-clustered-max-lights-limit`: Emitted once on the WebGPU backend if the requested `maxLights` exceeds `1024`.
- `webgpu-clustered-max-per-cluster-limit`: Emitted once on the WebGPU backend if the requested `maxLightsPerCluster` exceeds `128`.
- `webgl-clustered-texture-size-overflow`: Emitted once on WebGL if the required cluster count or light data exceeds the maximum texture size supported by the GPU.

## Compatibility / Breaking Changes

- **Renaming**: The legacy `webgpu-clustered-direct-light-contract.md` has been replaced by `clustered-lighting-contract.md` to reflect cross-backend support.
- **WebGL Fallbacks**: Applications targetting WebGL will automatically fall back to the legacy forward lighting path without crash/panic when an orthographic camera is bound or if texture size limits are exceeded.
- **WebGPU Buffer Sizes**: Clustered lighting options above `1024` max lights or `128` lights per cluster are automatically clamped to those respective hardware budgets on WebGPU.
