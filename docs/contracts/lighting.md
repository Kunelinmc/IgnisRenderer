# Lighting Contract

This document defines clustered lighting, environment IBL, irradiance grids, light probes, and reflection probes across rendering backends.

## Contract

### Units and world scale

- World-space positions, `range`, `thicknessFactor`, and
  `attenuationDistance` must use meters.
- Directional and ambient light `intensity` must represent lux-equivalent RGB
  irradiance. Point and spot light `intensity` must represent
  candela-equivalent RGB luminous intensity. Area light `intensity` must
  represent emitted-radiance-equivalent RGB values.
- Point and spot attenuation must be
  `pow(clamp(1 - pow(distance / range, 4), 0, 1), 2) /
  max(distance * distance, 0.0001)`. The denominator floor represents one
  centimeter and must be shared by runtime shading and probe capture.
- Spot cone attenuation must use `smoothstep(outerCos, innerCos, cosTheta)`.
- SH coefficients must store radiance. A Lambertian consumer must evaluate
  irradiance from SH and contribute `albedo * irradiance / PI` exactly once.
- A scene with no analytical light, environment, probe, or emissive source must
  produce black scene radiance.

### Material light transport

- Built-in PBR shading must conserve energy across diffuse, base specular,
  clearcoat, sheen, and transmission lobes.
- Analytical `AmbientLight` color must be consumed as view-independent diffuse
  irradiance. It must not be reused as fallback specular radiance, and its
  diffuse contribution must not be attenuated by view-dependent Fresnel.
  Environment maps and SH radiance may retain Fresnel-weighted diffuse and
  specular image-based lighting.
- Double-sided shading must orient the geometric normal from the rasterizer's
  front-facing classification. It must not flip a shading normal by testing
  `dot(normal, viewDir)`, because interpolated or hard vertex normals may cross
  that threshold while the rasterized face orientation remains unchanged.
  Normal-mapped and clearcoat normals must remain in the hemisphere of their
  oriented geometric reference normal.
- Dielectric F0 must be
  `0.16 * reflectance * reflectance * specularColor * specularFactor`;
  metallic F0 must use base color and must remain reflective when
  `specularFactor` is zero.
- Split-sum specular IBL must reconstruct ordinary material reflection as
  `F0 * A + B`. Iridescent material may replace F0 with the effective
  thin-film Fresnel approximation but must not apply Schlick Fresnel twice.
- Legacy Phong and Gouraud shading must use energy-normalized Blinn-Phong.
  Encoded diffuse, specular F0, and ambient reflectance must be decoded to
  linear light before evaluation. Direct lighting must use Schlick Fresnel,
  `(1 - F) * diffuse / PI`, and
  `F * ((shininess + 8) / (8 * PI)) * pow(NdotH, shininess)`.

### Clustered lighting

#### Options
Clustered lighting is configured via `ClusteredLightingOptions` on `RendererFeatureFlags.clusteredLightingOptions`.
- `tileSizePx` (number, optional): Tile size in pixels (must be $\ge 8$, defaults to `64`).
- `zSlices` (number, optional): Number of depth slices (must be $\ge 1$, defaults to `24`).
- `maxLights` (number, optional): Maximum number of active clustered lights per frame (must be $\ge 1$, defaults to `256`).
- `maxLightsPerCluster` (number, optional): Maximum lights stored in any single cluster (must be $\ge 1$, defaults to `64`).
- `cullingMode` (`"gather"` | `"scatter"`, optional): WebGPU-specific culling strategy. Defaults to `"gather"`. WebGL must accept and ignore this option.

---

#### WebGL Backend Specification

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

#### WebGPU Backend Specification

1. **Light Range & Culling Limits**:
   - `maxLights` must be clamped to a maximum of `1024`.
   - `maxLightsPerCluster` must be clamped to a maximum of `128`.
2. **GPU Culling Modes**:
   - **Gather Mode (`"gather"`)**: Uses one 128-thread workgroup per cluster. Workgroups test the active light list against their cluster boundaries in parallel. Out-of-bounds or overflow lights are discarded based on a deterministic estimated-luminance contribution score.
   - **Scatter Mode (`"scatter"`)**: Incurs three sequential compute passes:
     - Clear atomic counters and cluster headers.
     - Evaluate each light's influence sphere and append its index to
       intersecting clusters.
     - Resolve cluster index ranges.
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

#### Math and Cull Radius
- **Point and Spot Lights**: The cull radius must equal the light's `range`.
- **Area Lights**: The cull radius must equal the light's `range` plus the rectangle's half-diagonal:
  $$\text{cullRadius} = \text{range} + \sqrt{(\text{width} \cdot 0.5)^2 + (\text{height} \cdot 0.5)^2}$$
- **Directional Lights**: Directional lights are global uniforms and must never be appended to per-cluster lists.

#### Invalidation
The cached culling state and bindings must be invalidated when:
- The camera transform, projection matrices, or viewport size changes.
- Tile size, slice count, or culling mode configuration changes.
- Shader runtimes or backing buffer/texture identities are recreated.

### Environment IBL

- `IBLPrefilter` must accept an `IBLPrefilterServiceOptions` object at
  construction time. The object may provide one `IRenderBackend`.
- Per-call `IBLPrefilterOptions` must contain only execution settings and must
  not override the service backend.
- The one-shot helper must receive construction settings through
  `PrefilterEnvironmentIBLOptions.service`.
- When an `IRenderBackend` is provided, `IBLPrefilter` must resolve generic
  WebGPU compute and WebGL auxiliary raster capabilities and construct the
  corresponding lighting-owned executors. It must not inspect backend ids,
  initialize, restore, or destroy the backend.
- Lighting-owned CPU, Worker, WebGPU, and WebGL executors must implement the
  same internal execution contract.
- Backend executors must report an id, availability state, whether they accept
  requests in that state, and a descriptive unavailable reason.
- Work plans must contain output dimensions, mip levels, and roughness only.
  Source revision data used by deferred work must travel with the execution
  request instead of the work plan.
- `IBLPrefilter.prefilter(envMap, options)` must return an HDR `Texture` whose
  `mipmaps` encode roughness levels.
- Equirectangular input and output directions must use
  `phi = u * 2 * PI - PI` and `theta = v * PI`, matching runtime and shader
  environment sampling.
- Prefiltered equirectangular output must use horizontal repeat and vertical
  clamp addressing. Sampling across the longitude seam may wrap, while the
  north and south poles must not wrap into each other.
- `prefilterEnvironmentIBL(envMap, options)` must provide a one-shot helper with
  the same behavior as constructing `IBLPrefilter` and calling `prefilter`.
- `projectEnvironmentTextureToSH(envMap, options)` must project a valid
  environment texture into radiance SH coefficients and must not prefilter
  specular IBL data.
- Environment SH projection must interpret byte-backed texture channels in the
  `0..255` domain and `Float32Array` channels in the normalized or HDR linear
  domain. Equivalent byte-backed and float-backed inputs must project to the
  same radiance after color-space decoding.
- `IBLPrefilterOptions` must use `maxSampleWidth`, `maxSampleHeight`, and
  `maxMipLevels` for output limits.
- `IBLPrefilterOptions.acceleration` must support `auto`, `single-thread`,
  `multi-thread`, `webgpu`, and `webgl`.
- `single-thread` must execute prefiltering synchronously on the calling
  JavaScript thread. `multi-thread` must distribute mip work through the Worker
  scheduler.
- If `acceleration` is `webgpu`, the configured backend must expose an
  accepting `WEBGPU_COMPUTE_EXTENSION`. If it is `webgl`, the backend must
  expose an accepting `WEBGL_AUXILIARY_RASTER_EXTENSION`.
- An explicit WebGL request made while the context is lost must wait in the
  backend context work queue until restoration or cancellation. A WebGL request
  already executing when loss occurs must reject and must not replay.
- WebGL fragment acceleration must require `EXT_color_buffer_float` and either
  `OES_texture_float_linear` or `OES_texture_half_float_linear`.
- If `acceleration` is `auto`, a ready backend executor must take priority over
  Worker and single-thread CPU execution.
- Executor failures after execution starts must be surfaced and must not cause
  `auto` to replay work through another executor.
- `auto` must treat a lost WebGL context as temporarily unavailable and fall
  back immediately instead of waiting for restoration.
- If `acceleration` is `webgpu`, an `IRenderBackend` without the matching IBL
  executor or with an unavailable WebGPU device or queue must cause a
  descriptive error.
- `IBLPrefilter` and `prefilterEnvironmentIBL` must not accept direct WebGPU
  compute sources, `Renderer` instances, or renderer-like wrappers as service
  options.
- `Renderer` must not expose environment IBL prefilter or update methods.
- `Renderer.warmup()` must not create `LightProbe` instances, assign
  `LightProbe.sh`, or assign `ReflectionProbe.prefilteredMap`.
- `maxMipLevels` is an upper bound. The resolved output mip count must not
  exceed the natural mip chain of the resolved base dimensions.
- Multi-level prefiltered textures and reflection-probe atlases must use
  `LinearMipmapLinear`; single-level textures must use `Linear`.
- WebGL fragment acceleration must read each generated mip back into a
  backend-agnostic `Float32Array`. It must not expose or retain native WebGL
  texture handles in the returned `Texture`.

### Irradiance probe grids

`LightType.IrradianceProbeGrid` must identify grid lights as
`"irradianceProbeGrid"`. `IrradianceProbeGrid` is part of `SceneLight` and the
public `src/lights` exports.

`IrradianceProbeGridParams.dimensions` must be provided as `{ x, y, z }`. Each
axis is floored to an integer and clamped to at least `1`. The total cell count
must be less than or equal to `256`; construction throws `RangeError` when the
limit is exceeded.

`halfExtents` defaults to `{ x: 5, y: 5, z: 5 }`. The grid volume is the node
transform multiplied by this local box. `blendDistance` defaults to `0.15` and
controls edge fade outside the normalized box metric. `priority` defaults to
`0`; if multiple grids exist, the renderer selects the highest priority grid,
then the nearest grid center to the camera, then the lowest `id`.

`source` defaults to `"manual"`. `"capturedScene"` uses `ProbeCaptureRuntime`
to capture one explicitly requested cell at a time. Capture fields match
`LightProbe`: `captureResolution`, `captureFar`, `includeEnvironment`,
`includeMeshes`, `includeTransparent`, `includeParticles`, and
`includeShadows`.

Grid SH data must use the engine SH contract: L=3 with `16` coefficients per
cell. Cell indexing is `x` fastest, then `y`, then `z`.

`IrradianceProbeGrid.sh` and `getCellSH()` return mutable cell-owned SH
storage for backward compatibility. Direct mutation of `sh`, a cell SH array,
or a coefficient component must advance `textureRevision` and
`captureRevision`, mark the touched cell valid, and invalidate scene lighting.
Applications should use `setCellSH()` for authored writes and `clearCell()` for
invalidating a cell.

Public methods:

```ts
getCellIndex(x: number, y: number, z: number): number;
getCellSH(indexOrCoord: number | { x: number; y: number; z: number }): SHCoefficients;
setCellSH(indexOrCoord: number | { x: number; y: number; z: number }, coeffs: SHCoefficients): void;
clearCell(indexOrCoord: number | { x: number; y: number; z: number }): void;
requestCapture(indexOrCoord?: number | { x: number; y: number; z: number }): void;
getRuntimeCache(): IrradianceProbeGridRuntimeCache;
```

`setCellSH` must mark the cell valid and advance the texture revision.
`clearCell` must mark the cell invalid. `requestCapture()` without an argument
must request all cells. Scene invalidation, elapsed time, and construction with
`source === "capturedScene"` must not request grid capture implicitly.

`IrradianceProbeGridRuntimeCache.worldToGrid3x3` must be a `Matrix3` instance.
Consumers must read its row-major values from `worldToGrid3x3.elements`.

### Localized light probes

- `LightProbe` must accept only `new LightProbe({ ...params })`.
- Empty probes must be constructed with `new LightProbe({})`.
- Authored SH coefficients must be passed as `new LightProbe({ sh })`.
- `LightProbe.sh` must expose mutable probe-owned storage. Assignment of the SH
  array, replacement of one coefficient, or mutation of a coefficient component
  must invalidate scene lighting when the probe is attached to a scene.
- `LightProbe` must not expose `color` or `intensity`; SH coefficients must
  carry the probe radiance scale directly.
- `LightProbe.shape` must support `"global"`, `"sphere"`, and `"box"`.
- `LightProbe.shape` must default to `"global"`.
- `LightProbe.radius` must be finite and must be sanitized to a positive value.
- `LightProbe.halfExtents` must be finite and each component must be sanitized
  to a positive value.
- `LightProbe.blendDistance` must be finite and must be sanitized to `>= 0`.
- `LightProbe.priority` must be finite and must be sanitized to an integer.
- `LightProbe.copy()` and `LightProbe.clone()` must preserve SH coefficients and
  localized probe properties.
- `LightProbe.getMetric(worldPosition)` must return the normalized box or sphere
  distance at the supplied world-space position and positive infinity for a
  global probe.
- `LightProbeRuntimeCache.worldToProbe3x3` must be a `Matrix3` instance.
  Consumers must read its row-major values from `worldToProbe3x3.elements`.
- Localized `LightProbe` selection must evaluate only probes in the highest
  active `priority` group.
- Higher numeric `priority` values must win over lower numeric `priority`
  values.
- Within the winning `priority` group, rendering backends must normalize the
  top two active probe weights and must blend only those two probes.
- Weight tie-breaks must be deterministic and must fall back to `LightProbe.id`.
- `blendDistance` must use normalized metric fade semantics consistent with
  `ReflectionProbe`, including an effective minimum floor derived from probe
  size.
- WebGPU and WebGL must treat `shape="global"` probes as contributors to the
  global SH buffer and must evaluate localized probes per-fragment.
- The Software backend must remain compatible by treating localized probes as
  global SH contributors.
- WebGPU and WebGL must clamp localized probe collection to `8` probes per
  frame.

### Captured light probes

- `LightProbe.source` must support `"environment"`, `"capturedScene"`, and
  `"manual"`.
- `LightProbe.source` must default to `"environment"`.
- `LightProbe.captureResolution` must default to `64x32`.
- `LightProbe.captureFar` must default to `200`.
- `LightProbe.includeEnvironment`, `includeMeshes`, `includeTransparent`,
  `includeParticles`, and `includeShadows` must default to `true`.
- `LightProbe.requestCapture()` must increment `captureRequestToken` and
  `captureRevision` and must invalidate scene lighting so an on-demand renderer
  executes the requested capture.
- `ProbeCaptureRuntime` must capture `LightProbe` instances only when
  `source === "capturedScene"` and `requestCapture()` has requested a newer
  capture generation.
- Construction, scene invalidation, and elapsed time must not request a
  `LightProbe` capture implicitly.
- Repeated requests before execution may coalesce to the newest capture
  generation.
- `ProbeCaptureRuntime` must write captured low-frequency radiance to
  `LightProbe.sh`.
- `ProbeCaptureRuntime` must invalidate captured SH writes with a
  non-capture-relevant dirty reason such as `probe-capture`; runtime capture
  writeback must not invalidate an in-flight capture generation by itself.
- `ProbeCaptureRuntime` must store radiance SH coefficients and must not store
  pre-convolved irradiance coefficients.
- `ProbeCaptureRuntime` must share one capture between `LightProbe` and
  `ReflectionProbe` targets when capture origin, `captureFar`, and include flags
  match.
- Environment IBL warmup and runtime update must mutate only `LightProbe`
  instances with `source === "environment"`.

### Reflection probes

`ReflectionProbe` is a specialized light source representing an influence volume where local specular reflections are captured or applied.

#### Configuration Parameters
When initializing a `ReflectionProbe` or passing options to its constructor `ReflectionProbeParams`, the following fields must be supported:

- `shape`: Specifies the volume boundary. Must be `"sphere"` or `"box"`. Defaults to `"sphere"`.
- `radius`: Sphere boundary radius. Must be a positive number greater than or equal to `1e-6`. Defaults to `5`.
- `halfExtents`: Box boundary dimensions. Must be an `IVector3`. Defaults to `{ x: 5, y: 5, z: 5 }`.
- `blendDistance`: Interpolation transition distance at the volume edge. Must be a non-negative number. Defaults to `0.15`.
- `blendExponent`: Exponent scaling the volume interpolation curve. Must be a positive number.
- `parallaxMode`: The parallax correction technique. Must be `"off"`, `"box"`, or `"sphere"`. Defaults to `"box"` if `shape` is `"box"`, otherwise `"off"`.
- `prefilteredMap`: A pre-baked specular `Texture` to override captured results. Defaults to `null`.
- `source`: The environment input source. Must be `"environment"` (analytical/sky only), `"capturedScene"` (dynamic scene capture), or `"manual"`. Defaults to `"environment"`.
- `captureResolution`: Dimension configuration of the target map. Must be a `Partial<ReflectionProbeCaptureResolution>` mapping `width` and `height`. Defaults to `{ width: 512, height: 256 }`.
- `captureFar`: Far clipping distance used for capture cameras. Defaults to `200`.
- `includeEnvironment`: When `true`, captures environment background maps. Defaults to `true`.
- `includeMeshes`: When `true`, includes scene geometry/meshes in capture. Defaults to `true`.
- `includeTransparent`: When `true`, renders transparent geometry during scene capture. Defaults to `true`.
- `includeParticles`: When `true`, renders active particle simulations during scene capture. Defaults to `true`.
- `includeShadows`: When `true`, integrates direct shadow maps during scene capture. Defaults to `true`.

#### Capture & Placement Rules
- **Capture Origin**:
  - If a `ReflectionProbe` is parented under a non-root scene `Node`, the capture origin must resolve from the parent's world position, while the probe's local transform continues to define the influence volume and parallax proxy.
  - If a `ReflectionProbe` is attached directly to the scene root (e.g. `scene.add(probe)`), the capture origin must resolve from the probe's own world position.
- **Budgeting & Performance**:
  - The runtime capture executor must prioritize requested probes nearest to the active camera position first.
  - The frame budget for probe capture updates must default to `4ms`. If a task exceeds this limit, the capture resolution must temporarily scale down using steps `1.0 -> 0.75 -> 0.5`.
  - An unfinished capture must invalidate the next frame with the non-capture-relevant `probe-capture` reason so on-demand renderers continue the task.
- **Explicit Requests**:
  - `ReflectionProbe.requestCapture()` must increment `captureRequestToken` and `captureRevision` and must invalidate the attached scene so an on-demand renderer executes the request.
  - `ProbeCaptureRuntime` must execute a reflection probe capture only when `source === "capturedScene"` and a newer explicit request generation exists.
  - Construction, scene invalidation, and elapsed time must not request a reflection probe capture implicitly.
  - Clone operations must not copy pending capture requests for reflection probes, light probes, or irradiance grid cells.
- **Recursion Prevention**:
  - During a probe capture render pass, the features `enableReflection` and `enableSSR` must be forced to `false` to avoid feedback loops.
  - Shadow mapping must reuse the main frame's shadow maps; a dedicated shadow map render pass must not be triggered for the probe.
- **WebGPU Lifecycle**:
  - `WebGPUReflectionProbeCapturePass` must not resolve a `ComputeRuntime`
    while `WebGPUBackend` is still initializing.
  - The capture readback runtime must be created lazily on the first capture,
    after the backend compute facade exposes its initialized device and queue.
  - Scene capture render targets must use the shared WebGPU legacy MRT color
    attachment formats so capture render passes remain compatible with scene
    pipelines.
- **Capture Sampling**:
  - Cubemap faces must use the order `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`.
  - Cubemap-to-equirectangular conversion and final shader sampling must use
    `phi = u * 2 * PI - PI` and `theta = v * PI`.
  - Float-backed HDR capture input must preserve radiance values above `1`.
  - Prefiltered equirectangular textures must repeat horizontally and clamp
    vertically.
  - A multi-probe atlas must isolate linear filtering between probes. The
    runtime must preserve horizontal longitude wrapping inside each layer and
    must not let filtering cross into an adjacent probe layer.

#### Backend Support Matrix
The following table outlines features and fallback behaviors across backends:

| Feature / Capability                              | WebGPU Backend                           | WebGL Backend                            | Software Backend                         |
| :------------------------------------------------ | :--------------------------------------- | :--------------------------------------- | :--------------------------------------- |
| **Capture Interface**                             | Supported via `PROBE_CAPTURE_EXTENSION`  | Fallback only (No extension registered)  | Fallback only (No extension registered)  |
| **Scene Mesh Capture (`includeMeshes`)**          | Fully Supported (renders real geometry)  | Not Supported (falls back to analytical) | Not Supported (falls back to analytical) |
| **Transparent Geometry (`includeTransparent`)**   | Fully Supported (renders alpha passes)   | Not Supported (falls back to analytical) | Not Supported (falls back to analytical) |
| **Particle Simulation (`includeParticles`)**      | Fully Supported (renders active systems) | Not Supported (falls back to analytical) | Not Supported (falls back to analytical) |
| **Shadow Reuse (`includeShadows`)**               | Fully Supported (reuses depth bounds)    | Not Supported (falls back to analytical) | Not Supported (falls back to analytical) |
| **Environment Background (`includeEnvironment`)** | Fully Supported                          | Fully Supported (CPU-side fallback)      | Fully Supported (CPU-side fallback)      |
| **Analytical Lights**                             | Fully Supported (if fallback active)     | Fully Supported (CPU-side fallback)      | Fully Supported (CPU-side fallback)      |
| **Fallback CPU Rasterization**                    | Yes (applied if GPU capture fails)       | Yes (used exclusively for capture)       | Yes (used exclusively for capture)       |

- **Analytical Fallback CPU Rasterization**: When mesh-based capture is unavailable (WebGL, Software) or fails, the engine falls back to CPU analytical approximation. The CPU fallback computes local irradiance/radiance by accumulating ambient lighting, environment background (if configured), and direct/analytical light sources (directional, point, spot, area lights) mapped to lobes.

## Usage

### Clustered lighting

#### TypeScript Setup
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

#### WebGL GLSL Shader Integration

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

#### WebGPU WGSL Shader Integration

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

### Environment IBL

```ts
const prefilter = new IBLPrefilter({ backend: renderBackend });
const prefilteredMap = await prefilter.prefilter(environmentTexture, {
	acceleration: "auto",
	maxSampleWidth: 128,
	maxSampleHeight: 64,
	maxMipLevels: 5,
});

reflectionProbe.prefilteredMap = prefilteredMap;
reflectionProbe.markRuntimeDirty();
```

```ts
const sh = projectEnvironmentTextureToSH(environmentTexture, {
	maxSampleWidth: 128,
	maxSampleHeight: 64,
});

const prefilteredMap = await prefilterEnvironmentIBL(environmentTexture, {
	service: { backend: webgpuBackend },
	acceleration: "auto",
	maxSampleWidth: 128,
	maxSampleHeight: 64,
	maxMipLevels: 5,
});

lightProbe.sh = sh;
reflectionProbe.prefilteredMap = prefilteredMap;
```

```bash
bun tests/static/lighting/test_ibl_prefilter.mjs
```

### Irradiance probe grids

```ts
import { IrradianceProbeGrid } from "../src/lights";
import { SH } from "../src/maths/SH";

const grid = scene.add(new IrradianceProbeGrid({
	dimensions: { x: 4, y: 2, z: 4 },
	halfExtents: { x: 8, y: 3, z: 8 },
	source: "capturedScene",
	captureResolution: { width: 64, height: 32 },
}));

grid.position.set(0, 2, 0);
grid.requestCapture();

const authored = SH.empty();
authored[0] = { r: 12, g: 10, b: 8 };
grid.setCellSH({ x: 0, y: 0, z: 0 }, authored);
```

### Localized light probes

```ts
import { LightProbe } from "../src/lights/LightProbe";
import { SH } from "../src/maths/SH";

const hallwayProbe = new LightProbe({
	shape: "box",
	halfExtents: { x: 4, y: 3, z: 6 },
	blendDistance: 0.2,
	priority: 20,
	sh: SH.empty(),
});

const courtyardProbe = new LightProbe({
	shape: "sphere",
	radius: 12,
	blendDistance: 0.35,
	priority: 5,
	sh: SH.empty(),
});

const fallbackProbe = new LightProbe({ sh: SH.empty() });
fallbackProbe.shape = "global";

const hallwayMetric = hallwayProbe.getMetric({ x: 1, y: 0, z: 0 });
```

```bash
bun tests/static/lighting/test_light_probe_runtime.mjs
```

### Captured light probes

```ts
import { LightProbe } from "../src/lights/LightProbe";

const probe = new LightProbe({
	source: "capturedScene",
	shape: "box",
	halfExtents: { x: 4, y: 3, z: 6 },
	captureResolution: { width: 64, height: 32 },
	includeEnvironment: true,
	includeMeshes: true,
});

probe.requestCapture();
```

```bash
bun tests/static/lighting/test_probe_capture_runtime.mjs
```

### Reflection probes

#### Reflection Probe Initialization
```ts
import { ReflectionProbe } from "../src/lights/ReflectionProbe";

// Configure a reflection probe for explicitly requested scene captures
const probe = new ReflectionProbe({
	source: "capturedScene",
	captureResolution: { width: 512, height: 256 },
	captureFar: 200,
	includeEnvironment: true,
	includeMeshes: true,
	includeTransparent: true,
	includeParticles: true,
	includeShadows: true,
});

// Explicitly request each desired update
probe.requestCapture();
```

#### Probe Attachment Options

##### Parented to a Model Node
```ts
// Capture originates from model world position
// Probe local position offsets the volume for blend alignment
model.addChild(probe);
probe.position.set(0, 1.5, 0);
```

##### Attached Directly to the Scene
```ts
// Capture originates from probe world position
// Probe position translates both the capture point and volume together
scene.add(probe);
probe.position.set(10, 0, -5);
```

## Diagnostics

### Clustered lighting

- `webgpu-point-limit`: Emitted once per logger lifecycle when WebGPU forward
  shading skips point lights beyond `MAX_POINT_LIGHTS`. Repeated frames that
  remain over budget must not emit the warning again.
- `webgl-clustered-fragment-light-budget` / `webgpu-clustered-light-budget`: Emitted once if the active scene light count exceeds the configured `maxLights` budget.
- `webgl-clustered-perspective-only` / `webgpu-clustered-perspective-only`: Emitted once when clustered lighting is enabled on a non-perspective camera.
- `webgl-clustered-invalid-depth-range`: Emitted once on the WebGL backend if the camera near/far depth range is negative, zero, or too small for log slicing.
- `webgpu-clustered-max-lights-limit`: Emitted once on the WebGPU backend if the requested `maxLights` exceeds `1024`.
- `webgpu-clustered-max-per-cluster-limit`: Emitted once on the WebGPU backend if the requested `maxLightsPerCluster` exceeds `128`.
- `webgl-clustered-texture-size-overflow`: Emitted once on WebGL if the required cluster count or light data exceeds the maximum texture size supported by the GPU.

### Environment IBL

- `IBLPrefilter.prefilter()` must throw when `envMap` is not a valid 2D
  equirectangular texture or cubemap.
- `IBLPrefilter.prefilter()` must throw an `AbortError` when `signal` is
  aborted.
- Explicit `webgpu` acceleration must throw when the service backend does not
  expose an accepting `WEBGPU_COMPUTE_EXTENSION`.
- Explicit `webgl` acceleration must throw when the backend is uninitialized,
  lacks required extensions, or fails framebuffer or readback validation. A
  request submitted while the context is lost must wait for restoration or
  cancellation.
- Explicit `multi-thread` acceleration must throw when the Worker API is
  unavailable.

### Irradiance probe grids

`RangeError` is thrown when `dimensions.x * dimensions.y * dimensions.z > 256`.

`RangeError` is thrown when `getCellIndex`, `getCellSH`, `setCellSH`,
`clearCell`, or `requestCapture` receives a cell outside the grid.

WebGPU and WebGL emit a warning when more than one grid is active in a frame;
only the selected grid is used. WebGL compiles grid sampling only when the
scene shader can fit the optional grid sampler. It binds the SH texture to unit
`15` and emits a warning when that sampler path is unavailable.

If a captured-scene grid requests mesh capture without a compatible WebGPU face
capture source, capture falls back to analytic lights and environment only.

### Localized light probes

- Non-finite `radius`, `halfExtents`, `blendDistance`, or `priority` values
  will be sanitized during construction and runtime cache refresh.
- If more than `8` localized probes are available, WebGPU and WebGL will select
  the camera-relevant subset for the current frame.
- If a localized probe appears inactive at a fragment, its weight will resolve
  to `0` and it will not participate in blending.
- If no localized probe is active at a fragment, WebGPU and WebGL will fall
  back to the global SH buffer.
- If no global SH data is available, localized probe coverage may blend against
  zero SH outside probe influence regions.

### Captured light probes

- If `source === "manual"`, `ProbeCaptureRuntime` must not overwrite
  `LightProbe.sh`.
- If `source === "environment"`, environment IBL update may overwrite
  `LightProbe.sh`.
- If mesh capture is requested without a compatible GPU face capture source,
  runtime must emit `probe-mesh-capture-unsupported` and use analytical
  fallback capture.
- If a probe changes transform, source, include flags, or capture request token
  while a capture is in flight, stale results must not overwrite that probe.

### Reflection probes

- **`[probe-mesh-capture-unsupported]`**: This warning is logged when a non-WebGPU backend (WebGL or Software) is active and `includeMeshes` is set to `true`. This indicates that the backend is falling back to environment background and analytical lights only.
- **Low-Resolution Reflections**: If captured textures appear lower than the configured resolution, verify whether runtime budget pressure is forcing the scheduler to downscale resolution to `0.75` or `0.5`.
- **Render Loop Instability**: If reflections show recursive feedback artifacts, verify that `enableReflection` and `enableSSR` are correctly set to `false` during the capture pass.
- **Incorrect Projection Origin**: If reflections appear projected from the coordinate origin `[0, 0, 0]`, confirm that `scene.updateWorldMatrices()` has been executed prior to dispatching the capture stage.

## Verification

```bash
bun run test:lighting
bun run test:pointspot
bun run test:sh
bunx tsc --noEmit
```

## Related Documents

- [Rendering architecture](../architecture/rendering.md)
- [Shadow contract](shadows.md)
- [Materials contract](materials.md)
- [Migration guidance](../migrations/README.md)
