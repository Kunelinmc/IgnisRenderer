# Reflection Probe Contract

## Scope
This document defines the contract, configuration options, and runtime execution behavior for the `ReflectionProbe` subsystem. It specifies the feature capabilities and fallback mechanisms across the WebGPU, WebGL, and Software rendering backends.

## Background
Real-time reflection probes capture dynamic scene environments into cube textures to support specular reflections and image-based lighting (IBL). While hardware-accelerated backends such as WebGPU support full per-face scene rendering (including meshes, particles, transparent passes, and shadows), backends without full support or GPU-accelerated capture resources must fallback to a CPU-based analytical and environment background approximation.

## API/Contract
`ReflectionProbe` is a specialized light source representing an influence volume where local specular reflections are captured or applied.

### Configuration Parameters
When initializing a `ReflectionProbe` or passing options to its constructor `ReflectionProbeParams`, the following fields must be supported:

- `shape`: Specifies the volume boundary. Must be `"sphere"` or `"box"`. Defaults to `"sphere"`.
- `radius`: Sphere boundary radius. Must be a positive number greater than or equal to `1e-6`. Defaults to `5`.
- `halfExtents`: Box boundary dimensions. Must be an `IVector3`. Defaults to `{ x: 5, y: 5, z: 5 }`.
- `blendDistance`: Interpolation transition distance at the volume edge. Must be a non-negative number. Defaults to `0.15`.
- `blendExponent`: Exponent scaling the volume interpolation curve. Must be a positive number.
- `parallaxMode`: The parallax correction technique. Must be `"off"`, `"box"`, or `"sphere"`. Defaults to `"box"` if `shape` is `"box"`, otherwise `"off"`.
- `prefilteredMap`: A pre-baked specular `Texture` to override captured results. Defaults to `null`.
- `source`: The environment input source. Must be `"environment"` (analytical/sky only), `"capturedScene"` (dynamic scene capture), or `"manual"`. Defaults to `"environment"`.
- `captureUpdateMode`: Scheduling frequency for `"capturedScene"` source. Must be `"manual"`, `"onSceneDirty"`, or `"interval"`. Defaults to `"onSceneDirty"`.
- `captureIntervalSeconds`: Interval duration in seconds when update mode is `"interval"`. Defaults to `1`.
- `captureResolution`: Dimension configuration of the target map. Must be a `Partial<ReflectionProbeCaptureResolution>` mapping `width` and `height`. Defaults to `{ width: 512, height: 256 }`.
- `captureFar`: Far clipping distance used for capture cameras. Defaults to `200`.
- `includeEnvironment`: When `true`, captures environment background maps. Defaults to `true`.
- `includeMeshes`: When `true`, includes scene geometry/meshes in capture. Defaults to `true`.
- `includeTransparent`: When `true`, renders transparent geometry during scene capture. Defaults to `true`.
- `includeParticles`: When `true`, renders active particle simulations during scene capture. Defaults to `true`.
- `includeShadows`: When `true`, integrates direct shadow maps during scene capture. Defaults to `true`.

### Capture & Placement Rules
- **Capture Origin**:
  - If a `ReflectionProbe` is parented under a non-root scene `Node`, the capture origin must resolve from the parent's world position, while the probe's local transform continues to define the influence volume and parallax proxy.
  - If a `ReflectionProbe` is attached directly to the scene root (e.g. `scene.add(probe)`), the capture origin must resolve from the probe's own world position.
- **Budgeting & Performance**:
  - The runtime capture scheduler must prioritize nearest probes first relative to the active camera position.
  - The frame budget for probe capture updates must default to `4ms`. If a task exceeds this limit, the capture resolution must temporarily scale down using steps `1.0 -> 0.75 -> 0.5`.
- **Recursion Prevention**:
  - During a probe capture render pass, the features `enableReflection` and `enableSSR` must be forced to `false` to avoid feedback loops.
  - Shadow mapping must reuse the main frame's shadow maps; a dedicated shadow map render pass must not be triggered for the probe.

### Backend Support Matrix
The following table outlines features and fallback behaviors across backends:

| Feature / Capability | WebGPU Backend | WebGL Backend | Software Backend |
| :--- | :--- | :--- | :--- |
| **Capture Interface** | Supported via `PROBE_CAPTURE_EXTENSION` | Fallback only (No extension registered) | Fallback only (No extension registered) |
| **Scene Mesh Capture (`includeMeshes`)** | Fully Supported (renders real geometry) | Not Supported (falls back to analytical) | Not Supported (falls back to analytical) |
| **Transparent Geometry (`includeTransparent`)** | Fully Supported (renders alpha passes) | Not Supported (falls back to analytical) | Not Supported (falls back to analytical) |
| **Particle Simulation (`includeParticles`)** | Fully Supported (renders active systems) | Not Supported (falls back to analytical) | Not Supported (falls back to analytical) |
| **Shadow Reuse (`includeShadows`)** | Fully Supported (reuses depth bounds) | Not Supported (falls back to analytical) | Not Supported (falls back to analytical) |
| **Environment Background (`includeEnvironment`)** | Fully Supported | Fully Supported (CPU-side fallback) | Fully Supported (CPU-side fallback) |
| **Analytical Lights** | Fully Supported (if fallback active) | Fully Supported (CPU-side fallback) | Fully Supported (CPU-side fallback) |
| **Fallback CPU Rasterization** | Yes (applied if GPU capture fails) | Yes (used exclusively for capture) | Yes (used exclusively for capture) |

- **Analytical Fallback CPU Rasterization**: When mesh-based capture is unavailable (WebGL, Software) or fails, the engine falls back to CPU analytical approximation. The CPU fallback computes local irradiance/radiance by accumulating ambient lighting, environment background (if configured), and direct/analytical light sources (directional, point, spot, area lights) mapped to lobes.

## Usage

### Reflection Probe Initialization
```ts
import { ReflectionProbe } from "../src/lights/ReflectionProbe";

// Configure a reflection probe with manual scene capture features
const probe = new ReflectionProbe({
	source: "capturedScene",
	captureUpdateMode: "manual",
	captureResolution: { width: 512, height: 256 },
	captureFar: 200,
	includeEnvironment: true,
	includeMeshes: true,
	includeTransparent: true,
	includeParticles: true,
	includeShadows: true,
});

// Explicitly trigger a capture request when using manual mode
probe.requestCapture();
```

### Probe Attachment Options

#### Parented to a Model Node
```ts
// Capture originates from model world position
// Probe local position offsets the volume for blend alignment
model.addChild(probe);
probe.position.set(0, 1.5, 0);
```

#### Attached Directly to the Scene
```ts
// Capture originates from probe world position
// Probe position translates both the capture point and volume together
scene.add(probe);
probe.position.set(10, 0, -5);
```

## Errors & Diagnostics
- **`[probe-mesh-capture-unsupported]`**: This warning is logged when a non-WebGPU backend (WebGL or Software) is active and `includeMeshes` is set to `true`. This indicates that the backend is falling back to environment background and analytical lights only.
- **Low-Resolution Reflections**: If captured textures appear lower than the configured resolution, verify whether runtime budget pressure is forcing the scheduler to downscale resolution to `0.75` or `0.5`.
- **Render Loop Instability**: If reflections show recursive feedback artifacts, verify that `enableReflection` and `enableSSR` are correctly set to `false` during the capture pass.
- **Incorrect Projection Origin**: If reflections appear projected from the coordinate origin `[0, 0, 0]`, confirm that `scene.updateWorldMatrices()` has been executed prior to dispatching the capture stage.

## Compatibility / Breaking Changes
- **Cross-Backend Fallbacks**: Mesh, transparent, particle, and shadow capture paths are active only under the WebGPU backend v1. Non-WebGPU backends (WebGL and Software) execute CPU analytical and environment fallback calculations.
- **Default Capture Resolution**: The default `captureResolution` is established at `512x256`.
- **Default Feature Flags**: The new options `includeMeshes`, `includeTransparent`, `includeParticles`, and `includeShadows` default to `true`.
- **Projection Matrix Structuring**: `ReflectionProbeRuntimeCache.worldToProbe3x3` is a `Matrix3` instance. Downstream shaders or binders reading raw values must retrieve row-major arrays from `worldToProbe3x3.elements`.
- **Pipeline Stage Renaming**: The legacy renderer stage `reflection-probe-capture` has been unified under `probe-capture`.
- **Obsolete Documentation**: The document `docs/reflection-probe-captured-scene-webgpu-v1-contract.md` has been deprecated and replaced by this document.
