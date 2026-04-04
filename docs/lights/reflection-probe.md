# ReflectionProbe

## Scope

This document covers `ReflectionProbe` — IgnisRenderer's local specular IBL probe — including its shape/parallax model, runtime cache lifecycle, and the pipeline helpers in `src/pipeline/reflectionProbeRuntime.ts`. For the diffuse counterpart see [light-probe.md](./light-probe.md). For shared base contracts see [overview.md](./overview.md).

---

## Background

`ReflectionProbe` captures a pre-filtered, mip-mapped **equirectangular texture** that represents the environment's specular radiance. During shading, backends blend up to two overlapping probes to produce smooth transitions, apply **parallax correction** to redirect the reflection ray so it appears to originate from the correct position in the probe volume, and sample the pre-filtered map at a mip level corresponding to surface roughness.

The runtime pipeline caps active probes at `MAX_ACTIVE_REFLECTION_PROBES = 8` per frame. Probes without a valid `prefilteredMap` are excluded from selection.

Two capture volume shapes are supported:
- **Sphere** — influence falls off by distance from probe origin.
- **Box (AABB)** — influence falls off by Chebyshev distance from probe center in local space, aligned to the node's orientation.

---

## API/Contract

### `ReflectionProbe`

**File**: `src/lights/ReflectionProbe.ts`  
**`type`**: `LightType.ReflectionProbe` (`"reflectionProbe"`)

```ts
interface ReflectionProbeParams extends LightParams {
    shape?:         ReflectionProbeShape;
    radius?:        number;
    halfExtents?:   IVector3;
    blendDistance?: number;
    blendExponent?: number;
    parallaxMode?:  ReflectionProbeParallaxMode;
    prefilteredMap?: Texture | null;
}

new ReflectionProbe(params?: ReflectionProbeParams)
```

#### Shape types

```ts
type ReflectionProbeShape = "sphere" | "box";
```

| Value | Influence metric | Parallax ray |
|---|---|---|
| `"sphere"` | Euclidean distance from probe origin ÷ `radius` | Ray intersects a local sphere of `radius` |
| `"box"` | Chebyshev distance from probe center ÷ `halfExtents` (per axis) | Ray intersects a local AABB of `±halfExtents` |

#### Parallax correction modes

```ts
type ReflectionProbeParallaxMode = "off" | "box" | "sphere";
```

| Value | Behavior |
|---|---|
| `"off"` | No correction. Reflection uses raw world-space direction. |
| `"box"` | Ray intersects the probe's local AABB; hit point becomes new sample direction. |
| `"sphere"` | Ray intersects the probe's local sphere; hit point becomes new sample direction. |

Default `parallaxMode` is `"box"` when `shape === "box"`, otherwise `"off"`.

#### Parameters

| Member | Type | Default | Description |
|---|---|---|---|
| `shape` | `ReflectionProbeShape` | `"sphere"` | Capture volume shape. |
| `radius` | `number` | `5` | Sphere influence radius. Clamped to ≥ `1e-6`. |
| `halfExtents` | `Vector3` | `{ x:5, y:5, z:5 }` | Box half-extents. Each axis internally clamped ≥ `1e-6` when computing `invHalfExtents`. |
| `blendDistance` | `number` | `0.15` | World-unit distance beyond probe boundary over which influence fades to zero. Effective floor is 10% of probe size. |
| `blendExponent` | `number` | `1` | Power applied to raw blend weight. `> 1` = sharper boundary, `< 1` = softer. Clamped to `≥ 0.01`. |
| `parallaxMode` | `ReflectionProbeParallaxMode` | shape-dependent | Parallax correction mode. |
| `prefilteredMap` | `Texture \| null` | `null` | Pre-filtered equirect texture with mip levels. Probes with `null` or invalid textures are excluded from active collection. |

---

### Runtime cache

`ReflectionProbe` maintains a lazy `ReflectionProbeRuntimeCache`. The cache is rebuilt automatically when `getRuntimeCache()` detects that any of the following have changed since the last rebuild: `shape`, `radius`, `halfExtents`, `blendDistance`, `blendExponent`, `parallaxMode`, or the node's world matrix.

```ts
interface ReflectionProbeRuntimeCache {
    probeToWorldMatrix:     Matrix4;   // Copy of node's world matrix at last update
    worldToProbeMatrix:     Matrix4;   // Full inverse (rotation + translation)
    worldToProbe3x3:        Matrix3Arr;// Rotation-only inverse (for direction transforms)
    probeWorldPosition:     IVector3;  // Cached world-space position
    invHalfExtents:         IVector3;  // 1 / halfExtents per axis (box probes)
    radiusInv:              number;    // 1 / radius (sphere probes)
    effectiveBlendDistance: number;    // max(blendDistance, 10% of probe size)
    blendExponent:          number;    // Sanitized exponent value
}
```

| Method | Description |
|---|---|
| `probe.getRuntimeCache()` | Returns the cache. Rebuilds lazily if dirty or if state changed. |
| `probe.markRuntimeDirty()` | Marks cache as dirty. Next `getRuntimeCache()` call will rebuild. Call after programmatic mutations to probe properties. |
| `probe.refreshRuntimeCache()` | Forces an immediate synchronous rebuild. |

---

### Pipeline helpers

**File**: `src/pipeline/reflectionProbeRuntime.ts`

These functions are consumed by backends. Application code should not need to call them directly.

| Function | Signature | Description |
|---|---|---|
| `collectActiveReflectionProbes` | `(lights, maxCount?) → ReflectionProbe[]` | Filters `SceneLight[]` to probes with valid textures. Sorted by `probe.id`, capped at 8. |
| `collectReflectionProbeEnvironment` | `(lights, maxCount?) → { probes, atlas }` | Returns active probes + a horizontally-stitched atlas `Texture`. |
| `refreshReflectionProbeCaches` | `(lights) → void` | Forces cache rebuild on all probes in the list. |
| `selectTopTwoReflectionProbes` | `(worldPos, probes) → { firstIndex, secondIndex, firstWeight, secondWeight }` | Selects the top two probes by influence weight at `worldPos`. Weights are normalized to sum to 1. |
| `computeProbeMetric` | `(worldPos, probe) → number` | Returns normalized boundary distance: 0 = probe center, 1 = exactly at boundary. > 1 = outside. |
| `computeProbeRawWeight` | `(metric, effectiveBlendDistance, blendExponent) → number` | Converts metric to a [0, 1] weight via smoothstep + exponent. Returns 0 for out-of-range points. |
| `computeParallaxCorrectedDirection` | `(worldPos, reflDir, probe) → { direction, valid }` | Returns parallax-corrected reflection direction. `valid = false` if correction failed or mode is `"off"`. |
| `sampleReflectionProbesSpecular` | `(worldPos, reflDir, roughness, probes, fallback) → RGB \| null` | CPU path: blends specular IBL at `roughness` from top two probes. Returns `null` if no probes qualify and no fallback. |
| `buildReflectionProbeAtlasTexture` | `(probes) → Texture \| null` | Stitches probe maps horizontally (one probe per column-strip, all mip levels). Cached by probe ID + texture version. |
| `isTextureReadyForEnvironment` | `(texture) → boolean` | Returns `true` if texture has valid dimensions, data, and is not a load-error fallback. |
| `directionToEquirectUV` | `(direction) → { u, v }` | Converts a normalized world-space direction to equirectangular UV coordinates. |

**Atlas cache behavior**:
- `buildReflectionProbeAtlasTexture` maintains a module-level LRU cache keyed by probe IDs + texture versions.
- The cache is cleared when it exceeds 32 entries.
- All probes in the input array must share the same base texture dimensions and mip count; a mismatch returns `null`.

**Probe selection algorithm**:
1. For each probe, compute a `metric` (normalized boundary distance).
2. Convert to raw weight via smoothstep and `blendExponent`.
3. Retain top two probes with weights > `REFLECTION_PROBE_WEIGHT_EPSILON` (`1e-6`).
4. Ties in weight are broken by `probe.id` lexicographic order for determinism.
5. Weights are normalized to sum to 1 before return.

---

## Usage

### Adding a sphere reflection probe

```ts
import { ReflectionProbe } from "../lights";
import { Texture } from "../core/Texture";

// envMap: a pre-filtered equirectangular Texture with mip chain
const probe = new ReflectionProbe({
    shape: "sphere",
    radius: 8,
    blendDistance: 2,
    prefilteredMap: envMap,
});
probe.position.set(0, 1.5, 0);
scene.add(probe);
```

### Adding a box probe with parallax correction

```ts
const roomProbe = new ReflectionProbe({
    shape: "box",
    halfExtents: { x: 10, y: 3, z: 6 },
    blendDistance: 1.0,
    blendExponent: 2,
    parallaxMode: "box",
    prefilteredMap: roomEnvMap,
});
roomProbe.position.set(0, 3, 0);
scene.add(roomProbe);
```

### Updating a probe after moving it

```ts
// After modifying position/rotation programmatically:
roomProbe.position.set(5, 3, 0);
scene.updateWorldMatrices();

// getRuntimeCache() will detect the matrix change automatically,
// but you can also force an immediate rebuild:
roomProbe.markRuntimeDirty();
const cache = roomProbe.getRuntimeCache();
```

### Checking probe validity before use

```ts
import { isTextureReadyForEnvironment } from "../pipeline/reflectionProbeRuntime";

if (isTextureReadyForEnvironment(probe.prefilteredMap)) {
    // Safe to include probe in active collection
}
```

### CPU specular sampling (SoftwareBackend)

```ts
import {
    collectActiveReflectionProbes,
    sampleReflectionProbesSpecular,
} from "../pipeline/reflectionProbeRuntime";

const probes = collectActiveReflectionProbes(scene.getLights());
const specular = sampleReflectionProbesSpecular(
    surfaceWorldPos,    // { x, y, z }
    reflectionDir,      // Normalized reflection vector
    roughness,          // [0, 1]
    probes,
    scene.skybox,       // Fallback texture (or null)
);
// specular: { r, g, b } in linear [0, 1], or null if no probe/fallback
```

---

## Errors & Diagnostics

| Symptom | Likely cause |
|---|---|
| Probe not appearing in specular | `prefilteredMap` is `null`, lacks data, or `isLoadErrorFallback` is `true`. Call `isTextureReadyForEnvironment(probe.prefilteredMap)`. |
| Probe influence not updating after move | `markRuntimeDirty()` was not called, or `scene.updateWorldMatrices()` was not run. The dirty check compares the stored matrix signature. |
| Hard seam between two adjacent probes | `blendDistance` is too small on one or both probes. Increase to create an overlap region. |
| Parallax correction artifacts | Geometry is outside the probe volume. Verify `halfExtents` or `radius` encompasses receiving surfaces. |
| Atlas building returns `null` | Probes have mismatched texture dimensions or mip counts. All active probes must share the same equirect resolution and mip chain length. |
| Over 8 probes in scene but only 8 active | `MAX_ACTIVE_REFLECTION_PROBES = 8` is a hard cap. Probes are sorted by `probe.id` and truncated. |
| Reflection direction looks wrong | `parallaxMode` is `"off"` while `shape` is `"box"`. Set `parallaxMode: "box"` explicitly. |
| CPU specular returns `null` | No probes have valid textures and `fallbackTexture` is `null`. Provide a skybox or at least one valid probe. |

---

## Compatibility / Breaking Changes

- **Runtime cache is not rebuilt on direct property mutation**: Assignment to `probe.shape`, `probe.radius`, `probe.halfExtents`, etc., does not automatically mark the cache dirty. Always call `probe.markRuntimeDirty()` after mutating these fields, or rely on the matrix-change detection in `getRuntimeCache()` if only the transform changed.
- **Atlas cache is module-scoped**: `buildReflectionProbeAtlasTexture` stores its cache in module-level state. In test environments that require isolation, the atlas cache may carry over between test cases. There is no public API to flush it; it self-clears when it exceeds 32 entries.
- **`blendExponent` minimum**: Values below `0.01` are silently clamped. Passing `0` does not disable the exponent — it produces `blendExponent = 0.01`.
- **`prefilteredMap` ownership**: `ReflectionProbe` holds a reference to the texture object. Mutating `Texture.version` externally without calling `markRuntimeDirty()` will not invalidate the atlas cache key and may result in stale cached atlases.
