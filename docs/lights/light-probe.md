# LightProbe

## Scope

This document covers `LightProbe` — IgnisRenderer's baked diffuse irradiance probe — and the supporting `SH` (Spherical Harmonics) math utilities in `src/maths/SH.ts`. For the specular counterpart see [reflection-probe.md](./reflection-probe.md). For shared base contracts see [overview.md](./overview.md).

---

## Background

`LightProbe` captures the **diffuse irradiance** component of an environment as a compact set of third-order Spherical Harmonics (L3 SH) coefficients. Rather than storing a full 360° image, SH represents low-frequency lighting as 16 RGB coefficients — enough to reconstruct smooth ambient shading from any surface normal with a single dot-product convolution.

This approach is a standard technique in real-time rendering ("Ambient Dice", "Precomputed Radiance Transfer") and is used by the engine's ambient path in both SoftwareBackend and WebGPUBackend.

Key constants (from `src/lights/constants.ts`):

| Constant | Value | Purpose |
|---|---|---|
| `BAKED_LIGHT_PROBE_SH_SCALE` | `1 / Math.PI` | Applied to baked coefficients to normalize against the Lambertian cosine integral, preventing π-fold over-brightening. |
| `PBR_AMBIENT_FALLBACK_LINEAR` | `0.05` | Fallback ambient level used when no `LightProbe` is present in the scene. |

---

## API/Contract

### `LightProbe`

**File**: `src/lights/LightProbe.ts`  
**`type`**: `LightType.LightProbe` (`"lightProbe"`)

```ts
new LightProbe(sh?: SHCoefficients | null, intensity?: number)
```

| Member | Type | Default | Description |
|---|---|---|---|
| `sh` | `SHCoefficients` | `SH.empty()` | 16 RGB SH coefficients. Initialized to all-zero when `null` is passed. |
| `intensity` | `number` | `1.0` | Scale multiplier applied to reconstructed irradiance. |

**`SHCoefficients`** type (`src/maths/types.ts`):

```ts
// Tuple of 16 RGB values, one per SH band coefficient
type SHCoefficients = [
    { r: number; g: number; b: number },  // Y00 (L=0)
    { r: number; g: number; b: number },  // Y1-1 (L=1)
    // ... 14 more entries up to Y33 (L=3)
];
```

**Methods**:

| Method | Signature | Description |
|---|---|---|
| `copy` | `copy(source: LightProbe \| SHCoefficients): this` | Mutates `this.sh` in-place from another probe or raw coefficients. Also copies `intensity` when source is a `LightProbe`. Returns `this`. |

---

### `SH` utility class

**File**: `src/maths/SH.ts`

Static utility class for all Spherical Harmonics operations. Uses **L=3 SH basis** (16 coefficients, bands 0–3) in a **+Y-up coordinate system**.

| Method | Signature | Description |
|---|---|---|
| `evalBasis` | `(n: IVector3) → number[]` | Computes 16 SH basis values for a normalized direction vector. |
| `projectDirectionalLight` | `(dir: IVector3, color: RGB) → SHCoefficients` | Projects a directional light source into SH by multiplying basis values with the color. |
| `calculateIrradiance` | `(n: IVector3, coeffs: SHCoefficients) → RGB` | Reconstructs irradiance at surface normal `n` using Lambertian cosine convolution. Output is clamped to ≥ 0 per channel. |
| `addCoeffs` | `(a: SHCoefficients, b: SHCoefficients) → SHCoefficients` | Pairwise addition of two SH coefficient arrays. |
| `empty` | `(orderSH?: number) → SHCoefficients` | Returns all-zero coefficients. Default order is 3 → 16 coefficients. |
| `serialize` | `(coeffs: SHCoefficients) → number[]` | Flattens to a `number[]` in r, g, b interleaved order (length = 48). |
| `deserialize` | `(flat: number[]) → SHCoefficients` | Reconstructs `SHCoefficients` from a flat `number[]`. |

**Lambertian convolution factors** used in `calculateIrradiance`:

| Band | Factor |
|---|---|
| L=0 (1 coefficient) | `π` |
| L=1 (3 coefficients) | `2π / 3` |
| L=2 (5 coefficients) | `π / 4` |
| L=3 (7 coefficients) | `0` (odd bands > 1 have zero Lambertian response) |

**Constraints**:
- Input direction `n` to `evalBasis` and `calculateIrradiance` must be normalized.
- `calculateIrradiance` clamps to 16 coefficients regardless of array length mismatch; extra coefficients in `coeffs` are silently ignored.

---

## Usage

### Adding a simple sky probe

```ts
import { Scene } from "../core/Scene";
import { LightProbe } from "../lights";
import { SH } from "../maths/SH";

const scene = new Scene();

// Project a sky hemisphere into SH
const skyDir   = { x: 0, y: 1, z: 0 };
const skyColor = { r: 120, g: 150, b: 220 }; // mid-blue sky (linear-space)
const sh = SH.projectDirectionalLight(skyDir, skyColor);

const probe = new LightProbe(sh, 1.0);
scene.add(probe);
```

### Reconstructing irradiance at a surface

```ts
const surfaceNormal = { x: 0, y: 1, z: 0 };
const irradiance = SH.calculateIrradiance(surfaceNormal, probe.sh);
// irradiance.r / g / b are linear-space values
```

### Loading baked SH from external data

```ts
// External bakers (e.g. Lys, cmftStudio) export 48 floats (16 coeff × RGB)
const flatCoeffs: number[] = loadFromAsset("env_sh.json");
const sh = SH.deserialize(flatCoeffs);

// Scale to avoid π-fold over-brightening in the ambient path
import { BAKED_LIGHT_PROBE_SH_SCALE } from "../lights/constants";
const scaledSH = sh.map(c => ({
    r: c.r * BAKED_LIGHT_PROBE_SH_SCALE,
    g: c.g * BAKED_LIGHT_PROBE_SH_SCALE,
    b: c.b * BAKED_LIGHT_PROBE_SH_SCALE,
})) as SHCoefficients;

const probe = new LightProbe(scaledSH, 1.0);
scene.add(probe);
```

### Blending two probes

```ts
import { SH } from "../maths/SH";

const blended = SH.addCoeffs(probeA.sh, probeB.sh);
// Divide by 2 for average blending
const averaged = blended.map(c => ({ r: c.r / 2, g: c.g / 2, b: c.b / 2 }));
const result = new LightProbe(averaged as SHCoefficients, 1.0);
```

### Copying one probe into another (zero allocation)

```ts
// In-place update (avoids creating a new LightProbe)
probeA.copy(probeB);
```

---

## Errors & Diagnostics

| Symptom | Likely cause |
|---|---|
| Flat / uniform ambient from probe | `sh` is all-zero (`SH.empty()`). Bake or assign coefficients before rendering. |
| Over-bright ambient from baked equirect | `BAKED_LIGHT_PROBE_SH_SCALE` (`1/π`) was not applied. |
| `calculateIrradiance` returns black | `n` is a zero vector. Direction must be normalized. |
| Irradiance looks wrong on Y-axis-dominant surfaces | Basis is calibrated for +Y-up. Confirm engine coordinate convention matches the baking tool. |
| Serialized coefficients mismatch | `SH.serialize` outputs 48 floats (16 coefficients × 3 channels). Tools that output 9 or 25 coefficients use different SH orders (L=2 or L=4). |

---

## Compatibility / Breaking Changes

- **16-coefficient contract**: `SH.empty()` always returns 16 coefficients (L=3). External tools that export L=2 (9 coefficients) must be zero-padded to 16 before use with `LightProbe`.
- **`copy()` mutates in-place**: `probe.copy(source)` modifies `probe.sh` directly. If the original coefficients must be preserved, clone them first with `JSON.parse(JSON.stringify(probe.sh))`.
- **`intensity` is not applied by `calculateIrradiance`**: The `SH.calculateIrradiance` utility does not multiply by `probe.intensity`. Backends are responsible for applying the intensity multiplier during ambient accumulation.
