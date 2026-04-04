# Shadow Mapping

## Scope

This document covers the shadow infrastructure shared across all shadow-capable lights in IgnisRenderer: the `ShadowMap` data container, `ShadowParams` bias tuning, and the `ShadowCaster` / `ShadowCameraResult` contracts. Runtime shadow buffer management and PCF sampling are implemented inside each rendering backend and are not covered here.

---

## Background

Shadow mapping in IgnisRenderer is split across two concerns:

1. **Definition layer** (`src/lights/`): Each shadow-capable light holds a `ShadowCaster` instance that knows how to compute a shadow camera matrix for that light's geometry. This is backend-agnostic.
2. **Runtime layer** (per-backend): Each backend allocates a depth buffer, renders the shadow pass using the matrices from `setupShadowCamera`, and samples those buffers during shading.

The `ShadowMap` class in `src/lights/ShadowMapping.ts` is the shared handshake object passed from the definition layer to the backend.

Lights that support shadows:

| Light type | Default `castShadow` | Shadow camera type |
|---|---|---|
| `DirectionalLight` | `true` | Orthographic, scene-radius-sized |
| `SpotLight` | `true` | Perspective, cone-FOV |
| `AreaLight` | `true` | Perspective, 120° fixed FOV |
| `PointLight` | —  | Not supported |
| `AmbientLight` | — | Not supported |
| `LightProbe` | — | Not applicable |
| `ReflectionProbe` | — | Not applicable |

---

## API/Contract

### `ShadowCaster` interface

**File**: `src/lights/Light.ts`

```ts
interface ShadowCaster {
    setupShadowCamera(ctx: {
        sceneBounds: { center: IVector3; radius: number };
        worldMatrix: Matrix4;
    }): ShadowCameraResult | null;
}

interface ShadowCameraResult {
    view:       Matrix4;   // Light-space view matrix
    projection: Matrix4;   // Shadow projection matrix
    lightDir:   IVector3;  // Normalized world-space light direction
}
```

- `setupShadowCamera` must return `null` if the shadow cannot be computed (e.g., degenerate scene bounds).
- `worldMatrix` is the light node's current world transform, updated by `scene.updateWorldMatrices()` before each render frame.
- `sceneBounds` is typically obtained from `scene.getBounds()`.

---

### `ShadowMap` class

**File**: `src/lights/ShadowMapping.ts`

Stateful container for per-light shadow map data. Constructed by backends; not intended for direct construction by application code.

```ts
new ShadowMap(size?: number, params?: ShadowParams)
```

| Member | Type | Description |
|---|---|---|
| `size` | `number` | Shadow map resolution in pixels (default: `1024`). |
| `params` | `ShadowParams` | Bias and quality parameters (see below). |
| `viewMatrix` | `Matrix4 \| null` | Last computed light-space view matrix. |
| `projectionMatrix` | `Matrix4 \| null` | Last computed projection matrix. |
| `viewProjectionMatrix` | `Matrix4 \| null` | Combined VP for shadow comparison. |
| `latestLightDir` | `IVector3` | Last reported world-space light direction. |
| `stabilizedBoundsRadius` | `number \| null` | Cached scene-bounds radius for stable frustum fitting. |

---

### `ShadowParams`

All fields are optional. Missing fields use the defaults below.

| Parameter | Default | Description |
|---|---|---|
| `shadowBias` | `0.0005` | Constant depth offset added to shadow map depth. Reduces self-shadowing (acne) at the cost of potential peter-panning. |
| `shadowSlopeBias` | `0.01` | Additional bias proportional to surface slope relative to light direction. Helps on grazing-angle surfaces. |
| `shadowNormalBias` | `0.1` | Pushes the shadow receiver along its surface normal before comparison. Reduces acne on curved surfaces. |
| `shadowNormalBiasMin` | `0.01` | Minimum clamp on the normal bias. |
| `shadowTexelBias` | `1.0` | Bias expressed in shadow-map texel units. Accounts for resolution-dependent acne. |
| `shadowMaxBias` | `0.01` | Maximum total bias clamp (prevents extreme offsets on very steep surfaces). |
| `shadowPCF` | `1` | PCF (Percentage-Closer Filtering) kernel radius in texels. `1` = 3×3 kernel, `2` = 5×5, etc. |
| `shadowStrength` | `1` | Shadow blend factor. `0` = fully transparent shadow, `1` = fully opaque. |

---

### Shadow constants

**File**: `src/lights/constants.ts`

| Constant | Value | Description |
|---|---|---|
| `MIN_SHADOW_NEAR` | `0.01` | Minimum shadow camera near plane (used by `SpotLight`, `AreaLight`). |
| `MIN_SHADOW_FAR` | `0.02` | Minimum shadow camera far plane. |
| `SHADOW_NEAR_FAR_GAP` | `0.01` | Enforced minimum gap between near and far planes. |

---

## Usage

### Disabling shadows on a specific light

```ts
import { DirectionalLight } from "../lights";

const sun = new DirectionalLight({ intensity: 3 });
sun.castShadow = false; // Override default (DirectionalLight defaults to true)
scene.add(sun);
```

### Checking if a light casts shadows

```ts
import { isShadowCastingLight } from "../lights";

for (const light of scene.getLights()) {
    if (isShadowCastingLight(light)) {
        // light.shadow is a ShadowCaster
        const result = light.shadow.setupShadowCamera({
            sceneBounds: scene.getBounds(),
            worldMatrix: light.worldMatrix,
        });
    }
}
```

### Configuring shadow quality (via backend)

The backend constructs `ShadowMap` objects. To override defaults, pass a custom `ShadowParams` when configuring the backend or after construction:

```ts
import { ShadowMap } from "../lights/ShadowMapping";

// Example: high-resolution, softened shadows
const shadowMap = new ShadowMap(2048, {
    shadowBias: 0.001,
    shadowSlopeBias: 0.02,
    shadowNormalBias: 0.05,
    shadowPCF: 2,
    shadowStrength: 0.9,
});
```

---

## Errors & Diagnostics

| Symptom | Likely cause |
|---|---|
| Shadow acne (dark stripes on lit surfaces) | `shadowBias` too low. Increase `shadowBias` or add `shadowSlopeBias`. |
| Peter-panning (shadow floats off caster) | `shadowBias` or `shadowNormalBias` too high. Reduce incrementally. |
| Shadows appear very faint | `shadowStrength` is < 1. Set to `1.0` for full opacity. |
| Hard shadow edges despite `shadowPCF > 1` | Backend-specific: verify that PCF sampling is wired to `params.shadowPCF`. |
| `setupShadowCamera` returns `null` | Scene bounds are degenerate (no visible meshes). Ensure meshes are in `scene.getMeshInstances()` before rendering. |
| Near-plane clip cuts away shadow casters close to light | `MIN_SHADOW_NEAR` is `0.01`. Do not place casters closer than 0.01 world units to the light. |

---

## Compatibility / Breaking Changes

- **`ShadowMap` is not public API**: Application code must not construct `ShadowMap` directly for use with backends. Use the backend's own configuration API.
- **`shadowSamples` / `shadowSearchSamples`**: These optional fields exist on `ShadowParams` for backend-specific advanced sampling (PCSS, etc.) and have no engine-default. Their interpretation is backend-defined.
