# SpotLight · AreaLight

## Scope

This document covers `SpotLight` and `AreaLight` — the two directional, finite-geometry emitters in IgnisRenderer. Both support shadow casting using perspective shadow cameras. For shared base contracts see [overview.md](./overview.md). For shadow bias tuning see [shadow-mapping.md](./shadow-mapping.md).

---

## Background

- **`SpotLight`** emits from a point in a cone shape. It is the go-to light for flashlights, headlights, and stage spotlights. Shadow coverage matches the cone frustum.
- **`AreaLight`** emits from a rectangular planar surface oriented along the node's local +Y axis. Shadow is approximated with a wide-angle perspective projection (120° FOV) because true area light penumbra requires ray tracing.

Both lights share the pattern of computing shadow cameras using `Matrix4.lookAt` and `Matrix4.perspective`.

---

## API/Contract

### `SpotLight`

**File**: `src/lights/SpotLight.ts`  
**`type`**: `LightType.Spot` (`"spot"`)

Cone-shaped emitter. `castShadow` defaults to `true`.

```ts
interface SpotLightParams extends LightParams {
    position?:   IVector3;
    direction?:  IVector3;
    outerAngle?: number;
    innerAngle?: number;
    penumbra?:   number;
    range?:      number;
}

new SpotLight(params?: SpotLightParams)
```

| Member | Type | Default | Description |
|---|---|---|---|
| `direction` | `IVector3` | `{ x:0, y:-1, z:0 }` | World-space cone axis. Must be unit-length. |
| `outerAngle` | `number` | `Math.PI / 4` | Half-angle of the outer cone in radians (full cone = `outerAngle × 2`). |
| `innerAngle` | `number \| undefined` | `undefined` | Half-angle of the inner (full-intensity) cone in radians. When `undefined`, the backend may use `outerAngle` as the inner boundary. |
| `penumbra` | `number` | `0` | Soft-edge falloff factor in [0, 1]. `0` = hard edge, `1` = maximum softness. |
| `range` | `number` | `1000` | Maximum effective radius in world units. Attenuation reaches zero at boundary. |
| `castShadow` | `boolean` | `true` | Defaults to enabled. |

**Shadow camera contract** (`SpotShadowCaster.setupShadowCamera`):

| Output field | Value |
|---|---|
| `view` | `Matrix4.lookAt(worldPosition, worldPosition + direction, up)` |
| `projection` | `Matrix4.perspective(outerAngle × 2 × (180/π), aspectRatio=1, near, far)` |
| `lightDir` | Normalized world-space direction |

- `far` = `min(range, distanceToSceneCenter + sceneRadius)`, clamped to `MIN_SHADOW_FAR`.
- `near` = `max(MIN_SHADOW_NEAR, distanceToSceneCenter − sceneRadius)`, clamped below `far − SHADOW_NEAR_FAR_GAP`.
- `up` is `{ x:0, y:1, z:0 }` unless `|direction.y| ≥ 0.999`, then `{ x:0, y:0, z:1 }`.

Shadow frustum constants (from `src/lights/constants.ts`):

| Constant | Value | Description |
|---|---|---|
| `MIN_SHADOW_NEAR` | `0.01` | Minimum shadow camera near plane. |
| `MIN_SHADOW_FAR` | `0.02` | Minimum shadow camera far plane. |
| `SHADOW_NEAR_FAR_GAP` | `0.01` | Minimum gap enforced between near and far. |

---

### `AreaLight`

**File**: `src/lights/AreaLight.ts`  
**`type`**: `LightType.RectArea` (`"rectArea"`)

Rectangular planar emitter. `castShadow` defaults to `true`. The emitting surface lies in the node's local XZ plane; light emits along the local +Y axis.

```ts
interface AreaLightParams extends LightParams {
    position?: IVector3;
    width?:    number;
    height?:   number;
    range?:    number;
}

new AreaLight(params?: AreaLightParams)
```

| Member | Type | Default | Description |
|---|---|---|---|
| `width` | `number` | `100` | Width of emitting rectangle in world units. |
| `height` | `number` | `100` | Height of emitting rectangle in world units. |
| `range` | `number` | `1000` | Maximum effective radius in world units. |
| `castShadow` | `boolean` | `true` | Defaults to enabled (overridable via `params.castShadow`). |

**Shadow camera contract** (`AreaShadowCaster.setupShadowCamera`):

| Output field | Value |
|---|---|
| `view` | `Matrix4.lookAt(areaCenter, areaCenter + localUpInWorld, up)` |
| `projection` | `Matrix4.perspective(120°, aspectRatio=1, MIN_SHADOW_NEAR, max(range, MIN_SHADOW_FAR))` |
| `lightDir` | Normalized world-space +Y direction |

> [!NOTE]
> The 120° FOV is a fixed approximation. Area light shadow penumbra is not physically accurate; for soft shadows use `shadowPCF` in [shadow-mapping.md](./shadow-mapping.md).

---

## Usage

```ts
import { Scene } from "../core/Scene";
import { SpotLight, AreaLight } from "../lights";

const scene = new Scene();

// Spotlight (stage / flashlight)
const spot = new SpotLight({
    color: { r: 255, g: 255, b: 220 },
    intensity: 6,
    position: { x: 0, y: 8, z: 0 },
    direction: { x: 0, y: -1, z: 0 },
    outerAngle: Math.PI / 6,   // 30° half-angle → 60° total cone
    innerAngle: Math.PI / 10,  // 18° full-intensity region
    penumbra: 0.3,
    range: 25,
});
scene.add(spot);

// Ceiling fluorescent panel
const panel = new AreaLight({
    color: { r: 210, g: 230, b: 255 },
    intensity: 3,
    position: { x: 0, y: 5, z: 0 },
    width: 4,
    height: 1.5,
    range: 20,
});
scene.add(panel);

// Rotate the area light so it emits sideways (toward -Z)
panel.rotation.set({ x: Math.PI / 2, y: 0, z: 0 });
```

---

## Errors & Diagnostics

| Symptom | Likely cause |
|---|---|
| `SpotLight` shadow frustum cuts geometry | `range` too small, or `outerAngle` too narrow for the scene. Increase `range`. |
| Shadow near-plane artifacts on `SpotLight` | Light is very close to casters. `MIN_SHADOW_NEAR` is `0.01`; reduce light proximity or increase `shadowBias`. |
| `AreaLight` shadow misaligned | The area light's local +Y axis does not point toward receivers. Adjust node rotation. |
| `AreaLight` shadow too sharp | Use `shadowPCF ≥ 2` in the backend's `ShadowMap` params to soften. |
| `LightType` switch fails for `AreaLight` | `AreaLight.type === "rectArea"`, not `"areaLight"`. Match against `LightType.RectArea`. |

---

## Compatibility / Breaking Changes

- **`AreaLight` enum value**: `AreaLight` is registered under `LightType.RectArea` (`"rectArea"`). Any code that discriminates on `light.type` must use this string, not `"areaLight"`.
- **`SpotLight.innerAngle`** defaults to `undefined`. Backends that assume `innerAngle === outerAngle` when `undefined` must handle this explicitly — do not assume a zero inner cone.
