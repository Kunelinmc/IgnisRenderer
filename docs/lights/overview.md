# Lights — Overview

## Scope

This document is the entry point for IgnisRenderer's lighting system. It describes the shared base abstractions, type contracts, and scene-graph integration. Each concrete light type is documented in its own file in this folder:

| File | Coverage |
|---|---|
| [overview.md](./overview.md) | This document — base `Light`, `LightType`, `SceneLight`, shadow contracts |
| [ambient-light.md](./ambient-light.md) | `AmbientLight` analytical light |
| [directional-light.md](./directional-light.md) | `DirectionalLight` parallel emitter |
| [point-light.md](./point-light.md) | `PointLight` omnidirectional emitter |
| [spot-area.md](./spot-area.md) | `SpotLight`, `AreaLight` |
| [shadow-mapping.md](./shadow-mapping.md) | `ShadowMap`, `ShadowParams`, bias tuning |
| [light-probe.md](./light-probe.md) | `LightProbe`, SH coefficients, `SH` math utilities |
| [reflection-probe.md](./reflection-probe.md) | `ReflectionProbe`, parallax correction, runtime cache, pipeline helpers |

---

## Background

All lights extend the abstract `Light<TType>` class, which itself extends `Node`. Lights are full scene-graph citizens: they support hierarchical transforms, world-matrix propagation, and are registered in `ECSWorld` automatically when added to a `Scene`.

The pipeline collects lights each frame via `scene.getLights()` (backed by `ECSWorld.findLights()`), then dispatches the resulting `SceneLight[]` to the active backend — SoftwareBackend, WebGPUBackend, or WebGLBackend.

The coordinate system is **Right-Handed** (+Y up, −Z forward). All direction vectors in light definitions must conform to this convention.

---

## API/Contract

### Base class: `Light<TType>`

**File**: `src/lights/Light.ts`

Abstract base. All concrete light types extend this.

| Member | Type | Default | Description |
|---|---|---|---|
| `type` | `LightType` | — | Discriminant enum. Readonly. |
| `color` | `RGB` | `{ r:255, g:255, b:255 }` | Linear-space RGB. Channels are in [0, 255]; shaders decode to linear. |
| `intensity` | `number` | `1` | Linear output multiplier. |
| `castShadow` | `boolean` | `false` | Enables shadow map generation. |
| `shadow` | `ShadowCaster \| undefined` | — | Assigned automatically by subclasses that support shadows. |

**`LightParams`** (constructor input):

```ts
interface LightParams extends NodeParams {
    color?: RGB;
    intensity?: number;
    castShadow?: boolean;
}
```

---

### `LightType` enum

```ts
enum LightType {
    Ambient         = "ambient",
    Directional     = "directional",
    Point           = "point",
    Spot            = "spot",
    LightProbe      = "lightProbe",
    ReflectionProbe = "reflectionProbe",
    RectArea        = "rectArea",   // used by AreaLight
}
```

---

### `SceneLight` union

Exported from `src/lights/index.ts`. All backends accept this type.

```ts
type SceneLight =
    | AmbientLight
    | DirectionalLight
    | PointLight
    | SpotLight
    | LightProbe
    | ReflectionProbe
    | AreaLight;
```

---

### Shadow contracts

**`ShadowCaster`** interface (implemented internally by each shadow-capable light):

```ts
interface ShadowCaster {
    setupShadowCamera(ctx: {
        sceneBounds: { center: IVector3; radius: number };
        worldMatrix: Matrix4;
    }): ShadowCameraResult | null;
}

interface ShadowCameraResult {
    view:       Matrix4;
    projection: Matrix4;
    lightDir:   IVector3;
}
```

**`ShadowCastingLight`** type-guard:

```ts
type ShadowCastingLight = SceneLight & { shadow: ShadowCaster };

function isShadowCastingLight(light: SceneLight): light is ShadowCastingLight;
```

---

### Scene integration

```ts
// Add any light to the scene
scene.add(light);

// Query all active lights (used by backends)
const lights: SceneLight[] = scene.getLights();
```

---

## Usage

```ts
import { Scene } from "../core/Scene";
import {
    AmbientLight, DirectionalLight, PointLight,
    SpotLight, AreaLight, LightProbe, ReflectionProbe,
    isShadowCastingLight,
} from "../lights";

const scene = new Scene();

const sun = new DirectionalLight({
    color: { r: 255, g: 240, b: 220 },
    intensity: 3,
    direction: { x: -0.5, y: -1, z: -0.5 },
});
scene.add(sun);

// Type-narrowing via LightType discriminant
for (const light of scene.getLights()) {
    if (isShadowCastingLight(light)) {
        const result = light.shadow.setupShadowCamera({
            sceneBounds: scene.getBounds(),
            worldMatrix: light.worldMatrix,
        });
    }
}
```

---

## Errors & Diagnostics

| Symptom | Likely cause |
|---|---|
| Light has no effect after `scene.add()` | World matrices not yet updated. Call `scene.updateWorldMatrices()` before rendering. |
| `scene.getLights()` returns empty array | Light was added to a detached sub-tree, not to `scene.root` or its descendants. |
| `isShadowCastingLight` returns `false` unexpectedly | `castShadow` is `false` or `shadow` was not assigned by the subclass (e.g., `PointLight`). |

---

## Compatibility / Breaking Changes

- **`LightType.RectArea`**: `AreaLight` uses the enum value `"rectArea"`, not `"areaLight"`. Discriminant switches must match this string exactly.
- **`Node` deprecation**: `Light` extends the deprecated `Node` ECS facade. Prefer `scene.getLights()` over manual tree traversal to collect lights.
