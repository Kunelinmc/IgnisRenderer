# DirectionalLight

## Scope

This document covers the `DirectionalLight` analytical light type in IgnisRenderer. For shared base contracts see [overview.md](./overview.md). For shadow configuration see [shadow-mapping.md](./shadow-mapping.md).

## Background

`DirectionalLight` represents an infinite-distance parallel emitter, such as the sun or moon. It is the only type in the core engine that uses an **orthographic** shadow frustum.

It extends `Light<TType>` and is registered in `ECSWorld` automatically when added to a `Scene`.

## API/Contract

**File**: `src/lights/DirectionalLight.ts`  
**`type`**: `LightType.Directional` (`"directional"`)

Parallel infinite-distance light. `castShadow` defaults to `true`. Uses an **orthographic** shadow camera scoped to the scene bounding sphere.

```ts
interface DirectionalLightParams extends LightParams {
    direction?: IVector3;
}

new DirectionalLight(params?: DirectionalLightParams)
```

| Member | Type | Default | Description |
|---|---|---|---|
| `direction` | `IVector3` | `{ x:0, y:-1, z:0 }` | World-space light direction. Must be unit-length. |
| `color` | `RGB` | `{ r:255, g:255, b:255 }` | Emission color (linear-space). |
| `intensity` | `number` | `1` | Output multiplier. |
| `castShadow` | `boolean` | `true` | Defaults to enabled. |

**Shadow camera contract** (`DirectionalShadowCaster.setupShadowCamera`):

| Output field | Value |
|---|---|
| `view` | `Matrix4.lookAt(lightPos, sceneCenter, up)`, where `lightPos = sceneCenter − direction × (radius × 1.5)` |
| `projection` | `Matrix4.ortho(±size, ±size, 0, depth)`, where `size = radius × 1.2`, `depth = radius × 3` |
| `lightDir` | Normalized world-space direction after applying node's `worldMatrix` rotation |

`up` is `{ x:0, y:1, z:0 }` unless `|direction.y| ≥ 0.999`, in which case `{ x:0, y:0, z:1 }` is used.

## Usage

```ts
import { Scene } from "../core/Scene";
import { DirectionalLight } from "../lights";

const scene = new Scene();

// Sun
const sun = new DirectionalLight({
    color: { r: 255, g: 240, b: 200 },
    intensity: 3.5,
    direction: { x: -0.5, y: -1, z: -0.3 },
});
// Normalize direction before passing:
// sun.direction = Vector3.normalize(sun.direction);
scene.add(sun);
```

## Errors & Diagnostics

| Symptom | Likely cause |
|---|---|
| Shadow is clipped | `sceneBounds` radius is incorrect. Ensure `scene.getBounds()` includes all renderable meshes. |
| Shadow shimmers on camera move | Direction is not unit-length. Normalize `direction` before setting. |

## Compatibility / Breaking Changes

N/A
