# PointLight

## Scope

This document covers the `PointLight` analytical light type in IgnisRenderer. For shared base contracts see [overview.md](./overview.md).

## Background

`PointLight` represents an omnidirectional point emitter with inverse-square attenuation within a finite radius. It is commonly used for light sources such as lamps, torches, or explosions.

It extends `Light<TType>` and is registered in `ECSWorld` automatically when added to a `Scene`.

## API/Contract

**File**: `src/lights/PointLight.ts`  
**`type`**: `LightType.Point` (`"point"`)

Omnidirectional emitter. Attenuation follows a physics-based inverse-square model, reaching zero at `range`.

```ts
interface PointLightParams extends LightParams {
    position?: IVector3;
    range?: number;
}

new PointLight(params?: PointLightParams)
```

| Member | Type | Default | Description |
|---|---|---|---|
| `range` | `number` | `1000` | Maximum attenuation radius in world units. |
| `color` | `RGB` | `{ r:255, g:255, b:255 }` | Emission color (linear-space). |
| `intensity` | `number` | `1` | Output multiplier. |

> [!NOTE]
> `PointLight` does **not** support shadow casting in the current core renderer version. `castShadow` must remain `false`. No `ShadowCaster` is attached.

**Position**: Set via the inherited `Node.position` property or by passing `params.position`.

## Usage

```ts
import { Scene } from "../core/Scene";
import { PointLight } from "../lights";

const scene = new Scene();

// Ceiling lamp
const lamp = new PointLight({
    color: { r: 255, g: 200, b: 140 },
    intensity: 4,
    position: { x: 0, y: 4, z: 0 },
    range: 15,
});
scene.add(lamp);
```

## Errors & Diagnostics

| Symptom | Likely cause |
|---|---|
| Illuminates nothing beyond close range | `range` is too small. Default is 1000; reduce only when intentional. |

## Compatibility / Breaking Changes

N/A
