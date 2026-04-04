# AmbientLight

## Scope

This document covers the `AmbientLight` analytical light type in IgnisRenderer. For shared base contracts see [overview.md](./overview.md).

## Background

`AmbientLight` represents a uniform, directionless fill that applies to all surfaces equally. It is typically used to provide a base level of illumination to prevent unlit areas from being completely black.

It extends `Light<TType>` and is registered in `ECSWorld` automatically when added to a `Scene`.

## API/Contract

**File**: `src/lights/AmbientLight.ts`  
**`type`**: `LightType.Ambient` (`"ambient"`)

Emits uniform light in all directions. Has no position, direction, or geometry. Does **not** cast shadows.

```ts
new AmbientLight(params?: LightParams)
```

| Param | Type | Default | Description |
|---|---|---|---|
| `color` | `RGB` | `{ r:255, g:255, b:255 }` | Uniform emission color (linear-space). |
| `intensity` | `number` | `1` | Output multiplier. |

**Constraints**:
- `castShadow` must remain `false`. No shadow infrastructure is attached.
- `AmbientLight` contributes a flat additive term to all surfaces, independent of surface normals. For environment-aware ambient, use `LightProbe` instead.

## Usage

```ts
import { Scene } from "../core/Scene";
import { AmbientLight } from "../lights";

const scene = new Scene();

// Flat fill-light
const ambient = new AmbientLight({ 
    color: { r: 40, g: 40, b: 60 }, 
    intensity: 0.5 
});
scene.add(ambient);
```

## Errors & Diagnostics

| Symptom | Likely cause |
|---|---|
| `AmbientLight` appears flat / washed out | Use `LightProbe` for directional environment lighting instead. |

## Compatibility / Breaking Changes

N/A
