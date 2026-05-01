# Shadow Map Decoupling Contract
## Scope
This document defines the new shadow architecture where `Light` and shadow state are fully decoupled and shadows are managed by `Scene`.

## Background
The previous model stored shadow configuration directly in `Light` (`castShadow` and `shadow`), which coupled rendering policy to light entities and made multi-backend orchestration harder.

## API/Contract
- `Scene` must expose `scene.shadows` as the single entrypoint for shadow lifecycle.
- `scene.shadows.createSingle(options)` must create a `SingleShadowMap`.
- `scene.shadows.createVSM(options)` must create a `VSMShadowMap`.
- `scene.shadows.createCSM(options)` must create a `CSMShadowMap`.
- `scene.shadows.bind(light, shadowMap)` must bind one shadow map to one light.
- `scene.shadows.rebind(light, shadowMap)` must replace the current binding for that light.
- `scene.shadows.unbindLight(light)` must remove the binding for that light.
- `scene.shadows.destroy(shadowMap)` must remove the shadow map and all light bindings that reference it.
- `Light` must not expose `castShadow`, `shadow`, `setShadowStrategy`, `setSingleMapShadow`, or `setCSMShadow`.
- `CSMShadowMap` must support all light categories in this release:
	- Directional cascades.
	- Spot cascades.
	- Point cube cascades (`cascadeCount * 6` slices).
- `VSMShadowMap` may preserve VSM parameters while runtime sampling should fallback to PCF in v1.

## Usage
```ts
import { Scene, DirectionalLight, CSMShadowMap } from "../src/index";

const scene = new Scene();
const light = scene.add(new DirectionalLight({ intensity: 1.2 }));
const shadowMap = scene.shadows.createCSM({
	size: 2048,
	priority: 3,
	cascadeCounts: { directional: 4, spot: 3, point: 2 },
	lambda: 0.65,
	blendRatio: 0.1,
	stabilize: true,
});

scene.shadows.bind(light, shadowMap);
```

```bash
bunx tsc --noEmit
```

## Errors & Diagnostics
- Binding a light that is not in the active scene graph may result in zero active shadow slices at frame build time.
- Destroying a bound `ShadowMapBase` will invalidate all associated light bindings by contract.
- Backends may emit fallback warnings when `CSM` is requested but backend-specific support is unavailable.

## Compatibility / Breaking Changes
- Breaking: all direct light shadow APIs are removed (`castShadow`, `shadow`, and shadow strategy mutators).
- All callers must migrate to `scene.shadows.*` binding APIs.
