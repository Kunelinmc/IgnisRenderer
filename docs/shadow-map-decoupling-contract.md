# Shadow Map Decoupling Contract

## Scope
This document defines the shadow mapping contract, including configuration, decoupling from light sources, and how capabilities are queried from the session's profile.

## Background
Historically, lights held shadow configuration directly. The decoupled model moves shadow management to `Scene` and relies on the session's `RenderBackendProfile.shadow` to check what capabilities the active backend session supports.

## API/Contract
- `RenderBackendProfile.shadow`
	- Must contain shadow capabilities and budgets for the active backend.
	- `backendKey`: unique backend identifier string.
	- `supportsFilterModes`: array of supported filter modes (e.g. `"pcf"`, `"vsm"`).
	- `supportsDirectionalCSM`: boolean indicating directional cascade support.
	- `supportsSpotCSM`: boolean indicating spot cascade support.
	- `supportsPointCSM`: boolean indicating point/cube cascade support.
	- `maxDynamicShadowCost`: numerical budget for dynamic shadow calculations.
- `scene.shadows`
	- Must be the single entry point for binding and managing shadow maps.
	- `createSingle(options)`, `createVSM(options)`, `createCSM(options)`: create corresponding shadow maps.
	- `bind(light, shadowMap)`, `unbindLight(light)`: bind or unbind a shadow map to a light.
	- `destroy(shadowMap)`: destroy a shadow map and clear its bindings.

## Usage
```ts
import { Scene, DirectionalLight, CSMShadowMap } from "../src";

const scene = new Scene();
const light = scene.add(new DirectionalLight({ intensity: 1.2 }));

// Query backend shadow capability first
const profile = renderer.backendProfile;
if (profile.shadow.supportsDirectionalCSM) {
	const shadowMap = scene.shadows.createCSM({ size: 2048 });
	scene.shadows.bind(light, shadowMap);
}
```

## Errors & Diagnostics
- If a CSM shadow map is requested on a backend that does not support CSM cascades, the backend must emit a fallback warning.
- Binding lights that are not present in the scene graph will result in zero active shadow cascades at frame build time.

## Compatibility / Breaking Changes
- Lights no longer expose `castShadow`, `shadow`, or strategy mutators.
- All query of shadow capabilities must go through `renderer.backendProfile.shadow`.
