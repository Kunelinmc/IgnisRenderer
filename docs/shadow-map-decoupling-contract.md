# Shadow Map Decoupling Contract

## Scope
This document defines the shadow mapping contract, including configuration, decoupling from light sources, and how capabilities are queried from the session's profile.

## Background
Historically, lights held shadow configuration directly. The decoupled model moves shadow management to `Scene` and relies on the attached backend's `RenderBackendProfile.shadow` to check what capabilities the active backend supports.

## API/Contract
- `RenderBackendProfile.shadow`
	- Must contain shadow capabilities and budgets for the active backend.
	- `backendKey`: unique backend identifier string.
	- `supportsFilterModes`: array of supported filter modes (e.g. `"pcf"`, `"vsm"`).
	- `supportsDirectionalCSM`: boolean indicating directional cascade support.
	- `supportsSpotCSM`: boolean indicating spot cascade support.
	- `supportsPointCSM`: boolean indicating point/cube cascade support.
	- `maxDynamicShadowCost`: numerical budget for dynamic shadow calculations.
	- `supportsPagedShadows`: optional boolean indicating paged shadow scheduling support.
	- `supportsPagedShadowRendering`: optional boolean indicating complete paged
	  shadow rendering support.
	- `maxPagedShadowPages`: optional physical page budget for paged shadows.
	- `pagedShadowPageSizeRange`: optional supported page-size range.
- `scene.shadows`
	- Must be the single entry point for binding and managing shadow maps.
	- `create(kind, options)` must create a shadow map through the configured `ShadowMapRegistry`.
	- `createSingle(options)`, `createVariance(options)`, `createCascaded(options)`, `createPaged(options)`: must remain compatibility helpers for the built-in shadow map kinds.
	- `registerMapType(kind, factory)` must register user-defined shadow map factories on the manager's registry.
	- `bind(light, shadowMap)`, `unbindLight(light)`: bind or unbind a shadow map to a light.
	- `destroy(shadowMap)`: destroy a shadow map and clear its bindings.
- `ShadowMapRegistry`
	- Must map each `ShadowMapKind` string to a factory that returns a `ShadowMapBase`.
	- Must support built-in kinds `single`, `variance`, `cascaded`, and `paged-shadow` through `createDefaultShadowMapRegistry()`.
	- May register external custom kinds using any non-empty string.
	- Must throw when `create(kind, options)` is called for an unregistered kind.
- `ShadowMapBase`
	- Must expose `kind` as the stable map kind recorded in `ShadowBindingRecord.shadowMapKind`.
	- Must expose `resolveCascadeCount(lightType)` for shadow budget and render-set sizing.
	- Custom maps that emit a CSM-compatible `ShadowConfig` should override `resolveCascadeCount(lightType)`.

## Usage
```ts
import {
	Scene,
	DirectionalLight,
	ShadowMapBase,
	ShadowMapRegistry,
} from "../src";

class ExternalCascadeShadowMap extends ShadowMapBase {
	public readonly kind = "external-cascade";

	public override resolveCascadeCount(): number {
		return 3;
	}

	public override toLegacyShadowConfig(_lightType, overrides = {}) {
		return this.createCSMLegacyConfig(overrides.cascadeCount ?? 3, {
			size: overrides.size,
			lambda: 0.5,
			blendRatio: 0.2,
			stabilize: true,
		});
	}
}

const registry = new ShadowMapRegistry().register(
	"external-cascade",
	(options) => new ExternalCascadeShadowMap(options)
);
const scene = new Scene({ shadows: { registry } });
const light = scene.add(new DirectionalLight({ intensity: 1.2 }));

// Query backend shadow capability first
const profile = renderer.backendProfile;
if (profile.shadow.supportsDirectionalCSM) {
	const shadowMap = scene.shadows.create("external-cascade", { size: 2048 });
	scene.shadows.bind(light, shadowMap);
}
```

## Errors & Diagnostics
- If a CSM shadow map is requested on a backend that does not support CSM cascades, the backend must emit a fallback warning.
- Binding lights that are not present in the scene graph will result in zero active shadow cascades at frame build time.
- Calling `scene.shadows.create(kind, options)` with an unregistered `kind` must throw an error naming the missing kind.
- Registering an empty shadow map kind must throw an error.

## Compatibility / Breaking Changes
- Lights no longer expose `castShadow`, `shadow`, or strategy mutators.
- All query of shadow capabilities must go through `renderer.backendProfile.shadow`.
- Built-in helper methods remain source-compatible, but shadow map creation is now registry-backed.
