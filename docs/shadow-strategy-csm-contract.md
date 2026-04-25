# Shadow Strategy CSM Contract
## Scope
This document defines the cross-backend shadow strategy contract introduced by `ShadowConfig`, including `single-map` and `csm` behavior, fallback rules, and backend capability constraints.

## Background
The previous shadow path used light-bound camera builders. The new path moves shadow behavior to a strategy contract so renderers can share one API while selecting backend-specific execution.

## API/Contract
The following contracts must be respected:

- Shadow generation must be gated by `Light.castShadow` as the primary switch.
- `Light.shadow` must be configured by `ShadowConfig`.
- `ShadowConfig` must support `strategy: "single-map"` and `strategy: "csm"`.
- `DirectionalLight` and `SpotLight` may use `csm`.
- `SpotLight` with `csm` must remain single-map equivalent in current forward shading paths (`cascadeCount = 1` runtime behavior).
- `PointLight` and `RectArea` must resolve to single-map behavior in v1 and must not be hard-disabled by light type checks when `castShadow` is `true`.
- `csm` strategy fields:
  - `cascadeCount` must be one of `2 | 3 | 4`.
  - `splitMode` must be `"practical"` in v1.
  - `lambda` must be clamped to `[0, 1]`.
  - `maxDistance` should default to camera far when omitted.
  - `blendRatio` must be clamped to `[0, 1]`.
  - `stabilize` may be disabled, but should default to enabled.
  - `priority` may be used for CSM budget arbitration.
- Prepared frame shadow data must use `ShadowRenderSet` with:
  - `requestedStrategyType`
  - `effectiveStrategyType`
  - `resolvedConfig`
  - `slices[]` (`ShadowSlice`)
- `ShadowSlice` must include:
  - `shadowMap`
  - `splitNear`
  - `splitFar`
  - `atlasRect`
- Backends must expose capability-based behavior:
  - WebGPU must support directional CSM in v1.
  - WebGL and Software must fallback to single-map equivalent results when CSM is requested.
- Fallback warnings must be emitted with a stable key that includes backend and light identity. Callers should de-duplicate by warning key.

## Usage
Example: configure a directional light with CSM and allow backend fallback.

```ts
import { DirectionalLight } from "../src/lights/DirectionalLight";

const sun = new DirectionalLight({
	intensity: 2.0,
	direction: { x: -0.3, y: -1.0, z: -0.2 },
});
sun.castShadow = true;
sun.shadow = {
	strategy: "csm",
	size: 2048,
	cascadeCount: 4,
	splitMode: "practical",
	lambda: 0.65,
	maxDistance: 120,
	blendRatio: 0.1,
	stabilize: true,
	priority: 10,
};
```

Example: enable shadow generation on a point light through `castShadow`.

```ts
import { PointLight } from "../src/lights/PointLight";

const point = new PointLight({
	range: 60,
	castShadow: true,
});
```

Verification command:

```bash
bun tests/test_shadow_strategy_csm.mjs
```

## Errors & Diagnostics
- `shadow-strategy-fallback-<backend>-<lightId>`: backend does not execute requested CSM and uses single-map fallback.
- `webgl-directional-light-limit`: directional light count exceeded backend limit.
- `webgpu-directional-light-limit`: directional light count exceeded backend limit.

Common triggers:
- Requesting more CSM directional lights than backend budget allows.
- Supplying out-of-range `lambda` or `blendRatio` values (values are clamped).

## Compatibility / Breaking Changes
- Breaking change: `Light.shadow` no longer exposes `setupShadowCamera`; it now stores `ShadowConfig`.
- Existing scenes must migrate shadow configuration to strategy form.
- Legacy runtime maps using plain `ShadowMap` may still be consumed by compatibility helpers, but new integrations should use `ShadowRenderSet`.
