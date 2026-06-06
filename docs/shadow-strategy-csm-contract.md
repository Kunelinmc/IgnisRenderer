# Shadow Strategy CSM Contract
## Scope
This document defines the cross-backend `CSM` shadow contract in the decoupled `scene.shadows` model.

## Background
The previous model bound shadow controls to `Light` (`castShadow` and `light.shadow`). The current model decouples shadows into `ShadowMap` objects managed by `Scene`, and all backends consume unified slice descriptors from `ShadowRenderSet`.

## API/Contract
- Shadow generation must be enabled only by explicit binding through `scene.shadows.bind(light, shadowMap)`.
- `Light` must not carry any shadow strategy fields.
- `scene.shadows.createCSM(options)` must produce `ShadowConfig` with `strategy: "csm"` when consumed as legacy config.
- `CSM` must support `DirectionalLight`, `SpotLight`, and `PointLight`.
- Cascade defaults must be `4/3/2` for `directional/spot/point`.
- Directional `CSM` must use practical camera-frustum splits.
- When `stabilize` is `true`, directional `CSM` must retain a per-cascade
  light-space center across metadata updates, move the retained center only
  after the current frustum center leaves the cascade safety region, and snap
  the retained center to the shadow texel grid.
- Spot `CSM` must use cone-depth practical splits.
- Point `CSM` must use full cube cascades with total slice count `cascadeCount * 6`.
- `cascadeCount` must be normalized to `1..4`.
- `lambda` and `blendRatio` must be clamped to `[0, 1]`.
- `ShadowRenderSet` must expose:
  - `requestedStrategyType`
  - `effectiveStrategyType`
  - `resolvedConfig`
  - `slices[]`
- Dynamic budget selection must rank shadows by `priority`, then `light.intensity`, then camera relation score.
- Dynamic degradation order must be: reduce cascade count, then reduce resolution, then disable low-score shadows.
- `VSMShadowMap` must keep `filterMode: "vsm"` metadata and preserve `shadowMomentBias`, `shadowBleedReduction`, and `shadowMinVariance` in `ShadowConfig.params`, while runtime filtering should fallback to PCF in v1.

## Usage
Example: configure and bind directional CSM through `Scene`.

```ts
import { Scene, DirectionalLight } from "../src";

const scene = new Scene();
const sun = new DirectionalLight({
	intensity: 2.0,
	direction: { x: -0.3, y: -1.0, z: -0.2 },
});
scene.add(sun);

const csm = scene.shadows.createCSM({
	size: 2048,
	priority: 10,
	cascadeCounts: { directional: 4, spot: 3, point: 2 },
	lambda: 0.65,
	maxDistance: 120,
	blendRatio: 0.1,
	stabilize: true,
});
scene.shadows.bind(sun, csm);
```

Verification commands:

```bash
bunx tsc --noEmit
bun tests/static/lighting/test_shadow_strategy_csm.mjs
```

## Errors & Diagnostics
- `shadow-strategy-fallback-<backend>-<lightId>` may be emitted when backend capability policy demotes `CSM`.
- Excessive requested shadow cost may disable low-priority bindings in a frame.
- Invalid `lambda`, `blendRatio`, or `cascadeCount` values are normalized before runtime metadata build.

## Compatibility / Breaking Changes
- Breaking: `castShadow`, `light.shadow`, and light shadow mutator APIs are removed.
- Existing scenes must migrate to explicit `scene.shadows.create*` and `scene.shadows.bind` calls.
- Legacy helpers that consume `ShadowConfig` and `ShadowRenderSet` remain valid for backend internals.
