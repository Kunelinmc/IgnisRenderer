# Shadow Contract

This document defines shadow configuration, strategies, residency, scheduling, and transparent transmission across backends.

## Contract

### Shadow configuration

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

### Cascaded shadow maps

- Shadow generation must be enabled only by explicit binding through `scene.shadows.bind(light, shadowMap)`.
- `Light` must not carry any shadow strategy fields.
- `scene.shadows.createCascaded(options)` must produce `ShadowConfig` with `strategy: "csm"` when consumed as legacy config.
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
- `VarianceShadowMap` must keep `filterMode: "vsm"` metadata and preserve
  `shadowMomentBias`, `shadowBleedReduction`, and `shadowMinVariance` in
  `ShadowConfig.params`, while runtime filtering should fall back to PCF.

### Paged shadows

- `scene.shadows.createPaged(options)` must create a `PagedShadowMap` with the `kind` property set to `"paged-shadow"`.
- `ShadowRenderSet.storageMode` must be `"paged"` when the active rendering backend supports paged shadow rendering (i.e. `RenderBackendProfile.shadow.supportsPagedShadowRendering` is `true`).
- `ShadowRenderSet.storageMode` must be `"atlas"` when the active backend does not support paged shadow rendering.
- `ShadowRenderSet.layout.regions` must mirror the active slices of the `ShadowRenderSet`.
- `ShadowLayout.storageMode` must match `ShadowRenderSet.storageMode`.
- `RenderBackendProfile.shadow.supportsPagedShadows` may advertise backend support for paged shadow scheduling metadata.
- `RenderBackendProfile.shadow.supportsPagedShadowRendering` must be `true` before a backend may keep `PagedShadowMap` in `"paged"` rendering mode.
- The rendering backend must support directional paged shadow render sets only. Spot and point lights must use shadow atlas fallback.
- The GPU residency and dirty-page allocation (including request flags, request compaction, residency allocation, LRU metadata, and dirty physical page compaction) must execute in GPU compute passes. The CPU may upload frame-local caster bounds and issue grouped draw calls, but the GPU must be the authoritative owner of page-table allocation.
- The GPU page table and residency buffers are authoritative after initialization. CPU mirrors may exist for diagnostics but must not decide page residency.
- The paged shadow runtime must implement the following frame graph nodes:
  - The `paged-shadow-page-mark` node: Must write `pageRequestFlags`, `compactedRequests`, and request counters on the GPU.
  - The `paged-shadow-page-allocate` node: Must update `pageTable`, `residencyState`, `freeList`, `dirtyPhysicalPages`, and allocation counters on the GPU.
  - The `paged-shadow-depth` node: Must read `dirtyPhysicalPages` and render depth information into the physical depth atlas.
  - The `paged-shadow-feedback` node: Must write next-frame feedback flags after main scene depth is available, to be consumed on the subsequent frame.
- The `paged-shadow-depth` node must build paged shadow draw instance buffers and `drawIndexedIndirect` argument records on the GPU before the render pass. The CPU may bind each draw candidate's geometry and animation state, but it must not enumerate dirty pages or build per-page MVP instances.
- If feedback flags are unavailable or empty (such as on the initial frame), the backend must seed page requests from conservative caster bounds.
- CPU frame uploads may include one-frame tombstone caster bounds for removed shadow casters. The GPU must use those bounds to mark affected resident pages dirty, but tombstones must not produce depth draw instances.
- WebGPU must write physical depth pages into a `depth32float` 2D texture atlas whose side length is `ceil(sqrt(physicalPageCount)) * pageSize`.
- Paged shadow shader sampling must treat non-resident, out-of-range, or invalid page-table entries as fully lit visibility.

### Transparent transmission

- `PreparedScene.shadowTransmitterPackets` must contain transparent primitives
  with `castShadows = true`.
- PBR transparent shadow transmission must use `transmissionFactor`,
  `albedo`, `ior`, `thicknessFactor`, `attenuationColor`, and
  `attenuationDistance`.
- The transmittance model must use Beer-Lambert absorption:
  `attenuationColor ^ (thicknessFactor / attenuationDistance)`.
- `attenuationDistance <= 0`, `attenuationDistance = Infinity`, or
  `thicknessFactor <= 0` must disable volume absorption.
- PBR surface interface loss should use normal-incidence Fresnel derived from
  `ior`.
- Non-PBR and alpha-blended materials may use opacity-weighted color filtering
  as a compatibility fallback.
- Shadow visibility used by lighting must be RGB transmittance. Fully opaque
  blockers must contribute `vec3(0)`, unblocked white light must contribute
  `vec3(1)`, and transparent blockers must multiply the unblocked sample by
  material transmittance.
- Raster backends may ignore refraction and caustic focusing in this contract.
  Those effects require a separate light transport pass.

## Usage

### Shadow configuration

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

### Cascaded shadow maps

Example: configure and bind directional CSM through `Scene`.

```ts
import { Scene, DirectionalLight } from "../src";

const scene = new Scene();
const sun = new DirectionalLight({
	intensity: 2.0,
	direction: { x: -0.3, y: -1.0, z: -0.2 },
});
scene.add(sun);

const csm = scene.shadows.createCascaded({
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

### Paged shadows

```ts
import { DirectionalLight, Scene } from "../src";

const scene = new Scene();
const sun = scene.add(new DirectionalLight({ intensity: 2 }));
const shadow = scene.shadows.createPaged({
	size: 2048,
	virtualResolution: 8192,
	pageSize: 128,
	physicalPageCount: 1024,
	maxPagesPerFrame: 128,
	feedbackMode: "screen-feedback",
});

scene.shadows.bind(sun, shadow);
```

#### Verification Commands

```bash
bun tests/static/lighting/test_shadow_manager.mjs
bun tests/static/webgpu/test_webgpu_paged_shadow_runtime.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_planner.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_compiler.mjs
bun tests/static/shaders/test_shader_source.mjs
bunx tsc --noEmit
```

### Transparent transmission

```ts
import { PBRMaterial } from "../src/materials/PBRMaterial";

const redGlass = new PBRMaterial({
	albedo: { r: 255, g: 255, b: 255 },
	transmissionFactor: 1,
	ior: 1.5,
	thicknessFactor: 0.2,
	attenuationDistance: 0.4,
	attenuationColor: { r: 255, g: 32, b: 32 },
});

meshPrimitive.material = redGlass;
meshPrimitive.castShadows = true;
```

## Diagnostics

### Shadow configuration

- If a CSM shadow map is requested on a backend that does not support CSM cascades, the backend must emit a fallback warning.
- Binding lights that are not present in the scene graph will result in zero active shadow cascades at frame build time.
- Calling `scene.shadows.create(kind, options)` with an unregistered `kind` must throw an error naming the missing kind.
- Registering an empty shadow map kind must throw an error.

### Cascaded shadow maps

- `shadow-strategy-fallback-<backend>-<lightId>` may be emitted when backend capability policy demotes `CSM`.
- Excessive requested shadow cost may disable low-priority bindings in a frame.
- Invalid `lambda`, `blendRatio`, or `cascadeCount` values are normalized before runtime metadata build.

### Paged shadows

- Backends without `supportsPagedShadowRendering` must use atlas fallback metadata and must not fail a frame because a scene requested `PagedShadowMap`.
- If GPU compute resources cannot be encoded, the runtime may keep conservative CPU mirror metadata for diagnostics and fallback, but the GPU page-table buffers must remain valid resources.
- If feedback flags are missing or empty, conservative caster-bound requests must still seed residency without making the CPU the residency owner.
- If the count of dirty pages is zero, the depth pass may be skipped.
- If a physical page is evicted, the previous virtual page-table entry must be reset to `0xffffffff`.
- Debug state may report buffer capacities and last uploaded candidate counts. Resident and dirty counts must come from GPU counters or explicit readback, not from CPU-owned page maps.
- Implementations must not use `"variance"` to identify paged shadows because `"variance"` identifies `VarianceShadowMap`.

### Transparent transmission

- If the backend cannot allocate a shadow transmittance atlas, it should fall
  back to white transmittance for the frame.
- If WebGL has insufficient texture units for the transmittance atlas, it must
  keep scalar shadows enabled and sample white transmittance.
- If `transmissionFactor` is `0`, PBR materials must not produce colored
  transparent shadows unless alpha blending is active.

## Compatibility

### Shadow configuration

- Lights no longer expose `castShadow`, `shadow`, or strategy mutators.
- All query of shadow capabilities must go through `renderer.backendProfile.shadow`.
- Built-in helper methods remain source-compatible, but shadow map creation is now registry-backed.

### Cascaded shadow maps

- Breaking: `castShadow`, `light.shadow`, and light shadow mutator APIs are removed.
- Existing scenes must migrate to explicit `scene.shadows.create*` and `scene.shadows.bind` calls.
- Legacy helpers that consume `ShadowConfig` and `ShadowRenderSet` remain valid for backend internals.

### Paged shadows

N/A

### Transparent transmission

The feature is additive. Existing opaque shadow behavior must remain unchanged.

## Verification

```bash
bun run test:lighting
bun tests/static/webgpu/test_webgpu_bridge.mjs
bunx tsc --noEmit
```

## Related Documents

- [Lighting contract](lighting.md)
- [Particles contract](particles.md)
- [WebGPU contract](webgpu.md)
