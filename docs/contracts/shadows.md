# Shadow Contract

`ShadowFramePlan` is the sole cross-backend shadow planning input for a frame.
It is an immutable planner result; backends must not add allocation, residency,
atlas, or feedback state to it.

See [renderer contract](renderer.md) and [WebGPU contract](webgpu.md) for the
frame lifecycle and backend resource ownership rules.

## Authoring

Applications configure built-in definitions through `scene.shadows`.

`createSingle()`, `createVariance()`, `createCascaded()`, `createPaged()`,
`bind()`, `unbindLight()`, `destroy()`, `clear()`, and `getBoundShadowMap()`
are the supported authoring and lifecycle API. Built-in definitions keep their
observable properties and `update()` method.

`ShadowMapKind` is limited to `"single"`, `"variance"`, `"cascaded"`, and
`"paged-shadow"`. Custom registry entries and public definition subclasses are
not supported.

## Planning

`ShadowPlanner` must derive projection, technique fallback, cascade count,
resolution degradation, storage selection, and render jobs from immutable
definition snapshots, light state, camera state, caster bounds, and backend
capabilities. A plan must not change after it is published.

`PreparedShadowLight.pagedSettings` contains only logical paged settings:
`virtualResolution`, `pageSize`, `pageGridSize`, `physicalPageCount`,
`clipmapLevels`, `maxPagesPerFrame`, `cacheFrames`, and `feedbackMode`.
Page-table offsets, physical atlas dimensions, native resources, feedback, and
residency must remain backend-private.

Directional cascaded shadows must use practical camera-frustum splits. Spot
cascades must use cone-depth splits, and point cascades must produce six cube
faces per cascade. Cascade counts are clamped to the built-in range, and
unsupported projection or storage modes must produce an explicit fallback
diagnostic.

Projection stabilization history belongs to per-renderer `ShadowPlannerState`.
`FrameCoordinator` must pass that state to the static `ShadowPlanner.plan()`
entrypoint. The state must be reset when the definition revision, binding,
effective projection shape, technique, or resolution changes, and history must
be removed for inactive or disabled lights.
Caster-bound radius shrink must be smoothed while stabilization is active so a
temporary bounds reduction does not immediately resize the projection.

## Backend Runtime

Backends consume `ShadowFramePlan` plus their prepared caster and transmitter
packet collections. A backend integration may represent late concrete work as
`ShadowWorkSet`; the built-in backends currently receive the equivalent
`PreparedScene.shadowCasterPackets` and `PreparedScene.shadowTransmitterPackets`
collections. Backends own their mutable atlas slots, textures, framebuffers,
page tables, physical page residency, and binding caches. Native allocation
failure may make sampling fully lit for that frame but must not mutate the
shared plan.

Atlas caster passes must iterate only explicit `"atlas"` and
`"atlas-fallback"` jobs. Paged runtime layout is derived from prepared lights;
backend layout offsets must not be written into prepared slices or definitions.

## Breaking Migration

This contract removes `ShadowConfig`, runtime `ShadowMap`, `ShadowRenderSet`,
`ShadowFrameState`, `ShadowMapRegistry`, `LegacyShadowPlanAdapter`,
`createLegacyShadowFramePlan()`, and `resolveLegacyShadowMaps()`. There is no
compatibility adapter or transition window. Custom backend integrations must
consume `ShadowFramePlan`; application code must continue to author shadows
through `scene.shadows`.
