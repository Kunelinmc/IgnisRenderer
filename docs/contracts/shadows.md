# Shadow Contract

`ShadowFramePlan` is the sole cross-backend shadow planning input for a frame.
It is an immutable planner result; backends must not add allocation, residency,
atlas, or feedback state to it.

See [renderer contract](renderer.md) and [WebGPU contract](webgpu.md) for the
frame lifecycle and backend resource ownership rules.

## Authoring

Applications configure built-in definitions through `scene.shadows`.

`createSingle()`, `createCascaded()`, `createPaged()`,
`bind()`, `unbindLight()`, `destroy()`, `clear()`, and `getBoundShadowMap()`
are the supported authoring and lifecycle API. Built-in definitions keep their
observable properties and `update()` method.

`ShadowFilterMode` is limited to `"pcf"` and `"pcss"`. Definitions select
the filter through top-level `filterMode`, select `"low"`, `"medium"`, or
`"high"` through `sampling.quality`, and control artistic attenuation through
top-level `strength`. `sampling` must not expose filter radii, sample counts,
search counts, or strength. The defaults are `"pcf"`, `"medium"`, and `1`.

Sampling quality changes only the fixed filter tap budget. It must not change
the PCF radius, PCSS blocker-search radius, or maximum PCSS penumbra radius.
Built-in sampling must be deterministic within a frame and must not depend on
TAA or temporal history.

`ShadowMapKind` is limited to `"single"`, `"cascaded"`, and
`"paged-shadow"`. Custom registry entries and public definition subclasses are
not supported.

## Planning

`ShadowPlanner` must derive projection, technique fallback, cascade count,
resolution degradation, storage selection, and render jobs from immutable
definition snapshots, light state, camera state, caster bounds, and backend
identity. A plan must not change after it is published.

`RenderBackendProfile` must not expose shadow-planning capability metadata.
`ShadowPlanner` must own the fixed policies for built-in backend identifiers.
Unknown backend identifiers must use the planner's fixed cross-backend policy;
custom backends cannot inject or override shadow planning policy.

Shadow warnings must be limited to invalid projection results and
planner-selected degradation, fallback, or disabling. Renderer coordination
must publish those warning diagnostics through its deduplicated warning
channel.

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

Atlas storage supports PCF and PCSS on every built-in backend. Paged storage
supports PCF only. A paged definition requesting PCSS must retain its requested
quality, resolve its effective filter to PCF, and emit `filter-fallback`.

Directional cascade depth comparison bias must be compensated by the
reciprocal light-space depth range of the selected orthographic cascade. The
compensation factor must be clamped to at most `1` so a depth range below one
world unit does not amplify authored bias. WebGPU, WebGL, Software, atlas, and
paged-shadow consumers must apply the same compensation to constant, texel,
and slope depth bias before comparing receiver and shadow depths. Normal bias
remains a world-space receiver offset and must not use depth-range
compensation. Single projections and perspective spot or point projections
must retain their existing bias behavior.

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

PCF logical taps must represent bilinear `2x2` percentage-closer results.
WebGPU may use hardware comparison filtering; WebGL and Software must reproduce
the same interpolation from four depth comparisons. PCSS must average
individually linearized blocker depths before estimating penumbra width.
Projection-depth reconstruction must use the selected slice projection rather
than backend-specific approximations.
