# Shadow Contract

`ShadowFramePlan` is the sole cross-backend shadow planning input for a frame.
It is an immutable atlas-oriented planner result; backends must not add native
allocation or residency state to it.

See [renderer contract](renderer.md) and [WebGPU contract](webgpu.md) for the
frame lifecycle and backend resource ownership rules.

## Authoring

Applications configure built-in definitions through `scene.shadows`.

`Scene.shadows` must be the sole authoring authority for shadow enablement.
`Renderer.features` must not expose a renderer-level shadow switch. A bound
shadow definition participates in planning only while its `enabled` property is
`true`; applications that need to enable or disable shadows must update the
relevant definition through its property or `update()` method.

`createSingle()`, `createCascaded()`, `bind()`, `unbindLight()`, `destroy()`,
`clear()`, and `getBoundShadowMap()` are the supported authoring and lifecycle
API. Built-in definitions keep their observable properties and `update()`
method.

`ShadowFilterMode` is limited to `"pcf"` and `"pcss"`. Definitions select
the filter through top-level `filterMode`, select `"low"`, `"medium"`, or
`"high"` through `sampling.quality`, and control artistic attenuation through
top-level `strength`. `sampling` must not expose filter radii, sample counts,
search counts, or strength. The defaults are `"pcf"`, `"medium"`, and `1`.

Sampling quality changes only the fixed filter tap budget. It must not change
the PCF radius, PCSS blocker-search radius, or maximum PCSS penumbra radius.
Built-in sampling must be deterministic within a frame and must not depend on
TAA or temporal history.

The active filter and blocker-search offsets for every quality preset must be
centered around the receiver. Sampling rotation may depend on the light and
cascade, but it must not hash quantized receiver coordinates or otherwise
change discontinuously when the receiver crosses a shadow-map texel boundary.

`ShadowMapKind` is limited to `"single"` and `"cascaded"`. Custom registry
entries and public definition subclasses are not supported.

## Planning

`ShadowPlanner` must derive projection and filter fallback, cascade count, and
resolution degradation from immutable definition snapshots, light state,
camera state, caster bounds, and backend identity. A plan must not change after
it is published.

`RenderBackendProfile` must not expose shadow-planning capability metadata.
`ShadowPlanner` must own the fixed policies for built-in backend identifiers.
Unknown backend identifiers must use the planner's fixed cross-backend policy;
custom backends cannot inject or override shadow planning policy.

Shadow warnings must be limited to invalid projection results and
planner-selected degradation, fallback, or disabling. Renderer coordination
must publish those warning diagnostics through its deduplicated warning
channel.

Directional cascaded shadows must use practical camera-frustum splits. Spot
cascades must use cone-depth splits, and point cascades must produce six cube
faces per cascade. Cascade counts are clamped to the built-in range, and
unsupported projection or filter modes must produce an explicit fallback
diagnostic. Atlas sampling supports PCF and PCSS on every built-in backend.

Directional cascade depth comparison bias must be compensated by the
reciprocal light-space depth range of the selected orthographic cascade. The
compensation factor must be clamped to at most `1` so a depth range below one
world unit does not amplify authored bias. WebGPU, WebGL, and Software
consumers must apply the same compensation to constant, texel, and slope depth
bias before comparing receiver and shadow depths. Normal bias remains a
world-space receiver offset and must not use depth-range compensation. Single
projections and perspective spot or point projections must retain their
existing bias behavior.

Projection stabilization history belongs to per-renderer `ShadowPlannerState`.
`FrameCoordinator` must pass that state to the static `ShadowPlanner.plan()`
entrypoint. The state must be reset when the definition revision, binding,
effective projection shape, technique, or resolution changes, and history must
be removed for inactive or disabled lights.
Caster-bound radius shrink must be smoothed while stabilization is active so a
temporary bounds reduction does not immediately resize the projection.

## Backend Runtime

Backends consume `ShadowFramePlan` plus prepared caster and transmitter
`DrawSubmission` collections. `PreparedSceneState.shadowCasterSubmissions` and
`PreparedSceneState.shadowTransmitterSubmissions` provide baseline mesh work;
late particle work may contribute additional submissions through the frame's
`ShadowWorkSet`. Shadow runtimes must not depend on main-camera `DrawPacket`
sorting state. Backends own their mutable atlas slots, textures, framebuffers,
and binding caches. Native allocation failure may make sampling fully lit for
that frame but must not mutate the shared plan.

Atlas caster passes must consume the prepared lights selected by the shared
plan. Backend allocation offsets must not be written into prepared slices or
definitions.

PCF logical taps must represent bilinear `2x2` percentage-closer results.
WebGPU may use hardware comparison filtering; WebGL and Software must reproduce
the same interpolation from four depth comparisons. PCSS must average
individually linearized blocker depths before estimating penumbra width.
Projection-depth reconstruction must use the selected slice projection rather
than backend-specific approximations.
