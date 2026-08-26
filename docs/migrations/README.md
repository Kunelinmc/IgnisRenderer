# Migration Guidance

This document collects the upgrade actions that remain relevant to current
IgnisRenderer consumers and contributors. Completed implementation history that
does not require an action is retained in Git rather than in the documentation
tree.

## Shadow Backend Profiles

`RenderBackendProfile.shadow` and `IShadowBackendCapabilities` are removed.
Backend profiles continue to advertise general shadow support through
`BackendCapabilities.shadows`, but they cannot configure light-type limits,
filter support, shadow budgets, transmission, or paged-shadow planning.

Remove the `shadow` property from custom backend profiles. `ShadowPlanner` now
selects fixed policies for built-in backend identifiers and a fixed
cross-backend policy for custom identifiers. There is no compatibility adapter
or custom shadow-policy registration API.

## Environment Lighting Ownership

Environment spherical harmonics projection and specular prefiltering are
explicit application or tooling operations. `Renderer` no longer schedules an
environment update stage or exposes environment bake/update methods.

Replace renderer-owned environment updates with:

- `projectEnvironmentTextureToSH(...)` for spherical harmonics coefficients.
- `IBLPrefilter` or `prefilterEnvironmentIBL(...)` for specular mip chains.
- Explicit assignment of the generated data to the target probes.
- `renderer.requestRender("lighting")` after replacing probe data.

The removed surface includes `Renderer.setEnvironmentIBLUpdateOptions`,
`Renderer.getEnvironmentIBLUpdateOptions`,
`Renderer.requestEnvironmentIBLUpdate`,
`bakeEnvironmentIBLFromEnvironmentMap`, and `EnvironmentIBLBake*` types.

`IBLPrefilter` accepts its backend through `service`. Positional sources,
`IBLPrefilterBackendSource`, `IBLPrefilterSourceOptions`,
`IBLPrefilterOptions.backend`, `IBLPrefilterOptions.computeSource`, and direct
WebGPU compute sources are removed.

## Mesh Bounds Mutation

`MeshAsset` no longer observes direct primitive-array mutation or polls
`IPrimitive.geometryVersion`. Primitive arrays are readonly snapshots and each
primitive belongs to one mesh asset.

Replace direct mutations with the owning asset API:

```ts
mesh.addPrimitive(primitive);
mesh.setPrimitives(primitives);
mesh.setPrimitiveGeometry(primitive, replacementGeometry);

primitive.geometry.positions[0] = nextX;
mesh.markPrimitiveGeometryDirty(primitive);
```

Use `replacePrimitive` and `removePrimitive` for other structural changes.
Direct writes to `mesh.boundingBox`, `mesh.boundingSphere`,
`primitive.geometryVersion`, and the `mesh.primitives` array are no longer
supported.

## WebGL Context Work

WebGL frame lifecycle operations, warmup, environment prefiltering, maintenance,
and custom render-target readback share the backend context work queue.
Applications must wait for an active renderer frame to finish before requesting
custom render-target readback.

## Post-Process Execution Declarations

Custom post-process implementations describe color, G-buffer, history,
transient, backend-shared resources, and frame requirements through one
`describeExecution()` result.

Migrate resource behavior away from graph metadata, context-binding metadata,
and descriptor-specific methods. Use the fixed declaration-checked resource
accessor during execution. There is no compatibility adapter for the removed
metadata and pass-ID execution paths.

## Shader Directive Runtime

Directive processing is owned by `ShaderRuntime`, `ShaderDirectiveStage`, and
`ShaderBackendCompileStage`. Replace `preprocessEngineShaderDirectives(...)`
and `ENGINE_DIRECTIVE_RUNTIME` with the staged runtime API.

`ShaderRule` supports `transform` and `replace`; source maps may carry
`schemaVersion` and optional column spans. Runtime caches are memory-only, so
persisted cache migration is unnecessary.

The cross-backend `ShaderDirectiveProfileRegistry`,
`DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY`,
`createDefaultShaderDirectiveProfileRegistry()`, and registry completeness
assertion are removed. Construct `ShaderBackendCompileStage` with one composed
backend-owned `profile`. Injection scripts must declare an argument schema;
backend directive hooks remain additive and may not replace profile modules.

Built-in engine injection scripts are removed. Custom shader integrations must
not invoke `ignis/material/uniform-block`,
`ignis/material/texture-binding`, `ignis/postprocess/luma`,
`ignis/postprocess/fxaa`, or `ignis/postprocess/volumetric`.
`ShaderMaterial.uniformBindings` and `textureBindings` now produce
backend-owned generated source blocks automatically. Shader chunks using those
bindings must remove matching manual uniform, sampler, binding, and helper
declarations. Generic hook-provided `ShaderInjectionScript` values remain
supported.

## Software Backend Lifecycle

`SoftwareBackend` participates in the same attach, initialize, frame
transaction, abort, restore, and destroy lifecycle as the GPU backends. Illegal
lifecycle calls now throw, and failed frames do not publish temporal or
post-process history.

The package-root `Rasterizer` export is removed without a public replacement.
Applications render through `Renderer` configured with `SoftwareBackend`.

## Renderer Construction

Renderer construction now uses
`new Renderer(canvas, backend, camera?, options?)`. Move `canvas`, `backend`,
and the optional initial `camera` out of the former single options object.
`RendererOptions` now contains only optional renderer configuration.

## Software HDR Color Pipeline

The Software backend color pipeline is now normalized linear RGBA32F. This is
a breaking rendering change: lighting and post-process intermediates are no
longer byte-scaled, radiance above `1.0` is preserved, and previous Software
screenshots or exact RGBA8 pixel values are not expected to match.

`FrameAttachments.pixels` remains available as an RGBA8 presentation and
diagnostic preview, but is no longer authoritative scene color. Custom
Software post-process implementations must replace direct access to that
buffer with the declaration-checked `context.resources.color` accessor. TAA
or other custom histories that represent color should use float resources and
must not clamp HDR radiance to `1.0`.

Continue using `RendererOptions.displayOutput`, `setDisplayOutput()`, and
`getDisplayOutputState()`. Software now attempts Display-P3 Float16 Canvas 2D
HDR for `"auto"` and `"hdr"`; unsupported browsers and non-HDR displays safely
fall back to SDR. Callers handling fallback reasons should accept the new
`canvas-hdr-output-unsupported` value.

## Screen-Space Global Illumination

WebGPU SSGI uses depth-based ray marching, independent temporal history, and
depth/normal-aware denoising. Remove legacy gather-mode configuration and pass
the current world-space distance and thickness settings directly. WebGL,
Software, and orthographic WebGPU cameras do not execute SSGI.

## WebGPU Deferred G-buffer

The WebGPU deferred G-buffer shader contract now uses seven compact color
attachments and two storage payloads. `gMaterialExt1` and `gMaterialExt2` are
removed. `gMaterialExt3` is `rgba16uint`, and `gMotionDepth.w` contains a packed
material word rather than a plain shading-model number.

Applications with `ShaderMaterial.deferredLighting === true` must update their
`fragment-deferred` chunks to the formats and field mapping in the
[WebGPU contract](../contracts/webgpu.md#deferred-lighting). Existing chunks
that declare four `rgba16float` storage outputs are not compatible and do not
receive a legacy adapter. TypeScript construction and entry-point APIs remain
unchanged.

Custom deferred fragments must emit color locations `0..6` in this order:
`gAlbedoAlpha`, `gNormalRoughMetal`, `gEmissiveOcclusion`, `gMotionDepth`,
`gSpecular`, `gCoatSheen`, and `gSheenReflectance`. Group `3`, binding `0` is
the write-only `rgba16float` `gMaterialExt0Out`; binding `1` is the write-only
`rgba16uint` `gMaterialExt3Out`. Custom deferred shaders always select the
extended layout. They must follow the shared material-word and extension packing
defined by `deferredGBufferCodec.wgsl`; the deferred resolve read ABI is color
bindings `0..6`, `gMaterialExt0In` at binding `7`, and `gMaterialExt3In` at
binding `8`.

## Shadow Definitions and Planning

The built-in `scene.shadows.createSingle`, `createVariance`,
`createCascaded`, `createPaged`, and `bind` workflows remain supported. Built-in
shadow objects are definition facades; backend-native textures and residency
are owned by the attached backend runtime.

Direct assignment to `enabled`, `size`, `priority`, `bias`, and `sampling`
remains supported and now invalidates the scene. Prefer `shadow.update({...})`
when changing multiple settings so the scene is invalidated once.

`ShadowMapRegistry`, `scene.shadows.registerMapType`, external
`ShadowMapBase` subclassing, `ShadowConfig`, `ShadowRenderSet`, and
`Renderer.shadowMaps` are removed. Migrate applications to built-in
`scene.shadows` definitions; custom backends must consume immutable
`ShadowFramePlan` directly. There is no compatibility adapter or transition
window.

## Related Documents

- [WebGL physical lighting migration](webgl-physical-lighting.md)

- [Renderer guide](../public/renderer.md)
- [Renderer contract](../contracts/renderer.md)
- [Lighting contract](../contracts/lighting.md)
- [Shader contract](../contracts/shaders.md)
- [Post-process contract](../contracts/postprocess.md)
- [WebGL contract](../contracts/webgl.md)
