# Migration Guidance

This document collects the upgrade actions that remain relevant to current
IgnisRenderer consumers and contributors. Completed implementation history that
does not require an action is retained in Git rather than in the documentation
tree.

## Opaque Renderer Resources

Remove reads and writes of `_gpuResource`, `_gpuTexture`, `_gpuView`,
`_webgpuTexture`, `_cpuData`, and `_cpuPixels` on renderer resources. Native
state is private to the backend. Keep application source data separately and
use resource factories, upload operations, and command encoders for GPU work.

Obtain resources from the owning renderer or backend extension instead of
constructing wrapper-shaped objects, cloning facades, or proxying them. The
resource-specific readonly `kind` distinguishes buffers, textures, samplers,
shader modules, pipelines, and binding groups. Every resource now declares an
idempotent `destroy()` operation; custom backend implementations must implement
the corresponding resource kind and explicit lifetime.

Shared binding entries accept renderer buffers, textures, samplers, or a
`BufferBindingRange` whose `buffer` is an `IRenderBuffer`. Replace raw native
binding entries with resources from the same owner. After device or context
restoration, recreate resources instead of reusing retired facades.

See the [renderer contract](../contracts/renderer.md).

## Fog Rendering

Remove `FogOptions.application` from fog configuration. `FogPass` now always
executes in the post-process stack; scene-mode fog is no longer supported.
The default post-process falloff, color, density, strength, order, and alpha
behavior remain unchanged. Configurations that previously selected scene mode
must account for screen-space depth and the later fog composition order,
especially when rendering transparent surfaces and particles.

Custom WebGPU scene and particle integrations must remove the frame-group fog
uniform at binding `4`. Other frame binding numbers remain unchanged. Custom
WebGL scene and particle integrations must remove `uFogParams0` and
`uFogParams1`; the dedicated fog post-process shader retains its own uniforms.

The unused `webgl.part.postProcessStubFragment` source is no longer available.
Use dedicated post-process passes instead of requesting the combined placeholder
shader.

## Texture Sampling Color Space

Remove uses of `Texture.formatExplicit`; current format and source encoding
determine native sampling regardless of how the format was initialized.

GPU color texture samples now arrive in linear RGB through native texture formats.
Remove shader-side source sRGB decoding and the removed per-texture decoding
uniforms from custom integrations. Display-output encoding remains separate.

Byte color textures default to `rgba8unorm-srgb`; linear byte textures use
`rgba8unorm`. Floating-point sources default to HDR. An explicit format without a
`colorSpace` now infers source encoding from that format: non-sRGB integer and
normalized formats imply linear data. Specify `colorSpace: "sRGB"` when uploading
sRGB source bytes through a compatible explicit non-sRGB format.

Material data slots use non-sRGB sampling formats. Custom `ShaderMaterial`
bindings retain their `linear` override by selecting the native sampling resource,
without shader decode constants. An sRGB source without a compatible sRGB format,
or a backend fallback that loses sRGB decoding, now throws instead of relying on
shader conversion. Import current shared WGSL declarations. The background
parameters now occupy 16 bytes. The environment header places the maximum mip
level at offset `12` and SH coefficients at offset `16`; its complete buffer is
4,224 bytes. Remove obsolete decoding-field padding from custom environment
layouts. Material transform vectors retain their existing aligned layout.

See the [material contract](../contracts/materials.md).

## WebGPU Parameter Uniform Layouts

Custom integrations must adopt the compact parameter sizes listed in the
[WebGPU contract](../contracts/webgpu.md). Denoiser and SSRF trace buffers now
contain twelve scalar words, and SSGI trace buffers contain sixteen. Animation
parameters contain six scalar words in both scene and shadow consumers.

Remove unused tail padding from gamma, tone-mapping, presentation, occlusion,
and SSR composition buffers. These standalone uniforms may occupy fewer than
16 bytes. The IBL prefilter buffer remains 32 bytes, but its source mip count
moves to byte offset `24` and its zero padding moves to the final word.

Replace `writeSSGIDenoiseParams()` in tooling with `resolveSSGIOptions()` to
inspect normalized settings. The obsolete eight-float packing helper has been
removed; GPU SSGI denoising uses the shared denoiser's 48-byte parameter block.

## Probe Capture Dirty Reason

Replace the removed `reflection-probe` dirty reason and
`RENDER_DIRTY_REASON_MASK["reflection-probe"]` with `probe-capture` and
`RENDER_DIRTY_REASON_MASK["probe-capture"]`. Probe capture progress and completed
reflection capture writeback now share this reason. The removed name has no
compatibility alias and follows the normal unknown-reason fallback.

## Scene and ECS Projection

Use scene nodes directly for rendering, interaction, animation, physics, and
prefab instantiation. Applications that need an ECS projection create and own
an `experimentalECS.ECSWorld`:

```ts
const world = new experimentalECS.ECSWorld(scene);
const entity = world.getEntityByNode(node);
```

Replace `scene.ecs` with the application-owned world and `node.entityId` with
`world.getEntityByNode(node)`. Call `world.destroy()` when finished.

- Bind animation through `AnimationMixer.bindNode()`.
- Pass nodes, body handles, or body ids to physics runtime methods.
- Treat `NodePrefab.instantiate(scene)` as returning the cloned root `Node`;
  resolve its entity through the optional world when needed.

See the [ECS projection contract](../contracts/ecs.md).

## Interaction

Replace `InteractionManager` with `InteractionController` and attach it through
`attach(scene, camera, physicsSystem?)`. Replace ECS component registration
with `controller.interactables.set(node, interactable)`, using `Interactable`
instead of the deprecated `InteractableComponent` alias.

Use `getSelection()` for the primary node, `getSelectedNodes()` for the full
selection, and `InteractionState` for outlines, drag rectangles, and gizmo UI.
See the [interaction guide](../public/interaction.md).

## Mesh Vertex Colors

Remove `IVertex.color` and `IPrimitiveGeometry.colors` from authored geometry.
`MeshAsset.fromFaces()` ignores per-vertex color input, and `GLTFLoader` ignores
`COLOR_0`. Use material colors or textures for surface color.
`RGBA` is no longer re-exported by the mesh modules; import it from the package
root instead.
See the [geometry contract](../contracts/geometry.md) and
[loader contract](../contracts/loaders.md).

## Mesh Morph Weights

`MeshAsset.defaultMorphWeights` and the second `MeshAsset` constructor argument
are removed. Pass initial weights to `MeshInstance` instead:

```ts
const mesh = new MeshAsset(primitives);
const instance = new MeshInstance({ mesh, morphWeights: initialWeights });
```

Without `morphWeights`, a new instance allocates zero weights for every
primitive's morph targets. `GLTFLoader` initializes loaded instances using
mesh weights and node overrides. Reusing only a loaded instance's `mesh` does
not preserve its pose; pass its `morphWeights` explicitly or clone the instance
or prefab. Supplied and cloned weights are copied into independent arrays.
See the [geometry contract](../contracts/geometry.md) and
[loader contract](../contracts/loaders.md).

## Deferred ShaderMaterial Outputs

Replace `fragmentDeferredEntryPoint` with `fragmentDeferredBaseEntryPoint` and
provide `fragmentDeferredExtensionsEntryPoint` in `ShaderMaterialParams`.
Defaults are `fsMainDeferredBase` and `fsMainDeferredExtensions`. Both functions
belong in the same WebGPU `fragment` chunk with `mode: "deferred"`.

The base entry returns the seven standard G-buffer outputs. The extension entry
returns `gMaterialExt0` at location `0` as `vec4<f32>` and `gMaterialExt3` at
location `1` as `vec4<u32>`. Share the surface evaluation between entries so
discard, fragment depth, and sample mask agree. Mark the shared vertex
`position` output with `@invariant` so separate pipelines can reliably use
depth equality. Remove old extension storage bindings and `textureStore` calls;
the engine no longer adapts legacy shaders.
Missing entries fail frame preflight. See the [WebGPU contract](../contracts/webgpu.md).

## WebGPU Frame Uniforms

Custom WGSL using frame uniforms must adopt the new layout without legacy
field aliases. `FrameCameraUniforms` now contains four unjittered matrices:
`viewProjection`, `prevViewProjection`, `inverseView`, and `inverseProjection`,
followed by `jitterNdc` and `prevJitterNdc`. Replace `cameraPosition` and
`environmentBasis*` with the inverse-view columns and shared camera helpers.

Move ambient color, light counts, lighting enablement, and clustered enablement
reads to `frameLights`; shadow enablement to `frameShadows`; and SH/IBL state
to `frameEnvironment`. Counts and flags use `u32`. Background texture formats
provide linear samples directly. Existing binding numbers remain stable,
but buffer sizes and offsets change. Import the shared frame declarations
instead of copying their layouts into custom shaders.

The camera declaration and math helpers use the `ignis/webgpu/frame-camera`
import. Helpers take only the matrices and jitter required by the operation:

```wgsl
#import <ignis/webgpu/frame-camera>

let worldPosition = cameraReconstructWorldPosition(
	frame.inverseView, frame.inverseProjection, frame.jitterNdc, uv, linearDepth
);
```

Use `ignis/webgpu/frame-uniforms` for the light, shadow, and environment
declarations in shader compositions that provide the shared light-data types.

Backend frame packers accept separate camera/light/shadow/environment inputs
and return `ArrayBuffer`, with an optional reusable writer. The old combined
`WebGPUFrameUniformInput` and packed option vectors have been removed.
See the [WebGPU contract](../contracts/webgpu.md) and
[binding reference](../reference/webgpu-bindings.md) for the complete ABI.

## Shadow Sampling Quality

`ShadowSamplingSettings` now contains only `quality`, with supported values
`"low"`, `"medium"`, and `"high"`. Shadow filter selection and strength moved
to top-level shadow-map options. There is no compatibility adapter for removed
sampling fields.

```ts
scene.shadows.createCascaded({
	filterMode: "pcss",
	sampling: { quality: "high" },
	strength: 0.9,
});
```

Migrate an old positive `sampling.radius` to `filterMode: "pcss"`; otherwise
use `"pcf"`. Move `sampling.strength` to `strength`. Approximate old sample
budgets as `<= 4` to `"low"`, `<= 8` to `"medium"`, and larger budgets to
`"high"`. Custom PCF radii, PCSS radii, filter counts, and blocker-search
counts no longer have exact equivalents because the built-in filters now use
fixed cross-backend quality presets.

## Shadow Backend Profiles

`RenderBackendProfile.shadow` and `IShadowBackendCapabilities` are removed.
Backend profiles continue to advertise general shadow support through
`BackendCapabilities.shadows`, but they cannot configure light-type limits,
filter support, shadow budgets, transmission, or shadow planning.

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

## WebGL Physical Lighting

Require WebGL2 with `EXT_color_buffer_float`, a successful `RGBA16F`
framebuffer-completeness probe, and linear filtering for half- or full-float
textures. Handle `WebGLCapabilityError` from `WebGLBackend.initialize()` when
these requirements are unavailable; there is no `RGBA8` scene-radiance fallback.
Inspect `Renderer.getDisplayOutputState()` for actual display output, since
internal HDR support does not guarantee HDR presentation.

- Treat one world unit as one meter, including light ranges, material
  thickness, and attenuation distance. Retune point and spot intensities for
  inverse-square falloff; the default intensity remains `1`.
- Remove compensating probe-capture intensity multipliers and application-side
  `PI` corrections for SH. Use the units in the
  [lighting contract](../contracts/lighting.md#units-and-world-scale).
- Retune Phong and Gouraud highlights for energy-normalized Blinn-Phong.
  `PhongMaterial.specular` defaults to `{ r: 56, g: 56, b: 56 }` for roughly
  four-percent linear dielectric reflectance. Provide light, environment,
  probe, or emissive radiance where older scenes relied on an ambient floor.
- Split materials or reduce active maps when warmup or drawing reports
  `material-texture-unit-overflow`; active glTF/KHR maps consume texture units.
- Review scenes with mixed alpha and transmission layers: mesh transmission
  uses sorted composition with a scene snapshot before each transmissive
  packet rather than weighted OIT.

See the [WebGL contract](../contracts/webgl.md).

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

Custom material UV selection is resolved during shader source generation.
The `webgpu.material.textureHelpers` and `webgl.material.textureHelpers` source
keys are removed, and `ignisSelectShaderMaterialUv` is no longer injected.
Custom chunks can use the existing per-binding `ignisSampleTexture_<token>`
and `ignisSampleTextureLevel_<token>` functions with their existing parameters,
or select their UV parameter explicitly.

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

Use `renderFrame(nowMs)` for a manually scheduled frame and `renderLoop()` for
continuous rendering instead of the deprecated `renderScene(nowMs)` alias.

## Compute Runtime Sources

Keep the `WebGPUBackend` used to construct the renderer and pass it directly
to `ComputeRuntime`. Replace `Renderer` and `{ backend }` wrapper inputs with
that backend. See the [compute runtime guide](../public/compute-runtime.md).

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

## Screen-Space Ambient Occlusion

Retune GPU SSAO settings authored in pixel units to view-space distances.
`SSAOOptions.radius` defaults to `0.65` and `bias` to `0.025` on every backend.
Keep bias well below the radius. See the
[post-process contract](../contracts/postprocess.md#ambient-occlusion).

## WebGPU Deferred G-buffer

The extended WebGPU deferred G-buffer uses seven compact color attachments for
the base entry and two color attachments for extension replay. `gMaterialExt1`
and `gMaterialExt2` are removed. `gMaterialExt3` is `rgba16uint`, and
`gMotionDepth.w` contains a packed material word rather than a plain
shading-model number.

For materials with `ShaderMaterial.deferredLighting === true`, update the WebGPU
`fragment` chunk with `mode: "deferred"` to the formats and field mapping in the
[WebGPU contract](../contracts/webgpu.md#deferred-lighting). Adopt the two entry
points described in [Deferred ShaderMaterial Outputs](#deferred-shadermaterial-outputs).
Legacy storage-output chunks require explicit migration; the engine does not
adapt them.

Emit base color locations `0..6` in this order:
`gAlbedoAlpha`, `gNormalRoughMetal`, `gEmissiveOcclusion`, `gMotionDepth`,
`gSpecular`, `gCoatSheen`, and `gSheenReflectance`. Emit `gMaterialExt0` at
extension location `0` as `vec4<f32>` and `gMaterialExt3` at location `1` as
`vec4<u32>`. Extension replay uses depth comparison `equal` with depth writes
disabled. Remove the old group `3` extension storage bindings and their
`textureStore` calls.

Custom deferred shaders always select the extended layout. Follow the shared
material-word and extension packing defined by `deferredGBufferCodec.wgsl`.
The deferred resolve read ABI remains color bindings `0..6`, `gMaterialExt0In`
at binding `7`, and `gMaterialExt3In` at binding `8`.

## Shadow Definitions and Planning

The supported built-in authoring workflows are
`scene.shadows.createSingle`, `createCascaded`, and `bind`. Paged shadows
(`createPaged`, `PagedShadowMap`, option types, shaders, and backend implementations)
have been removed completely. Applications must migrate directional definitions to
`createCascaded` and other definitions to `createSingle` as appropriate.

Built-in shadow objects are definition facades; backend-native textures are
owned by the attached backend runtime.

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

- [Renderer guide](../public/renderer.md)
- [Renderer contract](../contracts/renderer.md)
- [Lighting contract](../contracts/lighting.md)
- [Shader contract](../contracts/shaders.md)
- [Post-process contract](../contracts/postprocess.md)
- [WebGL contract](../contracts/webgl.md)
