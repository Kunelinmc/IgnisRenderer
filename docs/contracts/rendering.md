# Rendering Contract

This document defines cross-backend rendering features, capability gating, pass placement, fallback behavior, and diagnostics.

## Contract

### Order-independent transparency

- `BackendCapabilities.oit` must exist on all backends.
- `WebGPUBackend.profile.capabilities.oit` must be `true`.
- `WebGLBackend.profile.capabilities.oit` must be `true`.
- `SoftwareBackend.profile.capabilities.oit` must be `false`.
- `RendererFeatureRequest.enableOIT` must be accepted by feature resolution.
- `Renderer.features.enableOIT` defaults to `false`.
- `resolveFeatureState(...)` must auto-disable `enableOIT` when backend
  capability is `false` and must emit a feature warning.
- OIT must activate only when all runtime constraints are satisfied:
  - Backend is `WebGPU` or `WebGL`.
  - OIT runtime textures are available.
  - For `WebGPU`:
    - MRT scene targets must be available.
    - `sampleCount` must be exactly `1`.
    - Native command-encoder texture-copy access must be available.
  - For `WebGL`:
    - `EXT_color_buffer_float` must be available.
    - scene, post-process, and OIT framebuffers must be complete.
- When active, transparent packets must be partitioned:
  - `materialUsesTransmission(packet.material) === true`:
    route to legacy transmission path.
  - In `WebGLBackend`, `packet.material instanceof ShaderMaterial`:
    route to legacy transparent path.
  - otherwise:
    route to OIT path.
- Particle routing must follow:
  - `ParticleBlendMode.Alpha` -> OIT particle pipeline.
    - `WebGPUBackend` should use `fsMainOIT`.
    - `WebGLBackend` should use OIT pass-mode shading with separate `accum`
      and `reveal` draws.
  - `ParticleBlendMode.Additive` -> legacy additive pipeline.
- OIT resolve must use a separate fullscreen pass and must not read/write the
  same texture simultaneously.
  - `WebGPUBackend` must copy `sceneColorMain` into `oitSceneColorCopy` before
    any OIT accumulation draw, then resolve back into `sceneColorMain`.
  - If that in-frame copy fails while recording, WebGPU must use the legacy
    transparent path for the same frame. It must not silently discard OIT
    contributors after they have been classified.
  - `WebGLBackend` must copy `sceneColor` into `postColorTexture`, then resolve
    back into `sceneColor`.
- WebGPU deferred ordering contract:
  - OIT must not write G-buffer deferred material payload textures.
  - OIT must execute after `main-opaque` deferred lighting resolve.
  - `transmission` materials must remain on the legacy transparent path after
    OIT resolve or after alpha particles, matching existing pass ordering.
  - Deferred opaque payload compaction must not change transparent, OIT,
    transmission, particle, wireframe, or `depthWrite === false` routing.

### Occlusion culling

- `RendererFeatureRequest.enableOcclusionCulling` may request occlusion
  culling and must default to `false`.
- `RendererFeatureRequest.occlusionCullingOptions` may override
  `OcclusionCullingOptions`.
- `DEFAULT_OCCLUSION_CULLING_OPTIONS` must define:
  - `minCandidateScreenAreaPx = 64`.
  - `minOccluderScreenAreaPx = 256`.
  - `hysteresisFrames = 2`.
  - `maxReadbackLatencyFrames = 3`.
  - `debug = false`.
- `BackendCapabilities.occlusionCulling` must exist on all backends.
- `WebGPUBackend.profile.capabilities.occlusionCulling` must be `true` unless
  `WebGPUBackendOptions.enableOcclusionCulling === false`.
- `WebGLBackend.profile.capabilities.occlusionCulling` must be `false`.
- `SoftwareBackend.profile.capabilities.occlusionCulling` must be `false`.
- Backends that support occlusion culling must expose an
  `OcclusionCullingBackendAdapter` through the `renderer.occlusion-culling`
  backend extension.
- `Renderer` must resolve occlusion culling integration with
  `resolveOcclusionCullingBackendExtension(backend)?.api`.
- `resolveFeatureState(...)` must disable `enableOcclusionCulling` when the
  backend capability is `false` and must emit a feature warning.
- `PreparedScene.occlusion` may expose prepared-scene occlusion metadata,
  including candidates, culled packet ids, statistics, and the visibility
  source frame.
- `OcclusionVisibilityProvider` must provide synchronous snapshot queries only.
  It must not perform asynchronous GPU waits during prepared-scene building.
- `PreparedSceneBuilder` must only filter main-camera `opaquePackets`.
- `PreparedSceneBuilder` must not filter `transparentPackets`,
  `shadowCasterPackets`, `shadowTransmitterPackets`, reflection captures, or
  probe captures.
- Decal packet generation must run after opaque packet filtering, so hidden
  opaque receivers do not create decal work.
- WebGPU occlusion culling must use backend-internal frame graph nodes and must
  not add renderer-level global pipeline stages.
- The WebGPU internal frame graph may create an `occlusion-test` node only when
  occlusion culling is enabled and the prepared scene has eligible candidates.
- The `occlusion-test` node must execute after opaque depth or deferred depth is
  available.
- WebGPU occlusion culling must rely on the shared frame Hi-Z texture built
  from `gMotionDepth` after opaque depth is available. The occlusion runtime
  must not own a separate Hi-Z texture or build pipeline.
- Missing `gMotionDepth`, missing candidates, device restore, resize, camera
  reset, stale results, missing results, and packet signature changes must make
  affected candidates visible.
- A candidate may be hidden only after `hysteresisFrames` consecutive occluded
  GPU results.
- A single visible GPU result must immediately make the candidate visible.
- Eligibility must exclude blend, transmission, wireframe,
  `depthWrite = false`, custom shader, alpha-mask, and unsupported topology
  packets.

### HDR rendering

- `RendererOptions.displayOutput` may request `"sdr"`, `"auto"`, or `"hdr"`.
  The default must be `"sdr"`, with exposure `1.0` and HDR headroom `4.0`.
- Exposure must be finite and within `[0, 64]`. HDR headroom must be finite and
  within `[1, 16]`. Invalid values must throw `RangeError`.
- `Renderer.getDisplayOutputState()` must return `null` before initialization
  and the backend-resolved state afterwards.
- `Renderer.setDisplayOutput()` must merge partial settings with the current
  requested state and reconfigure presentation only at a frame boundary.
- `"auto"` must enable HDR only when the display reports high dynamic range
  and the backend's requested HDR canvas configuration can be verified.
- An explicit `"hdr"` request must use the same requirements. If any
  requirement fails, rendering must continue in SDR and report a fallback
  reason.
- WebGPU HDR presentation must use an `rgba16float` canvas, Display-P3 color
  space, and extended canvas tone mapping.
- Software HDR presentation must use a Display-P3 Canvas 2D context with
  `colorType: "float16"` and `rgba-float16` `ImageData`. A detached-canvas
  put/read probe must verify Float16 storage, Display-P3, and preservation of
  values above `1.0` before HDR may become active. `SoftwareBackend` must
  request and own the visible canvas context, provide it to the surface runtime
  for presentation, and provide it to the display-output manager for
  verification. The display-output manager must not request the visible context
  itself.
- WebGL HDR presentation must require a high-dynamic-range display query,
  `drawingBufferStorage()`, `drawingBufferFormat`, `drawingBufferColorSpace`,
  `EXT_color_buffer_float`, and a verified `RGBA16F` Display-P3 drawing buffer.
  Unsupported or rejected configurations must restore SDR presentation without
  failing the internal HDR backend.
- SDR presentation must use the user agent's preferred 8-bit canvas format,
  sRGB color space, and standard tone mapping when that member is supported.
- WebGL must resolve `"auto"` and `"hdr"` through its verified drawing-buffer
  capability. Probe failure must fall back to SDR without throwing. Explicit
  HDR failure must report the applicable display, Canvas HDR, or configuration
  reason; automatic fallback must not warn.
- WebGL must preserve internal scene radiance in `rgba16float` targets from
  scene shading through presentation. SDR output must apply exposure, ACES tone
  mapping, and the piecewise sRGB transfer function. HDR output must apply the
  shared hue-preserving shoulder, convert linear sRGB to linear Display-P3, and
  retain encoded RGB values above `1.0` up to the requested headroom.
- WebGL display-output changes and dynamic-range media-query changes must apply
  at a frame boundary. Resize and context restoration must reconfigure and
  verify the active drawing-buffer format and color space.
- Software must resolve `"auto"` and `"hdr"` through its Canvas 2D HDR probe.
  Probe failure must fall back to SDR without throwing. Explicit HDR failure
  must report the applicable display, Canvas HDR, or configuration reason.
- `WebGPUBackend` and `WebGLBackend` must preserve scene, post-process, OIT,
  motion, normal, specular, and transmission-background radiance in
  `rgba16float` render targets. `SoftwareBackend` must preserve authoritative
  scene and post-process radiance in one reusable RGBA32F target. Its public
  `FrameAttachments.pixels` remains an RGBA8 presentation and diagnostic
  buffer and must not be used as authoritative scene color.
- The logical post-process color domains are `scene-linear-hdr`,
  `display-linear`, and `display-encoded`.
- Built-in passes must declare their input and output domains. A declared pass
  whose input does not match the current domain must be skipped with a stable
  diagnostic.
- A custom pass without a color contract must be treated as domain-preserving.
  It must continue to run; HDR output must emit a stable warning.
- Presentation must complete any missing tone mapping, gamut conversion, or
  transfer encoding based on the final post-process domain.
- SDR mapping must use ACES fitted tone mapping and the piecewise sRGB transfer
  function. HDR mapping must apply exposure, a hue-preserving soft shoulder,
  linear-sRGB to linear Display-P3 conversion, and extended sRGB encoding
  without clamping encoded RGB to `1.0`.

### Early-Z prepass

#### Configuration Contract

The option `enableEarlyZPrepass?: boolean` is accepted by `SoftwareBackendOptions`, `WebGLBackendOptions`, and `WebGPUBackendOptions`.

- **Input contract**: Accepts `true`, `false`, or `undefined`.
- **Default behavior**: `undefined` must be treated as `true`.
- **Behavior contract**:
	- When `true`, the main opaque path must execute a depth-only prepass before color/G-buffer shading.
	- When `false`, the main opaque path must run without the prepass and must keep the legacy single-pass depth test and write behavior.

#### WebGL-Specific APIs

- `WebGLBackend.isEarlyZPrepassEnabled()` must return the resolved backend option.
- The WebGL backend capability `occlusionCulling` must remain `false`.
- `ShaderMaterial.chunks` may contain a WebGL depth prepass chunk with `backend: "webgl"`, `language: "glsl"`, and `stage: "fragment-depth"`.
- `ShaderMaterial.resolveWebGLDepthPrepassProgram(mode, options)` must return a custom WebGL vertex and depth fragment program when both vertex and depth-fragment chunks are available. It must return `null` when the material has not opted in.

#### Pass Scope & Eligibility Contract

- The Early-Z prepass must apply only to `main-opaque`.
- `main-transparent`, OIT passes, transmission passes, particles, and environment rendering must not use the prepass.
- **WebGPU Deferred Path**: When deferred lighting is active, the Early-Z prepass must run before the G-buffer geometry pass and must not run before the fullscreen deferred lighting resolve pass.
- **Depth-Write Disable**: Materials with `depthWrite === false` must not participate in the prepass. Their color draws must not use the read-only depth state since they were not prepassed.

#### Material Mask / Alpha Discard Contract

- Opaque mask materials (`material.alphaMode === MASK`) must run alpha-test discard in the depth prepass and must not write color targets.
- **Software Backend**: Triangles with `material.alphaMode === MASK` must be skipped in the prepass and must be evaluated in the regular color pass.
- **WebGL Backend**: Built-in material depth prepass fragments must only apply alpha mask discard. The discard test must use `uBaseColor.a * texture(uBaseMap, uv).a < uAlpha.x` when `uHasBaseMap == 1`, and `uBaseColor.a < uAlpha.x` otherwise. A WebGL `ShaderMaterial` using `AlphaMode.Mask` without a WebGL `fragment-depth` chunk should warn once and skip the Early-Z prepass for that material.
- **Animated geometry**: WebGL and WebGPU early-Z vertex processing must apply
  the same current morph and skinning deformation used by the matching color or
  G-buffer draw. Animated opaque packets must not be excluded solely because
  they use skinning or morph targets.
- **WebGPU Backend**: A `ShaderMaterial` with `alphaMode === MASK` must provide explicit depth prepass fragment shader configuration via `depthFragmentCode` and `depthFragmentEntryPoint`. If this configuration is missing, the engine must skip the Early-Z prepass for that material and log a warning once.

#### Pipeline and Render State Contract

- **WebGPU Backend**:
	- Opaque non-mask prepass must use depth-only pipeline state with `depthWriteEnabled = true` and `depthCompare = less`.
	- Opaque color draws that were prepassed must use read-only depth state with `depthWriteEnabled = false` and `depthCompare = less-equal` (`early-z-color`).
	- Opaque color draws that were not prepassed must keep legacy depth state (`depthWriteEnabled = material.depthWrite !== false` and `depthCompare = less`).
	- G-buffer draws that were prepassed must use the same read-only `early-z-color` depth state as legacy MRT color draws.
- **WebGL Backend**:
	- The depth prepass must use depth writes, `LESS` depth test, disabled blending, and disabled color writes.
	- The following color pass must use `LEQUAL` depth test and read-only depth only for packet IDs submitted by the depth prepass.
- **Software Backend**:
	- During color shading, when the early-depth buffer is present, the early gate must use `zCamValue <= earlyDepthBuffer[pixel]`.
	- Final visibility write to `attachments.depthBuffer` must keep strict `shadedDepth < depthBuffer[pixel]`.

#### Buffer and Clears Contract

- **Software Backend**:
	- The implementation must maintain an internal reusable early-depth buffer.
	- Full-frame rendering must clear the full early-depth buffer region to `Infinity` before prepass writes.
	- Incremental rendering must clear dirty rect regions to `Infinity` and may preserve non-dirty regions.
- **WebGPU Backend**:
	- Incremental dirty-rect flow must clear the dirty depth region to `1.0` before the prepass.
	- Incremental dirty-rect flow must clear scene color and every active
	  G-buffer attachment inside the dirty region before scene draws. It must
	  preserve attachment contents outside the dirty region.
	- The Early-Z prepass, G-buffer geometry pass, and legacy color pass must clip to resolved dirty rects.

### Projected decals

`Node.renderLayers` must be an unsigned bitmask. The default value is `1`
(layer bit 0).

`Decal.material` may reference any built-in `Material`. `ShaderMaterial` is not
executed as a decal source and must be skipped.

`Decal.receiverLayerMask` must match receiver pixels through
`MeshInstance.renderLayers & Decal.receiverLayerMask`. Bits `0..10` are the
cross-backend guaranteed receiver range. Higher bits may be used by CPU-side
APIs but must not be relied on for cross-backend decal receiver tests.

`Decal.priority` must sort decals in ascending order. Decals with equal priority
must retain scene traversal order.

WebGPU batch optimization must not change the observable result of overlapping
decals. Any device limit, material binding, texture binding, storage-texture, or
tile/bin overflow that prevents exact ordered batching must fall back to the
per-decal ordered path.

WebGPU device negotiation must require at least `28`
`maxSampledTexturesPerShaderStage` bindings so the complete decal material
surface and G-buffer snapshot can coexist in the fragment stage.

`Decal.channelBlendModes` may set per-channel blend behavior. Supported modes
are `disabled`, `lerp`, `replace`, `multiply`, `add`, and `normal`.

`disabled` must preserve the receiver value. `lerp` and `replace` must perform
opacity-weighted replacement. `multiply` and `add` must perform weighted
component operations. `normal` must perform normalized direction blending for
`normal`, `clearcoatNormal`, and `anisotropy`; for scalar and color channels it
must behave as `lerp`. `multiply` and `add` must not modify direction channels.

Software and WebGPU decal coverage must be the product of `Decal.opacity`,
material factor alpha, base-color texture alpha, and edge fade.
`AlphaMode.Mask` must reject decal coverage when the resolved material alpha is
below `alphaCutoff`. A decal must not reclassify an opaque receiver into the
transparent pass.

Software and WebGPU normal and clearcoat-normal projection must use the
inverse-transpose projector normal transform. Software and WebGPU anisotropy
tangents must rotate the sampled tangent-space direction by the material
anisotropy rotation, use the projector linear transform, and be orthogonalized
against the resolved receiver normal after direction blending.

Software must normalize decal source colors to linear space before blending.
When a legacy Phong surface stores encoded color values, Software must adapt the
value at the surface-modifier boundary so lighting observes the same linear
result as a PBR receiver.

Incremental rendering must track `DecalPacket` additions, removals, projector
transform changes, `Decal.material` state, `Decal.receiverLayerMask`,
`Decal.priority`, `Decal.opacity`, `Decal.edgeFade`, and
`Decal.channelBlendModes`.

`Renderer.requestRender("decal")` and `Scene.invalidate("decal")` must plan the
first incremental pass as `main-opaque`, must not force a full-frame render by
reason alone, and must reset temporal history.

The default channel modes must be:

```ts
{
	baseColor: "lerp",
	normal: "normal",
	roughness: "lerp",
	metalness: "lerp",
	emissive: "lerp",
	occlusion: "lerp",
	specular: "lerp",
	specularColor: "lerp",
	clearcoat: "lerp",
	clearcoatRoughness: "lerp",
	clearcoatNormal: "normal",
	sheenColor: "lerp",
	sheenRoughness: "lerp",
	transmission: "lerp",
	thickness: "lerp",
	iridescence: "lerp",
	iridescenceThickness: "lerp",
	anisotropy: "lerp",
}
```

All omitted channels must fall back to their default mode.

## Usage

### Order-independent transparency

```ts
import { Renderer } from "../src/rendering/Renderer";
import { WebGLBackend } from "../src/backends/webgl/WebGLBackend";

const backend = new WebGLBackend();
const renderer = new Renderer({ backend, canvas, camera });
await renderer.initialize();

renderer.features.enableOIT = true;
await renderer.renderFrame(performance.now());
```

```ts
// Feature negotiation contract:
// - If backend supports OIT, `enableOIT` remains true.
// - If backend does not support OIT, it is disabled and warning is emitted.
const resolved = resolveFeatureState(
	{ enableOIT: true },
	renderer.backendProfile.capabilities,
	renderer.backendProfile.id
);
```

### Occlusion culling

```ts
import { Renderer } from "../src/rendering/Renderer";
import { WebGPUBackend } from "../src/backends/webgpu/WebGPUBackend";

const backend = new WebGPUBackend();
const renderer = new Renderer({ backend, canvas, camera });
await renderer.initialize();

renderer.features.enableOcclusionCulling = true;
renderer.features.occlusionCullingOptions = {
	hysteresisFrames: 2,
	maxReadbackLatencyFrames: 3,
};
await renderer.renderFrame(performance.now());
```

```ts
const disabledBackend = new WebGPUBackend({
	enableOcclusionCulling: false,
});
const disabledRenderer = new Renderer({
	backend: disabledBackend,
	canvas,
	camera,
});

console.assert(
	disabledRenderer.backendProfile.capabilities.occlusionCulling === false
);
```

```bash
bunx tsc --noEmit
bun tests/static/pipeline/test_render_list_builder.mjs
bun tests/static/pipeline/test_prepared_scene_cache.mjs
bun tests/static/renderer/test_backend_extensions.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_planner.mjs
bun tests/static/webgpu/test_webgpu_occlusion_culling_runtime.mjs
```

### HDR rendering

```ts
import { Renderer, WebGPUBackend } from "ignisrenderer";

const renderer = new Renderer({
	backend: new WebGPUBackend(),
	canvas,
	displayOutput: {
		mode: "auto",
		exposure: 1,
		hdrHeadroom: 4,
	},
});

await renderer.initialize();
console.log(renderer.getDisplayOutputState());

await renderer.setDisplayOutput({ mode: "hdr" });
```

### Early-Z prepass

#### Backend Configuration

```ts
import { SoftwareBackend } from "../src/backends/software/SoftwareBackend";
import { WebGLBackend } from "../src/backends/webgl/WebGLBackend";
import { WebGPUBackend } from "../src/backends/webgpu/WebGPUBackend";

// SoftwareBackend setup
const softwareBackend = new SoftwareBackend({
	enableEarlyZPrepass: true,
});

// WebGLBackend setup
const webglBackend = new WebGLBackend({
	enableEarlyZPrepass: true,
});
console.log(webglBackend.isEarlyZPrepassEnabled());

// WebGPUBackend setup
const webgpuBackend = new WebGPUBackend({
	enableEarlyZPrepass: true,
});
```

#### Custom Shader Material Configuration (WebGL)

```ts
import { ShaderMaterial } from "../src/materials/ShaderMaterial";

const webglMaterial = new ShaderMaterial({
	chunks: [
		{
			backend: "webgl",
			language: "glsl",
			stage: "vertex",
			code: "#version 300 es\nvoid main() { gl_Position = vec4(0.0); }",
		},
		{
			backend: "webgl",
			language: "glsl",
			stage: "fragment-depth",
			code: "#version 300 es\nprecision highp float;\nvoid main() {}",
		},
	],
});

const depthProgram = webglMaterial.resolveWebGLDepthPrepassProgram("single");
```

#### Custom Shader Material Configuration (WebGPU)

```ts
import { AlphaMode, ShaderMaterial } from "../src/materials";

const webgpuMaterial = new ShaderMaterial({
	alphaMode: AlphaMode.Mask,
	vertexEntryPoint: "vsMain",
	depthFragmentEntryPoint: "fsDepthMask",
	depthFragmentCode: `
@fragment
fn fsDepthMask() {
	// alpha discard logic
}
`,
});
```

#### Verification Commands

```bash
# Software Backend tests
bun tests/static/software/test_software_early_z_prepass.mjs

# WebGL Backend tests
bun tests/static/webgl/test_webgl_backend_material_early_z.mjs

# WebGPU Backend tests
bun tests/static/webgpu/test_webgpu_bridge_material_pipelines.mjs
bun tests/static/webgpu/test_webgpu_frame_executor_targets_msaa.mjs
```

### Projected decals

```ts
import { Decal, PBRMaterial, TextureLoader } from "ignis-renderer";

const material = new PBRMaterial({
	albedoMap: await TextureLoader.load("paint.png"),
	normalMap: await TextureLoader.load("paint-normal.png"),
	roughness: 0.8,
});

const decal = new Decal({
	material,
	receiverLayerMask: 1,
	priority: 10,
	opacity: 0.75,
	channelBlendModes: {
		baseColor: "lerp",
		normal: "normal",
		metalness: "disabled",
	},
});

decal.position.set(0, 1, -2);
decal.scale.set(2, 2, 0.5);
scene.add(decal);
```

## Diagnostics

### Order-independent transparency

- `webgpu-oit-disabled-mrt-unavailable`:
  emitted when OIT is requested but MRT targets are unavailable.
- `webgpu-oit-disabled-msaa`:
  emitted when OIT is requested with `sampleCount > 1`.
- `webgpu-oit-disabled-runtime`:
  emitted when runtime OIT resources or native copy capability are unavailable.
- `webgpu-oit-copy-scene-color-failed`:
  emitted when scene-color copy for OIT resolve fails.
- `webgl-oit-disabled-runtime`:
  emitted when OIT is requested but WebGL float color-buffer OIT targets are
  unavailable.

All warnings should be emitted via `warn once` behavior.

### Occlusion culling

- `{backend}-feature-occlusion-culling` must be emitted once when
  `enableOcclusionCulling` is requested on a backend whose
  `BackendCapabilities.occlusionCulling` is `false`.
- `webgpu-hiz-build-failed` may be emitted when the shared WebGPU frame Hi-Z
  build fails. Affected candidates must remain visible for that frame.
- `webgpu-occlusion-encode-failed` may be emitted when the WebGPU runtime cannot
  record the visibility compute pass.
- `webgpu-occlusion-readback-failed` may be emitted when an asynchronous
  readback cannot be mapped or consumed.

All runtime failures must fall back to visible candidates for the affected
frame or snapshot. Diagnostics should use warn-once behavior where repeated
frames would otherwise produce duplicate warnings.

### HDR rendering

- `display-hdr-unavailable`: an explicit HDR request cannot be activated.
- `display-hdr-configuration-failed`: the browser rejected or ignored the HDR
  canvas configuration.
- `postprocess-color-domain-undeclared-<id>`: an HDR custom pass has no color
  contract and is assumed to preserve its input domain.
- `postprocess-color-domain-mismatch-<id>`: a declared pass was skipped because
  its expected input domain did not match the current domain.
- Display fallback reasons are `backend-unsupported`,
  `display-not-hdr-capable`, `canvas-tone-mapping-unsupported`, and
  `hdr-context-configuration-failed`. Software Canvas 2D capability failure
  uses `canvas-hdr-output-unsupported`.

### Early-Z prepass

- **Missing Configurations**:
	- `ShaderMaterial.resolveWebGLDepthPrepassProgram()` must return `null` when WebGL vertex or `fragment-depth` source is missing.
	- A WebGL `ShaderMaterial` using `AlphaMode.Mask` without a WebGL `fragment-depth` chunk should warn once and skip the Early-Z prepass for that material.
	- A WebGPU `ShaderMaterial` using `AlphaMode.Mask` without `depthFragmentCode` or `depthFragmentEntryPoint` must log a warning once and skip the Early-Z prepass.
- **Shader compilation failures**:
	- Custom WebGL depth prepass compile failures should throw in strict shader mode. In warn mode, the backend should warn once and skip the Early-Z prepass for that material.
	- WebGPU shader compile errors during Early-Z prepass setup should skip that material prepass path and keep color-pass rendering available.
- **Guards**: Non-finite or invalid rendering inputs must follow existing backend guards and must not introduce new crash paths.

### Projected decals

`Decal` instances with `material === null` must be skipped.

`Decal` instances using `ShaderMaterial` must be skipped.

`Decal` instances with a non-invertible world matrix must be skipped during
prepared-scene construction.

WebGPU adapters that cannot provide the required decal sampled textures must
fail device negotiation with the standard WebGPU minimum-limit diagnostic.

## Verification

```bash
bun tests/run_all.mjs tests/static/webgpu
bun tests/run_all.mjs tests/static/webgl
bunx tsc --noEmit
```

## Related Documents

- [Rendering architecture](../architecture/rendering.md)
- [Materials contract](materials.md)
- [WebGPU contract](webgpu.md)
- [WebGL contract](webgl.md)
