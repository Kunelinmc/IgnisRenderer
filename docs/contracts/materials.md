# Materials Contract

This document defines texture formats, PBR material extensions, and depth-write behavior shared by rendering backends.

## Contract

### Render-state revisions

- `Material.revision` must be a monotonic readonly revision of every public
  field that can change render output or pipeline selection.
- `Material.contentRevision` must advance whenever any material revision
  advances so on-demand renderers can detect changes without retaining every
  live material.
- Direct assignment to existing public material properties must remain
  supported. Nested color values and texture sampling transforms must be
  checked once per unique material before clean-frame rejection.
- Cached prepared-scene construction must refresh each unique material revision
  once per preparation frame and share it between submission bindings and cache
  signatures. A new preparation frame or cache reset must discard those revision
  observations. Uncached construction must still refresh material revisions.
- `Texture.samplingRevision` must advance when UV transforms, wrapping,
  filtering, rotation, format, or color-space sampling semantics change. Pixel upload
  changes must continue to use `Texture.version`.
- Backend material snapshots must be reusable until the material revision,
  referenced texture identity, texture upload version, or texture sampling
  revision changes.
- Canonical material revisions must also include render-affecting state owned
  privately by a material subclass. `ShaderMaterial` custom texture bindings must
  observe bound texture identity, upload version, and sampling revision without
  requiring the application to re-register an unchanged binding. These changes
  must participate in on-demand scheduling and packet and decal dirty coverage.

### Texture formats

- `Texture` must expose `format: TextureFormat`.
- Every public `*Texture` constructor must accept exactly one parameter object.
- `Texture` must accept `TextureParams`; positional `data`, `width`, `height`,
  and `colorSpace` arguments must not be supported.
- `CanvasTextureParams` and `VideoTextureParams` must include their required
  `context` and `video` sources respectively.
- Texture parameter types should extend `TextureBaseParams` for shared
  `colorSpace`, `label`, and `usageHint` metadata.
- `Texture` may be constructed from `TextureParams`:

```ts
const texture = new Texture({
	width: 2,
	height: 1,
	format: TextureFormat.R8Unorm,
	colorSpace: "Linear",
	levels: [
		{
			data: new Uint8Array([32, 200]),
			width: 2,
			height: 1,
		},
	],
});
```

- `TextureMipLevel` must describe `data`, `width`, `height`, and optional row layout
  fields for raw or compressed payloads.
- `TextureFormat` must use WebGPU-compatible format names such as `r8unorm`,
  `rg8unorm`, `rgba8unorm-srgb`, `rgba16float`, `depth32float`, and
  `bc1-rgba-unorm`.
- `TextureData` must be `Uint8Array`, `Uint8ClampedArray`, or `Float32Array`.
  Wider integer, depth, stencil, packed, and compressed payloads must be supplied as
  raw bytes through `Uint8Array`.
- Raw textures must use `data` or `levels` as their CPU-side source.
- `CanvasTexture` and `VideoTexture` must use their external canvas or video as
  the source of truth. They must keep `data`, `levels`, and `mipmaps` empty
  instead of retaining a duplicate CPU pixel buffer.
- `Texture.sourceKind` must expose the immutable source lifecycle
  classification `"static"` or `"dynamic"`. Plain `Texture` instances must be
  `"static"`; `CanvasTexture` and `VideoTexture` instances must be `"dynamic"`.
- `Texture.getUploadSource()` must be the polymorphic backend upload contract:
  raw textures return `TextureData`, while dynamic textures return a
  `TexImageSource`. Backends must not detect source ownership through concrete
  texture subclasses.
- `Texture.readPixelData()` may read external-source pixels on demand for CPU
  consumers. Backends should cache that result by texture `version` when
  repeated CPU sampling is required.
- External-source changes must advance the global texture content revision.
  Each `Renderer` must compare that revision independently so dynamic updates
  do not require a global collection of live `Texture` instances.
- `Texture.dispose()` must notify every registered backend cleanup observer.
  Registered observers must release their texture-specific native resources
  without storing backend handles on `Texture`.
- Backend cleanup observers registered on `Texture` must not keep either the
  texture or a destroyed backend runtime alive.
- `IRenderTexture.format` should report the actual backend format. If a backend
  cannot provide the requested format, `IRenderTexture.requestedFormat` should keep
  the requested value and `IRenderTexture.formatFallbackReason` should describe the
  fallback.
- Persistent custom render targets are stricter than general texture uploads:
  their actual attachment format must equal the requested format, and
  unsupported backend formats must throw instead of falling back.
- `RenderTargetReadbackResult` must report bytes in the attachment's actual
  format and must expose the backend-native row `origin`.
- GPU color-texture sampling must return linear RGB through the sampled texture
  format. WebGPU and WebGL shaders must not receive per-texture linear flags or
  perform source sRGB decoding. Alpha must remain linear.
- Byte color textures without an explicit format must default to
  `rgba8unorm-srgb` for `colorSpace: "sRGB"`, and `rgba8unorm` for linear data.
  Floating-point sources must default to linear HDR semantics. When only an
  explicit format is supplied, its sRGB, floating-point, or linear encoding must
  determine the default `colorSpace`.
- Upload resolution must choose the sRGB or non-sRGB member of a compatible
  format pair from the source `colorSpace` and the sampling role. Non-color
  material slots must use non-sRGB formats. A texture used in both roles must
  retain separate compatible native views or resources.
- Resolved texture formats and caches must depend on the current format and
  source encoding, regardless of whether a format was supplied or inferred.
- An sRGB source without a compatible sRGB sampling format must be rejected
  rather than sampled as linear encoded values. Unsupported sRGB formats must
  not silently fall back to non-sRGB formats. This requirement also applies to
  compressed textures, external native resources, and dynamic image uploads.
- Material, particle, decal, background, and IBL texture consumers must share
  this sampling contract. Software sampling must preserve equivalent decoding
  behavior in its CPU sampling and shading implementation.
- `ShaderMaterialTextureBinding.linear` must select native sampling encoding:
  `true` uses the non-sRGB format, and `false` uses the sRGB format. An omitted
  override must follow the texture source encoding. Generated sampling helpers
  must return native samples without decoding or shader specialization by encoding.
- Compressed texture formats must accept pre-compressed block data only. This
  contract does not require runtime transcoding from image formats into BC, ETC2,
  EAC, or ASTC blocks.

### PBR extensions

- GPU PBR shading must distinguish the rasterized triangle's geometric normal
  from interpolated and normal-mapped shading normals. The geometric normal
  must be derived from world-position derivatives and oriented to the visible
  side, with a finite fallback for degenerate derivatives.
  Derivative normalization must be independent of world area per pixel.
- Base and clearcoat shading normals must keep their reflection directions in
  the visible geometric hemisphere. Invalid normals must be corrected
  continuously at the numerical reflection horizon; valid normals outside that
  tolerance must retain their directions. Correction must not alter raster
  culling or front-face identity. Both authored normal-map frames must be built
  before correcting base and clearcoat normals independently.
- Anisotropic environment reflection directions must also respect the geometric
  hemisphere after bending. WebGPU deferred lighting may reconstruct this
  hemisphere from locally planar depth. Missing or discontinuous neighboring
  depth must select the corrected isotropic direction rather than an unreliable
  anisotropic ray. Reconstruction must use the validated neighbors explicitly;
  full-screen derivatives may include unrelated surfaces across a diagonal.
- WebGPU must apply the same correction before base and extended G-buffer
  encoding and in forward and transmission-capture shading. Deferred decoding
  must prevent quantization from producing a back-facing shading normal.
  Normal comparisons after G-buffer decoding may include its encoding error.

#### Resolved PBR features

- The material subsystem must resolve feature and texture flags directly from
  authored fields. It must not construct or expose numeric feature masks,
  texture masks, or bit assignments.
- `resolvePBRMaterialFeatures` is an internal material-subsystem operation.
  It must return named lobe flags, texture-presence flags, and effective
  texture flags. Backends must consume these semantics and must not duplicate
  the material activation rules.
- Resolution must reflect direct factor, nested color, and texture mutations
  without requiring a dirty flag or explicit update call. Published feature
  snapshots must retain their values after later authoring mutations. The
  resolver should reuse immutable snapshots when resolved semantics are unchanged.
- Texture presence must report non-null references independently of whether
  the parent lobe contributes. Effective texture flags must exclude maps whose
  parent lobe is inactive.
- Specular customization must be active when the clamped factor or color
  differs from its neutral default by more than `0.000001`, or when a
  non-annihilated specular texture can affect the result. Effective specular
  maps require both the resolved factor and maximum color component to exceed
  that epsilon. This flag must not represent the base PBR Fresnel lobe.
- Clearcoat, sheen, iridescence, anisotropy, and transmission must be active
  only when their clamped lobe multiplier exceeds `0.000001`.
- Generic materials using PBR shading must use the same resolver and neutral
  defaults as `PBRMaterial`. The legacy `specularColorFactor` alias may supply
  the specular color when `specularColor` is absent.
- Backends may encode resolved semantics in private shader flags. Such packing
  must not establish a material-owned shader ABI.

- `GLTFLoader.parseMaterials` must parse `KHR_materials_iridescence` and
  `KHR_materials_anisotropy` on PBR materials.
- `iridescenceFactor` must default to `0.0` and must be multiplied by the red
  channel of `iridescenceTexture`.
- `iridescenceIor` must default to `1.3`.
- `iridescenceThicknessMinimum` must default to `100.0` nanometers.
- `iridescenceThicknessMaximum` must default to `400.0` nanometers.
- `iridescenceThicknessTexture` must use the green channel. Missing thickness
  textures must behave as a sampled value of `1.0`.
- `iridescenceTexture` and `iridescenceThicknessTexture` must use linear
  non-color texture semantics.
- `PBREvaluator` must expose `iridescence`, `iridescenceIor`, and
  `iridescenceThickness` in `PBRSurfaceProperties`.
- `SoftwareBackend`, `WebGPUBackend`, and `WebGLBackend` built-in PBR lighting
  must apply iridescence through the base-layer Fresnel term.
- `anisotropyStrength` must default to `0.0`, must be clamped to the range
  `[0.0, 1.0]`, and must be multiplied by the blue channel of
  `anisotropyTexture`.
- `anisotropyRotation` must default to `0.0` radians and must rotate the
  anisotropy direction in tangent space.
- `anisotropyTexture` must use linear non-color texture semantics. Its red and
  green channels must encode a tangent-space direction mapped from `[0.0, 1.0]`
  to `[-1.0, 1.0]`.
- `PBREvaluator` must expose `anisotropyStrength`, `anisotropyTangent`, and
  `anisotropyBitangent` in `PBRSurfaceProperties`.
- `SoftwareBackend`, `WebGPUBackend`, and `WebGLBackend` built-in PBR lighting
  must use an anisotropic GGX base specular lobe when `anisotropyStrength` is
  greater than `0.0`.
- `WebGPUBackend` deferred lighting must encode the resolved anisotropy tangent
  and strength in its G-buffer extension payload and must use them for direct
  anisotropic specular and environment reflection direction.
- The WebGPU compact deferred payload must preserve `PhongMaterial.shininess`
  as an unclamped FP16 value. The same channel may store the normalized PBR
  specular factor only when the packed shading model is `PBR`.
- WebGPU deferred material feature bits must be derived from
  `resolvePBRMaterialFeatures` semantics. Missing feature bits must select the
  documented neutral defaults and must not trigger extension texture loads.

### Depth writes

- `MaterialParams.depthWrite?: boolean`
	- Input contract: accepts `true`, `false`, or `undefined`.
	- Default contract: `undefined` must be treated as `true`.
	- Behavior contract:
		- When `true`, opaque materials must keep existing depth-write behavior.
		- When `false`, opaque materials must still run the normal depth test but
		  must not update the main scene depth buffer.
		- Transparent materials must remain read-only for depth writes regardless
		  of this flag.
- `materialWritesDepth(material: Material): boolean`
	- Input contract: accepts a `Material` instance.
	- Output contract: returns `false` only when `material.depthWrite === false`.
	- Constraint contract: render backends must use this helper or equivalent
	  logic when selecting depth-write render state.
- Early Z pre-pass contract:
	- Materials with `depthWrite === false` must not participate in depth-only
	  pre-passes.
	- Color draws for these materials must not be promoted to read-only
	  `early-z-color` pipelines because they were never pre-passed.
- WebGPU deferred lighting contract:
	- Materials with `depthWrite === false` must not enter the deferred G-buffer
	  path.
	- These materials must use the legacy forward fallback after deferred
	  lighting resolve when MRT scene targets are available.
	- Transparent and transmission materials must stay outside the deferred
	  opaque G-buffer path regardless of `depthWrite`.
- Cache contract:
	- Material signatures and backend pipeline keys must include `depthWrite`.

### Presentation alpha

- Main scene color alpha represents presentation coverage, not stored material
  opacity.
- Built-in opaque and mask materials must write coverage `1.0`, even when a
  source texture or factor contains another alpha value. Blend materials must
  use their resolved opacity as source-over coverage.
- `ShaderMaterial.transparentOutputCompatible` owns the explicit custom-shader
  opt-in defined by the shader contract. Opaque rendering remains unaffected
  when transparent presentation is not requested.

## Usage

### Texture formats

Create a single-channel material data texture:

```ts
const roughnessMask = new Texture({
	width: 4,
	height: 4,
	format: TextureFormat.R8Unorm,
	colorSpace: "Linear",
	data: new Uint8Array(4 * 4),
});
```

Create an explicit hardware-sRGB texture:

```ts
const baseColor = new Texture({
	width: 1,
	height: 1,
	format: TextureFormat.RGBA8UnormSrgb,
	colorSpace: "sRGB",
	data: new Uint8Array([128, 64, 32, 255]),
});
```

Read a narrow WebGPU texture as RGBA floats:

```ts
const readback = await runtime.readTexture({
	texture,
	width: 2,
	height: 1,
	format: TextureFormat.R8Unorm,
});
const rgba = readback.toRGBAFloat32();
```

### PBR extensions

```ts
import { GLTFLoader } from "../src/loaders/GLTFLoader";
import { Texture } from "../src/core/Texture";

const loader = new GLTFLoader();
const iridescenceTexture = new Texture({
	data: new Uint8ClampedArray([255, 0, 0, 255]),
	width: 1,
	height: 1,
});
const thicknessTexture = new Texture({
	data: new Uint8ClampedArray([0, 128, 0, 255]),
	width: 1,
	height: 1,
});

const [material] = loader.parseMaterials(
	{
		materials: [
			{
				pbrMetallicRoughness: {},
				extensions: {
					KHR_materials_iridescence: {
						iridescenceFactor: 1.0,
						iridescenceTexture: { index: 0 },
						iridescenceIor: 1.3,
						iridescenceThicknessMinimum: 100.0,
						iridescenceThicknessMaximum: 400.0,
						iridescenceThicknessTexture: { index: 1 },
					},
				},
			},
		],
	},
	[iridescenceTexture, thicknessTexture]
);

console.assert(material.iridescenceMap?.colorSpace === "Linear");
console.assert(material.iridescenceThicknessMap?.colorSpace === "Linear");
```

```ts
import { GLTFLoader } from "../src/loaders/GLTFLoader";
import { Texture } from "../src/core/Texture";

const loader = new GLTFLoader();
const anisotropyTexture = new Texture({
	data: new Uint8ClampedArray([255, 128, 64, 255]),
	width: 1,
	height: 1,
});

const [material] = loader.parseMaterials(
	{
		materials: [
			{
				pbrMetallicRoughness: {},
				extensions: {
					KHR_materials_anisotropy: {
						anisotropyStrength: 0.8,
						anisotropyRotation: Math.PI / 2,
						anisotropyTexture: { index: 0, texCoord: 1 },
					},
				},
			},
		],
	},
	[anisotropyTexture]
);

console.assert(material.anisotropyStrength === 0.8);
console.assert(material.anisotropyMap?.colorSpace === "Linear");
```

### Depth writes

```ts
import { ShaderMaterial } from "../src/materials";

const material = new ShaderMaterial({
	depthWrite: false,
});

console.assert(material.depthWrite === false);
```

```bash
bun tests/static/shaders/test_shader_material.mjs
bun tests/static/webgpu/test_webgpu_bridge_material_pipelines.mjs
bun tests/static/software/test_software_early_z_prepass.mjs
```

## Diagnostics

### Texture formats

- `TextureFormatInfo` lookup must throw when a format string is unknown.
- `toRGBAFloat32()` must throw for integer, depth/stencil, compressed, and unsupported
  packed readback formats.
- Backends should warn once when a requested texture format falls back to another
  actual format.
- Custom render target allocation must throw instead of applying a format
  fallback.
- Compressed uploads must provide block payloads matching the requested block layout.
  Missing or undersized payloads are treated as zero-filled or truncated raw bytes.

### PBR extensions

- If `iridescenceTexture.index` or `iridescenceThicknessTexture.index` does not
  resolve to a loaded texture, the corresponding `PBRMaterial` texture field
  must remain `null`.
- If `iridescenceFactor` is `0.0`, render backends must produce the same result
  as the base PBR material.
- If `anisotropyTexture.index` does not resolve to a loaded texture, the
  corresponding `PBRMaterial.anisotropyMap` field must remain `null`.
- If `anisotropyStrength` is `0.0`, render backends must produce the same result
  as the base isotropic PBR material.
- If an anisotropy direction texture texel encodes a zero-length red/green
  direction, render backends must use the unrotated tangent direction
  `(1.0, 0.0)`.
- If `WebGLBackend` receives both `anisotropyMap` and
  `iridescenceThicknessMap` on the same material, it must disable
  `anisotropyMap` sampling and emit a once-per-key warning because the built-in
  scene shader preserves the common 16 `sampler2D` fragment limit.
- If a backend does not support a custom shader material path, this contract does
  not require it to emulate `KHR_materials_iridescence` or
  `KHR_materials_anisotropy` inside user-authored shader code.

### Depth writes

- No new warning key is required.
- Invalid or missing shader source must continue following existing
  `ShaderMaterial` diagnostics.
- Non-finite render inputs must continue following existing backend guards.

## Verification

```bash
bun tests/static/webgpu/test_webgpu_bridge_material_pipelines.mjs
bun tests/static/webgl/test_webgl_backend_material_early_z.mjs
bunx tsc --noEmit
```

## Related Documents

- [Rendering contract](rendering.md)
- [Lighting contract](lighting.md)
- [Shader contract](shaders.md)
