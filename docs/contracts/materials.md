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
- `Texture.samplingRevision` must advance when UV transforms, wrapping,
  filtering, rotation, or color-space sampling semantics change. Pixel upload
  changes must continue to use `Texture.version`.
- Backend material snapshots must be reusable until the material revision,
  referenced texture identity, texture upload version, or texture sampling
  revision changes.

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
- sRGB textures should prefer sRGB GPU formats. Shader-side sRGB decode must be
  skipped when the sampled backend texture format performs hardware sRGB decode.
- Compressed texture formats must accept pre-compressed block data only. This
  contract does not require runtime transcoding from image formats into BC, ETC2,
  EAC, or ASTC blocks.

### PBR extensions

#### Stable feature masks

- `PBRMaterialFeature` must define append-only unsigned feature bits with the
  following fixed assignments: `BASE_COLOR_MAP` bit 0,
  `METALLIC_ROUGHNESS_MAP` bit 1, `NORMAL_MAP` bit 2, `OCCLUSION_MAP` bit 3,
  `SPECULAR` bit 4, `CLEARCOAT` bit 5, `SHEEN` bit 6, `IRIDESCENCE` bit 7,
  `ANISOTROPY` bit 8, and `TRANSMISSION` bit 9. Existing assignments must not
  be reordered or reused.
- `PBRMaterialTextureFeature` must define append-only unsigned texture-presence
  bits with the following fixed assignments: `BASE_COLOR_MAP` bit 0,
  `METALLIC_ROUGHNESS_MAP` bit 1, `NORMAL_MAP` bit 2, `EMISSIVE_MAP` bit 3,
  `OCCLUSION_MAP` bit 4, `SPECULAR_MAP` bit 5, `SPECULAR_COLOR_MAP` bit 6,
  `CLEARCOAT_MAP` bit 7, `CLEARCOAT_ROUGHNESS_MAP` bit 8,
  `CLEARCOAT_NORMAL_MAP` bit 9, `SHEEN_COLOR_MAP` bit 10,
  `SHEEN_ROUGHNESS_MAP` bit 11, `TRANSMISSION_MAP` bit 12,
  `THICKNESS_MAP` bit 13, `IRIDESCENCE_MAP` bit 14,
  `IRIDESCENCE_THICKNESS_MAP` bit 15, and `ANISOTROPY_MAP` bit 16.
- `PBRMaterial.featureMask` and `PBRMaterial.textureMask` must be readonly
  computed properties. They must reflect direct material-field mutation without
  requiring a dirty flag or explicit update call.
- `textureMask` must report non-null texture references independently of whether
  the parent material lobe currently contributes to the resolved result.
- The base-color, metallic-roughness, normal, and occlusion feature bits must
  mirror their texture-presence bits. `SPECULAR` must identify a resolved
  `KHR_materials_specular` customization rather than the base PBR Fresnel lobe.
  It must be enabled when the resolved factor or color differs from its neutral
  default, or when a non-annihilated specular texture can affect the result.
- `CLEARCOAT`, `SHEEN`, `IRIDESCENCE`, `ANISOTROPY`, and `TRANSMISSION` must be
  enabled only when the resolved lobe multiplier is greater than the material
  epsilon. A parent feature with a zero multiplier must remain disabled even
  when one of its texture-presence bits is set.

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
- WebGPU deferred material feature bits must be derived from resolved material
  values. Missing feature bits must select the documented neutral defaults and
  must not trigger extension texture loads.

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
