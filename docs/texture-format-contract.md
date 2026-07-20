# Texture Format Contract

## Scope

This document defines the texture format contract for `Texture`, backend-owned
`IRenderTexture` resources, upload helpers, and compute readback.

## Background

IgnisRenderer previously inferred WebGPU upload formats from pixel array type and
effectively selected either `rgba8unorm` or `rgba16float`. That inference could not
represent sRGB storage, single-channel masks, RG textures, depth/stencil resources,
integer data textures, or pre-compressed block texture payloads.

## API/Contract

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
- `IRenderTexture.format` should report the actual backend format. If a backend
  cannot provide the requested format, `IRenderTexture.requestedFormat` should keep
  the requested value and `IRenderTexture.formatFallbackReason` should describe the
  fallback.
- sRGB textures should prefer sRGB GPU formats. Shader-side sRGB decode must be
  skipped when the sampled backend texture format performs hardware sRGB decode.
- Compressed texture formats must accept pre-compressed block data only. This
  contract does not require runtime transcoding from image formats into BC, ETC2,
  EAC, or ASTC blocks.

## Usage

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

## Errors & Diagnostics

- `TextureFormatInfo` lookup must throw when a format string is unknown.
- `toRGBAFloat32()` must throw for integer, depth/stencil, compressed, and unsupported
  packed readback formats.
- Backends should warn once when a requested texture format falls back to another
  actual format.
- Compressed uploads must provide block payloads matching the requested block layout.
  Missing or undersized payloads are treated as zero-filled or truncated raw bytes.

## Compatibility / Breaking Changes

All `*Texture` constructors now accept one parameter object. The positional
`Texture(data, width, height, colorSpace)`, `CanvasTexture(context, options)`,
and `VideoTexture(video, options)` forms have been removed. Callers must move
the source and options into `TextureParams`, `CanvasTextureParams`, or
`VideoTextureParams`.
