# EXR Loader Contract

## Scope

This document defines the contract for `EXRLoader` in `src/loaders/EXRLoader.ts`.

## Background

`EXRLoader` provides a native path for importing OpenEXR environment maps into
IgnisRenderer without converting them to Radiance `.hdr` first.

## API/Contract

- `new EXRLoader()` must create a loader with event behavior compatible with
  `Loader`.
- `EXRLoader.load(url, options?)` must fetch OpenEXR bytes from `url`, parse
  them with `parseAsync`, emit `load` on success, and return a `Texture` with
  `colorSpace === "HDR"`.
- `EXRLoader.load(url, options?)` must return a 1x1 black HDR fallback texture
  marked with `isLoadErrorFallback` when fetching or parsing fails.
- `EXRLoader.parse(buffer, options?)` must synchronously parse single-part
  scanline EXR files that use `NO_COMPRESSION`, `RLE_COMPRESSION`, or
  `PIZ_COMPRESSION`.
- `EXRLoader.parseAsync(buffer, options?)` must parse single-part scanline EXR
  files that use `NO_COMPRESSION`, `RLE_COMPRESSION`, `ZIPS_COMPRESSION`, or
  `ZIP_COMPRESSION`, or `PIZ_COMPRESSION`.
- `ZIPS_COMPRESSION` and `ZIP_COMPRESSION` must require runtime
  `DecompressionStream` support.
- The parser must reject multi-part, tiled, deep, PXR24, B44, B44A, DWAA, DWAB,
  and HTJ2K files with explicit diagnostics.
- The parser must require `R`, `G`, and `B` channels. It may use exact channel
  names or layer-suffixed names such as `beauty.R`.
- The parser must accept `HALF`, `FLOAT`, and `UINT` channel sample types with
  `xSampling === 1` and `ySampling === 1`.
- `EXRLoader.applyToEnvironment(target, texture, options?)` must assign
  `texture` to `environment.backgroundTexture` and `environment.iblTexture` by
  default.
- `EXRLoader.loadEnvironment(url, target, options?)` must load the texture and
  apply it to the provided `Environment` or scene-like target.
- `options.defaultAlpha` must define the alpha value used when the EXR has no
  `A` channel. The default value must be `1`.
- `options.background === false` must skip assignment to
  `environment.backgroundTexture`.
- `options.ibl === false` must skip assignment to `environment.iblTexture`.

## Usage

```ts
import { EXRLoader, Scene } from "../src";

const scene = new Scene();
const loader = new EXRLoader();

await loader.loadEnvironment("assets/studio.exr", scene.environment);
```

```ts
import { EXRLoader, Scene } from "../src";

const scene = new Scene();
const loader = new EXRLoader();
const texture = await loader.load("assets/studio.exr");

loader.applyToEnvironment(scene, texture, {
	background: true,
	ibl: true,
});
```

## Errors & Diagnostics

- `Invalid OpenEXR magic number` must be thrown when `buffer` does not start
  with the OpenEXR magic number.
- `Multi-part EXR files are not supported` must be thrown when the version field
  has the multi-part flag.
- `Deep EXR files are not supported` must be thrown when the version field has
  the deep-data flag.
- `Tiled EXR files are not supported` must be thrown when the version field has
  the tiled flag.
- `Unsupported EXR compression method` must be thrown for compression methods
  outside the supported contract.
- `ZIP/ZIPS compressed EXR data requires parseAsync() or load()` must be thrown
  when `parse` receives a compressed ZIP/ZIPS chunk.
- `ZIP/ZIPS EXR compression requires DecompressionStream support` must be thrown
  when `parseAsync` receives ZIP/ZIPS data in a runtime without
  `DecompressionStream`.
- `EXR image must provide R, G, and B channels` must be thrown when required
  color channels are absent.

## Compatibility / Breaking Changes

This change adds a new loader export (`EXRLoader`) and does not remove or change
existing loader APIs.
