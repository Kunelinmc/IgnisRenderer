# WebGPU Deferred Lighting Contract
## Scope
This document defines the v1 `WebGPUBackend` deferred lighting contract for
opaque and mask scene rendering.

## Background
`WebGPUBackend` may split its internal `main-opaque` implementation into
background, G-buffer, deferred lighting, and forward fallback GPU passes. This
does not add a global renderer frame-pass stage and does not affect
`SoftwareBackend` or `WebGLBackend` pass graphs.

## API/Contract
- Public control contract:
	- `WebGPUBackendOptions.enableDeferredLighting` must default to `true`.
	- When `enableDeferredLighting === false`, `WebGPUBackend` must not run the
	  deferred lighting resolve and must route opaque materials through the
	  legacy MRT forward path when MRT is available.
	- When `enableDeferredLighting !== false`, `WebGPUBackend` must attempt to
	  enable deferred lighting and must warn once if any runtime requirement
	  prevents correct activation.
	- `WebGPUBackend.isDeferredLightingEnabled()` must return the configured
	  public switch value.
- Runtime gating contract:
	- Deferred lighting must require `sampleCount === 1`.
	- Deferred lighting must require MRT scene targets.
	- Deferred lighting must require `maxColorAttachments >= 7`.
	- Deferred lighting must require
	  `maxColorAttachmentBytesPerSample >= 56`.
	- Deferred lighting must require
	  `maxStorageTexturesPerShaderStage >= 4`.
	- If any requirement is not met, `WebGPUBackend` must use the legacy MRT
	  forward path and must warn once.
- Pass ordering contract:
	- `main-opaque` must render environment/background into `sceneColorMain`
	  before lighting resolve when background rendering is enabled.
	- Builtin opaque and mask deferred materials must render surface payloads
	  into the G-buffer.
	- Deferred lighting must run as a fullscreen render pass that reads the
	  G-buffer, shadow data, environment data, and clustered-light buffers, then
	  writes `sceneColorMain`.
	- Deferred lighting must discard pixels with `gMotionDepth.z <= 0`, preserving
	  prior background color.
	- Non-deferred opaque fallback materials must render after deferred lighting
	  through the legacy MRT forward shader.
	- Transparent, OIT, transmission, and particles must render after opaque
	  lighting resolve through existing forward paths.
- G-buffer contract:
	- Color MRTs must be:
	  `gAlbedoAlpha`, `gNormalRoughMetal`, `gEmissiveOcclusion`,
	  `gMotionDepth`, `gSpecular`, `gCoatSheen`, and `gSheenReflectance`.
	- Deferred storage payload textures must be:
	  `gMaterialExt0`, `gMaterialExt1`, `gMaterialExt2`, and
	  `gMaterialExt3`.
	- `gMotionDepth.w` must store the material shading model.
	- `gMaterialExt3.xy` must store the encoded world-space anisotropy tangent,
	  `gMaterialExt3.z` must store the resolved anisotropy strength, and
	  `gMaterialExt3.w` is reserved.
	- The deferred lighting shader must branch on `PBR`, `Phong`, `Flat`, and
	  `Unlit` shading models inside the same fullscreen pass.
	- Opaque and mask `PBRMaterial` instances with `anisotropyStrength > 0.0`
	  or `anisotropyMap` may enter deferred lighting when all runtime gates pass.
	- `transmissionFactor > 0.0`, `AlphaMode.Blend`, OIT, and transparent
	  particles must remain on forward transparent paths.
- `ShaderMaterial` contract:
	- `ShaderTargetMode` must include `"deferred"`.
	- `ShaderStageKind` must include `"fragment-deferred"`.
	- `ShaderMaterialParams.deferredLighting` must opt a shader material into
	  deferred routing.
	- `ShaderMaterialParams.fragmentDeferredEntryPoint` must select the deferred
	  fragment entry point and must default to `fsMainDeferred`.
	- A `ShaderMaterial` must enter the G-buffer path only when
	  `deferredLighting === true` and a WebGPU deferred fragment chunk exists.
	- Non-opt-in `ShaderMaterial` instances must render after lighting through
	  the legacy MRT forward fallback.
- Camera reconstruction contract:
	- Perspective camera packing must keep `environmentBasisRight.w` as
	  `tanHalfFov` and `environmentBasisUp.w` as `aspect`.
	- Orthographic camera packing must store `halfWidth` in
	  `environmentBasisRight.w` and `halfHeight` in `environmentBasisUp.w`.
	- `environmentBasisBackward.w` must remain the orthographic flag.

## Usage
```ts
import { WebGPUBackend } from "../src/renderers/WebGPUBackend";

const backend = new WebGPUBackend({
	enableDeferredLighting: false,
});
```

```ts
import { ShaderMaterial } from "../src/materials/ShaderMaterial";

const material = new ShaderMaterial({
	deferredLighting: true,
	fragmentDeferredEntryPoint: "fsDeferred",
	chunks: [
		{
			language: "wgsl",
			stage: "fragment",
			mode: "deferred",
			code: deferredFragmentWGSL,
		},
	],
});
```

```bash
bun tests/static/shaders/test_shader_material.mjs
bun tests/static/webgpu/test_webgpu_bridge.mjs
bun tests/static/webgpu/test_webgpu_frame_executor_resilience.mjs
```

## Errors & Diagnostics
- `webgpu-deferred-disabled-mrt`:
  emitted when deferred lighting is requested but MRT scene targets are
  unavailable.
- `webgpu-deferred-disabled-msaa`:
  emitted when deferred lighting is unavailable because `sampleCount !== 1`.
- `webgpu-deferred-disabled-attachments`:
  emitted when `maxColorAttachments < 7`.
- `webgpu-deferred-disabled-bytes`:
  emitted when `maxColorAttachmentBytesPerSample < 56`.
- `webgpu-deferred-disabled-storage-textures`:
  emitted when `maxStorageTexturesPerShaderStage < 4`.
- `webgpu-deferred-runtime-fallback`:
  emitted when deferred frame target allocation fails after limits passed.
- Shader compile diagnostics for opt-in `ShaderMaterial` deferred chunks must
  follow the existing `ShaderMaterial` strict/warn/silent runtime behavior.

## Compatibility / Breaking Changes
No public renderer frame-pass stage is added. Builtin opaque and mask WebGPU
materials may use deferred lighting by default when runtime limits allow it.
Transparent, OIT, transmission, particles, SoftwareBackend, WebGLBackend, and
reflection probe capture keep their existing paths in v1.
