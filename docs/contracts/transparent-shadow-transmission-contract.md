# Transparent Shadow Transmission Contract

## Scope

This document defines direct-light attenuation through transparent shadow
casters for Software, WebGPU, and WebGL rendering backends.

## Background

Opaque shadow maps store blocker depth and produce scalar visibility. Colored
glass requires an additional transmittance term so direct light that is not
blocked by opaque geometry may be filtered by transparent material absorption.

## API/Contract

- `PreparedScene.shadowTransmitterPackets` must contain transparent primitives
  with `castShadows = true`.
- PBR transparent shadow transmission must use `transmissionFactor`,
  `albedo`, `ior`, `thicknessFactor`, `attenuationColor`, and
  `attenuationDistance`.
- The transmittance model must use Beer-Lambert absorption:
  `attenuationColor ^ (thicknessFactor / attenuationDistance)`.
- `attenuationDistance <= 0`, `attenuationDistance = Infinity`, or
  `thicknessFactor <= 0` must disable volume absorption.
- PBR surface interface loss should use normal-incidence Fresnel derived from
  `ior`.
- Non-PBR and alpha-blended materials may use opacity-weighted color filtering
  as a compatibility fallback.
- Shadow visibility used by lighting must be RGB transmittance. Fully opaque
  blockers must contribute `vec3(0)`, unblocked white light must contribute
  `vec3(1)`, and transparent blockers must multiply the unblocked sample by
  material transmittance.
- Raster backends may ignore refraction and caustic focusing in this contract.
  Those effects require a separate light transport pass.

## Usage

```ts
import { PBRMaterial } from "../src/materials/PBRMaterial";

const redGlass = new PBRMaterial({
	albedo: { r: 255, g: 255, b: 255 },
	transmissionFactor: 1,
	ior: 1.5,
	thicknessFactor: 0.2,
	attenuationDistance: 0.4,
	attenuationColor: { r: 255, g: 32, b: 32 },
});

meshPrimitive.material = redGlass;
meshPrimitive.castShadows = true;
```

## Errors & Diagnostics

- If the backend cannot allocate a shadow transmittance atlas, it should fall
  back to white transmittance for the frame.
- If WebGL has insufficient texture units for the transmittance atlas, it must
  keep scalar shadows enabled and sample white transmittance.
- If `transmissionFactor` is `0`, PBR materials must not produce colored
  transparent shadows unless alpha blending is active.

## Compatibility / Breaking Changes

The feature is additive. Existing opaque shadow behavior must remain unchanged.
