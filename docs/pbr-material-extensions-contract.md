# PBR Material Extensions Contract

## Scope

This document defines the engine contract for glTF PBR material extensions that
are represented by `PBRMaterial` fields and consumed by built-in lighting
pipelines.

## Background

glTF material extensions extend the core metallic-roughness material model.
IgnisRenderer must parse supported extension payloads into backend-agnostic
`PBRMaterial` fields before render backends evaluate lighting.

## API/Contract

- `GLTFLoader.parseMaterials` must parse `KHR_materials_iridescence` on PBR
  materials.
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

## Usage

```ts
import { GLTFLoader } from "../src/loaders/GLTFLoader";
import { Texture } from "../src/core/Texture";

const loader = new GLTFLoader();
const iridescenceTexture = new Texture(
	new Uint8ClampedArray([255, 0, 0, 255]),
	1,
	1
);
const thicknessTexture = new Texture(
	new Uint8ClampedArray([0, 128, 0, 255]),
	1,
	1
);

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

## Errors & Diagnostics

- If `iridescenceTexture.index` or `iridescenceThicknessTexture.index` does not
  resolve to a loaded texture, the corresponding `PBRMaterial` texture field
  must remain `null`.
- If `iridescenceFactor` is `0.0`, render backends must produce the same result
  as the base PBR material.
- If a backend does not support a custom shader material path, this contract does
  not require it to emulate `KHR_materials_iridescence` inside user-authored
  shader code.

## Compatibility / Breaking Changes

This change is additive. Existing PBR materials without
`KHR_materials_iridescence` must preserve previous rendering behavior.
