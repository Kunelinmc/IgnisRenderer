# Projected Decals Contract

## Scope

This document defines the projected texture decal contract for
`IgnisRenderer`.

WebGPU must apply box-projected decals to the deferred G-buffer after opaque
geometry and before deferred lighting. Software must apply projected decals
after base material evaluation and before lighting. WebGL does not currently
execute decals and must preserve the existing forward opaque output.

Backend-internal decal execution must not add renderer-level frame stages.

## Background

Decals reuse existing `Material` classes instead of introducing a dedicated
decal material type. Decal-specific projector and blending state lives on
`Decal`.

Supporting backends must preserve `Decal.priority` and scene traversal order for
overlapping decals. WebGPU may group contiguous compatible decals into an
ordered segment only when the observable result is unchanged. Decals that
cannot be grouped must use an ordered fallback path.

## API/Contract

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

`Decal.channelBlendModes` may set per-channel blend behavior. Supported modes
are `disabled`, `lerp`, `replace`, `multiply`, `add`, and `normal`.

`disabled` must preserve the receiver value. `lerp` and `replace` must perform
opacity-weighted replacement. `multiply` and `add` must perform weighted
component operations. `normal` must perform normalized direction blending for
`normal`, `clearcoatNormal`, and `anisotropy`; for scalar and color channels it
must behave as `lerp`. `multiply` and `add` must not modify direction channels.

Software decal coverage must be the product of `Decal.opacity`, material factor
alpha, base-color texture alpha, and edge fade. `AlphaMode.Mask` must reject
Software decal coverage when the resolved material alpha is below
`alphaCutoff`. A decal must not reclassify an opaque receiver into the
transparent pass.

Software normal and clearcoat-normal projection must use the inverse-transpose
projector normal transform. Software anisotropy tangents must use the projector
linear transform and must be orthogonalized against the resolved receiver
normal.

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

## Errors & Diagnostics

`Decal` instances with `material === null` must be skipped.

`Decal` instances using `ShaderMaterial` must be skipped.

`Decal` instances with a non-invertible world matrix must be skipped during
prepared-scene construction.

WebGPU devices that cannot bind the required decal sampled textures and samplers
must skip the decal pass for that frame.

## Compatibility / Breaking Changes

The WebGPU `ModelUniforms` layout includes `nodeRenderLayers` before
`textureTransformA`. Custom WebGPU shaders that redeclare `ModelUniforms` must
include the same field to keep texture transform offsets valid.

Decals do not affect transparent objects, particles, shadows, reflection
captures, or receivers that cannot provide the built-in material surface
contract. WebGL remains unsupported and must skip decals without changing frame
output.
