# Deferred Decals Contract

## Scope

This document defines the first-version texture decal contract for
`IgnisRenderer`.

The v1 implementation is WebGPU-only and applies box-projected decals to the
deferred G-buffer after opaque geometry and before deferred lighting. Software
and WebGL backends must skip decals without changing frame output.

## Background

Decals reuse existing `Material` classes instead of introducing a dedicated
decal material type. Decal-specific projector and blending state lives on
`Decal`.

The deferred decal pass snapshots the current G-buffer before each decal draw,
samples that snapshot, applies the decal blend rules, then writes the updated
G-buffer. This preserves `Decal.priority` and scene traversal order for
overlapping decals.

## API/Contract

`Node.renderLayers` must be an unsigned bitmask. The default value is `1`
(layer bit 0).

`Decal.material` may reference any existing `Material`. `ShaderMaterial` is not
executed by v1 decal rendering and must be skipped.

`Decal.receiverLayerMask` must match receiver pixels through
`MeshInstance.renderLayers & Decal.receiverLayerMask`. WebGPU v1 stores receiver
layer masks in `gMaterialExt3.w`, an `rgba16float` channel, so only bits `0..10`
are guaranteed exact in the shader. Higher bits may be used by CPU-side APIs but
must not be relied on for WebGPU v1 decal receiver tests.

`Decal.priority` must sort decals in ascending order. Decals with equal priority
must retain scene traversal order.

`Decal.channelBlendModes` may set per-channel blend behavior. Supported modes
are `disabled`, `lerp`, `replace`, `multiply`, `add`, and `normal`.

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
	clearcoatNormal: "normal",
	roughness: "lerp"
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

The WebGPU `ModelUniforms` layout adds `nodeRenderLayers` before
`textureTransformA`. Custom WebGPU shaders that redeclare `ModelUniforms` must
add the same field to keep texture transform offsets valid.

V1 decals do not affect transparent objects, particles, shadows, reflection
captures, or non-WebGPU backends.
