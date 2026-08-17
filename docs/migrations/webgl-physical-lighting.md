# WebGL Physical Lighting Migration

WebGL now requires a strict floating-point internal radiance pipeline and uses
the same physical light and glTF material semantics as the other renderer
paths. This is a breaking rendering change; constructor signatures are
unchanged.

## Runtime Requirements

WebGL2 runtimes must expose `EXT_color_buffer_float`, pass an `RGBA16F`
framebuffer-completeness probe, and support linear filtering for half- or
full-float textures. `WebGLBackend.initialize()` now throws
`WebGLCapabilityError` instead of rendering through an `RGBA8` fallback.

WebGL scene radiance remains linear HDR through presentation. Supported
Chromium runtimes may now select an `RGBA16F` Display-P3 drawing buffer;
unsupported runtimes retain exposure, ACES tone mapping, piecewise sRGB
encoding, and an 8-bit SDR drawing buffer. The backend-level
`profile.capabilities.displayHDR` flag reports the conditional implementation;
applications must inspect `Renderer.getDisplayOutputState()` for actual output.

## Light Units and World Scale

Treat one world unit as one meter. Author ranges, material thickness, and
attenuation distance in meters. Directional and ambient intensity is
lux-equivalent, point and spot intensity is candela-equivalent, and area-light
intensity is emitted-radiance equivalent. Existing scenes will usually require
retuning because point and spot lights now use inverse-square falloff.

The default intensity remains `1`. Remove compensating probe-capture intensity
multipliers and any application-side `PI` correction previously used for SH.

## Phong Appearance

Phong and Gouraud materials now use energy-normalized Blinn-Phong. Diffuse,
specular F0, and ambient colors are decoded from sRGB before lighting. The
default `PhongMaterial.specular` is `{ r: 56, g: 56, b: 56 }`, representing
approximately four-percent linear dielectric reflectance. Existing highlights
will be dimmer and more stable as shininess changes.

`PhongMaterial.ambient` remains supported as indirect diffuse reflectance. A
scene with no light, environment, probe, or emissive contribution now renders
black; the previous WebGL ambient floor is removed.

## PBR Textures and Sampler Limits

WebGL now evaluates public specular, clearcoat, sheen, transmission, thickness,
iridescence, and anisotropy maps using glTF/KHR channel and color-space rules.
Material combinations that exceed the device texture-unit limit throw
`material-texture-unit-overflow` during warmup or their first draw. Split
materials or reduce active maps instead of relying on silent sampler omission.

## Transmission Ordering

Frames containing mesh transmission use sorted transparent composition instead
of weighted OIT. Every transmissive packet snapshots the scene accumulated
behind it and performs depth-aware refraction. Screenshots involving mixed
alpha and transmission layers are expected to change.

## Related Documents

- [Rendering contract](../contracts/rendering.md)
- [Lighting contract](../contracts/lighting.md)
- [WebGL contract](../contracts/webgl.md)
