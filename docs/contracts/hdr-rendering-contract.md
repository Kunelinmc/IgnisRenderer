# HDR Rendering Contract

## Scope

This document defines internal HDR storage, Display HDR presentation, color
domains, capability negotiation, and SDR fallback behavior across rendering
backends.

## Background

IgnisRenderer performs lighting in a linear sRGB working space whose values may
exceed `1.0`. WebGPU preserves that range in floating-point scene and
post-process targets. Display HDR is an optional presentation mode; SDR remains
the compatibility default.

Display HDR uses relative headroom above SDR white. It does not define absolute
nits, HDR10 metadata, PQ, HLG, or automatic exposure.

## API/Contract

- `RendererOptions.displayOutput` may request `"sdr"`, `"auto"`, or `"hdr"`.
  The default must be `"sdr"`, with exposure `1.0` and HDR headroom `4.0`.
- Exposure must be finite and within `[0, 64]`. HDR headroom must be finite and
  within `[1, 16]`. Invalid values must throw `RangeError`.
- `Renderer.getDisplayOutputState()` must return `null` before initialization
  and the backend-resolved state afterwards.
- `Renderer.setDisplayOutput()` must merge partial settings with the current
  requested state and reconfigure presentation only at a frame boundary.
- `"auto"` must enable HDR only when the display reports high dynamic range,
  the canvas tone-mapping API is observable, and the requested configuration
  can be verified.
- An explicit `"hdr"` request must use the same requirements. If any
  requirement fails, rendering must continue in SDR and report a fallback
  reason.
- WebGPU HDR presentation must use an `rgba16float` canvas, Display-P3 color
  space, and extended canvas tone mapping.
- SDR presentation must use the user agent's preferred 8-bit canvas format,
  sRGB color space, and standard tone mapping when that member is supported.
- WebGL and Software are Display SDR backends. They must resolve `"auto"` to
  SDR without a warning and resolve explicit `"hdr"` to SDR with
  `backend-unsupported`.
- `WebGPUBackend` must preserve scene, post-process, and OIT radiance in
  `rgba16float` render targets.
- The logical post-process color domains are `scene-linear-hdr`,
  `display-linear`, and `display-encoded`.
- Built-in passes must declare their input and output domains. A declared pass
  whose input does not match the current domain must be skipped with a stable
  diagnostic.
- A custom pass without a color contract must be treated as domain-preserving.
  It must continue to run; HDR output must emit a stable warning.
- Presentation must complete any missing tone mapping, gamut conversion, or
  transfer encoding based on the final post-process domain.
- SDR mapping must use ACES fitted tone mapping and the piecewise sRGB transfer
  function. HDR mapping must apply exposure, a hue-preserving soft shoulder,
  linear-sRGB to linear Display-P3 conversion, and extended sRGB encoding
  without clamping encoded RGB to `1.0`.

## Usage

```ts
import { Renderer, WebGPUBackend } from "ignisrenderer";

const renderer = new Renderer({
	backend: new WebGPUBackend(),
	canvas,
	displayOutput: {
		mode: "auto",
		exposure: 1,
		hdrHeadroom: 4,
	},
});

await renderer.initialize();
console.log(renderer.getDisplayOutputState());

await renderer.setDisplayOutput({ mode: "hdr" });
```

## Errors & Diagnostics

- `display-hdr-unavailable`: an explicit HDR request cannot be activated.
- `display-hdr-configuration-failed`: the browser rejected or ignored the HDR
  canvas configuration.
- `postprocess-color-domain-undeclared-<id>`: an HDR custom pass has no color
  contract and is assumed to preserve its input domain.
- `postprocess-color-domain-mismatch-<id>`: a declared pass was skipped because
  its expected input domain did not match the current domain.
- Display fallback reasons are `backend-unsupported`,
  `display-not-hdr-capable`, `canvas-tone-mapping-unsupported`, and
  `hdr-context-configuration-failed`.

## Compatibility / Breaking Changes

- SDR remains the default display mode.
- Existing post-process pass IDs and enable/disable behavior remain available.
- Existing custom passes remain executable without a color contract.
- WebGPU SDR output changes from a gamma `2.2` approximation to exact piecewise
  sRGB encoding.
- WebGL floating-point internal targets remain governed by runtime extension
  support and do not imply Display HDR support.
