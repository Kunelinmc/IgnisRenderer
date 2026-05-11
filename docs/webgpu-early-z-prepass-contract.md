# WebGPU Early Z Pre-Pass Contract
## Scope
This document defines the `WebGPUBackend` Early Z depth pre-pass contract for
`main-opaque` rendering.

## Background
WebGPU opaque shading cost increases under overdraw. A depth pre-pass should
reduce unnecessary fragment shading by rejecting covered fragments before the
main color pass. When deferred lighting is active, the color pass is the
G-buffer geometry pass and the lighting resolve occurs after it.

## API/Contract
- `WebGPUBackendOptions.enableEarlyZPrepass?: boolean`
	- Input contract: accepts `true`, `false`, or `undefined`.
	- Default contract: `undefined` must be treated as `true`.
	- Behavior contract:
		- When `true`, `main-opaque` must run an Early Z depth pre-pass before
		  the color pass.
		- When `false`, `main-opaque` must keep the legacy single-pass depth
		  behavior.
- Pass scope contract:
	- Early Z pre-pass must apply only to `main-opaque`.
	- `main-transparent`, OIT, transmission, particles, and environment must not use
	  this pre-pass.
	- When deferred lighting is active, Early Z pre-pass must run before the
	  G-buffer geometry pass and must not run before the fullscreen lighting
	  pass.
- Pipeline contract:
	- Opaque non-mask pre-pass must use depth-only pipeline state with
	  `depthWriteEnabled = true` and `depthCompare = less`.
	- Opaque mask pre-pass must run alpha-test discard in a depth fragment entry
	  and must not write color targets.
	- Materials with `depthWrite === false` must be skipped by Early Z pre-pass.
	- Opaque color draws that were pre-passed must use read-only depth state with
	  `depthWriteEnabled = false` and `depthCompare = less-equal`.
	- Opaque color draws that were not pre-passed must keep legacy depth state
	  (`depthWriteEnabled = material.depthWrite !== false`,
	  `depthCompare = less`).
	- G-buffer draws that were pre-passed must use the same read-only
	  `early-z-color` depth state as legacy MRT color draws.
- `ShaderMaterial` contract:
	- `alphaMode = MASK` materials must provide explicit depth pre-pass fragment
	  contract via `depthFragmentCode` and `depthFragmentEntryPoint`.
	- If mask `ShaderMaterial` depth contract is missing, the engine must skip
	  Early Z pre-pass for that material and must log once warning.
- Incremental contract:
	- Incremental dirty-rect flow must continue to clear dirty depth region to
	  `1.0` before pre-pass.
	- Early Z pre-pass, G-buffer geometry pass, and legacy color pass must clip
	  to resolved dirty rects.

## Usage
```ts
import { WebGPUBackend } from "../src/renderers/WebGPUBackend";

const backend = new WebGPUBackend({
	enableEarlyZPrepass: true,
});
```

```ts
import { AlphaMode, ShaderMaterial } from "../src/materials";

const material = new ShaderMaterial({
	alphaMode: AlphaMode.Mask,
	vertexEntryPoint: "vsMain",
	depthFragmentEntryPoint: "fsDepthMask",
	depthFragmentCode: `
@fragment
fn fsDepthMask() {
	// alpha discard logic
}
`,
});
```

```bash
bun tests/test_webgpu_bridge.mjs
bun tests/test_webgpu_frame_executor_resilience.mjs
```

## Errors & Diagnostics
- Mask `ShaderMaterial` without depth pre-pass fragment contract must emit a
  once warning and must be skipped by Early Z pre-pass.
- Shader compile errors during Early Z pre-pass setup should skip that material
  pre-pass path and should keep color-pass rendering available.
- Non-finite or invalid draw inputs must continue following existing WebGPU
  draw guards and must not introduce new crash paths.

## Compatibility / Breaking Changes
This change is additive for builtin materials and default-enabled behavior.
Mask `ShaderMaterial` users should add explicit depth pre-pass fragment contract
to participate in Early Z pre-pass optimization.
