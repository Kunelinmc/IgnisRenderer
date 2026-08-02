# Material Depth Write Contract
## Scope
This document defines the `Material.depthWrite` contract for scene depth-buffer
writes across `SoftwareBackend`, `WebGPUBackend`, and `WebGLBackend`.

## Background
Depth testing and depth writing are separate render states. A material may need
to remain depth-tested against previously rendered geometry while leaving the
scene depth buffer unchanged for later draws, post-processing, or custom
compositing.

## API/Contract
- `MaterialParams.depthWrite?: boolean`
	- Input contract: accepts `true`, `false`, or `undefined`.
	- Default contract: `undefined` must be treated as `true`.
	- Behavior contract:
		- When `true`, opaque materials must keep existing depth-write behavior.
		- When `false`, opaque materials must still run the normal depth test but
		  must not update the main scene depth buffer.
		- Transparent materials must remain read-only for depth writes regardless
		  of this flag.
- `materialWritesDepth(material: Material): boolean`
	- Input contract: accepts a `Material` instance.
	- Output contract: returns `false` only when `material.depthWrite === false`.
	- Constraint contract: render backends must use this helper or equivalent
	  logic when selecting depth-write render state.
- Early Z pre-pass contract:
	- Materials with `depthWrite === false` must not participate in depth-only
	  pre-passes.
	- Color draws for these materials must not be promoted to read-only
	  `early-z-color` pipelines because they were never pre-passed.
- WebGPU deferred lighting contract:
	- Materials with `depthWrite === false` must not enter the deferred G-buffer
	  path.
	- These materials must use the legacy forward fallback after deferred
	  lighting resolve when MRT scene targets are available.
	- Transparent and transmission materials must stay outside the deferred
	  opaque G-buffer path regardless of `depthWrite`.
- Cache contract:
	- Material signatures and backend pipeline keys must include `depthWrite`.

## Usage
```ts
import { ShaderMaterial } from "../src/materials";

const material = new ShaderMaterial({
	depthWrite: false,
});

console.assert(material.depthWrite === false);
```

```bash
bun tests/static/shaders/test_shader_material.mjs
bun tests/static/webgpu/test_webgpu_bridge.mjs
bun tests/static/software/test_software_early_z_prepass.mjs
```

## Errors & Diagnostics
- No new warning key is required.
- Invalid or missing shader source must continue following existing
  `ShaderMaterial` diagnostics.
- Non-finite render inputs must continue following existing backend guards.

## Compatibility / Breaking Changes
This change is additive. Existing materials default to `depthWrite === true`
and preserve prior rendering behavior.
