# Software Early Z Pre-Pass Contract
## Scope
This document defines the `SoftwareBackend` Early Z pre-pass contract for
`main-opaque` rendering.

## Background
Software shading cost increases with overdraw. A depth-only pre-pass may reduce
unnecessary fragment shading when opaque geometry is heavily overlapping.

## API/Contract
- `SoftwareBackendOptions.enableEarlyZPrepass?: boolean`
	- Input contract: accepts `true`, `false`, or `undefined`.
	- Default contract: `undefined` must be treated as `true`.
	- Behavior contract:
		- When `true`, the software main opaque path must execute a depth-only
		  pre-pass before color shading.
		- When `false`, the software main opaque path must run without the
		  pre-pass and must keep the legacy depth test path.
- Pass scope contract:
	- Early Z pre-pass must apply only to `main-opaque`.
	- `main-transparent` and particle passes must not use this pre-pass.
- Mask material contract:
	- Triangles with `material.alphaMode === MASK` must be skipped in the
	  pre-pass.
	- These triangles must still be evaluated in the regular color pass.
- Depth-write material contract:
	- Triangles with `material.depthWrite === false` must be skipped in the
	  pre-pass.
	- These triangles must still be depth-tested and shaded in the regular color
	  pass, but they must not update `attachments.depthBuffer`.
- Buffer contract:
	- The implementation must maintain an internal reusable early-depth buffer.
	- Full-frame render must clear the full early-depth buffer region to
	  `Infinity` before pre-pass writes.
	- Incremental render must clear dirty rect regions to `Infinity` and may
	  preserve non-dirty regions.
- Depth-test contract:
	- During color shading, when early-depth buffer is present, early gate must
	  use `zCamValue <= earlyDepthBuffer[pixel]`.
	- Final visibility write to `attachments.depthBuffer` must keep strict
	  `shadedDepth < depthBuffer[pixel]`.

## Usage
```ts
import { SoftwareBackend } from "../src/renderers/SoftwareBackend";

const backend = new SoftwareBackend({
	rasterMode: "tile",
	enableEarlyZPrepass: true,
	tile: {
		tileSize: 32,
		workerCount: 4,
	},
});
```

```bash
bun tests/test_software_early_z_prepass.mjs
```

## Errors & Diagnostics
- No new warning key is required by this contract.
- If worker setup fails in tile mode, existing fallback diagnostics must remain
  unchanged.
- Non-finite or invalid rendering inputs must continue following existing
  software rasterizer guards and must not introduce new crash paths.

## Compatibility / Breaking Changes
This change is additive. Existing users are backward compatible because
`enableEarlyZPrepass` defaults to enabled behavior and can be disabled
explicitly.
