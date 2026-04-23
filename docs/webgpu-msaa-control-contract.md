# WebGPU MSAA Control Contract
## Scope
This document defines the explicit MSAA enable/disable control contract for
`WebGPUBackend`.

## Background
WebGPU rendering uses MSAA by default. Consumers need a direct boolean switch
to disable or re-enable MSAA without manually reasoning about sample counts.

## API/Contract
- `WebGPUBackendOptions.enableMSAA?: boolean`
	- Input contract: accepts `true`, `false`, or `undefined`.
	- Behavior contract:
		- When `enableMSAA` is `false`, backend initialization must start from
		  `1x` MSAA (`sampleCount = 1`).
		- When `enableMSAA` is `true` or omitted, backend initialization must use
		  the default MSAA preference (`4x`) and then clamp to supported values.
- `WebGPUBackend.setMSAAEnabled(enabled: boolean): void`
	- Input contract: accepts a boolean `enabled`.
	- Behavior contract:
		- When `enabled` is `false`, the backend must force `sampleCount = 1`.
		- When `enabled` is `true`, the backend must request multisampling again
		  using existing sample-count rules.
	- Constraint: final active count may be lower than requested when device
	  capabilities do not support higher MSAA counts.
- `WebGPUBackend.setMSAASampleCount(sampleCount: number): void`
	- Existing numeric control remains valid and may still be used for explicit
	  quality levels.
- `WebGPUBackend.getMSAASampleCount(): number`
	- Output contract: returns the resolved runtime sample count (`>= 1`).

## Usage
```ts
import { WebGPUBackend } from "../src/renderers/WebGPUBackend";

const backend = new WebGPUBackend({ enableMSAA: false });

// Enable MSAA explicitly at runtime.
backend.setMSAAEnabled(true);

// Disable MSAA explicitly at runtime.
backend.setMSAAEnabled(false);
```

```bash
bun tests/test_webgpu_backend_cache_and_dependency.mjs
```

## Errors & Diagnostics
- `setMSAASampleCount()` should ignore non-finite inputs.
- Requested MSAA counts that are unsupported must be clamped to the nearest
  supported value, and may end at `1x`.
- Runtime MSAA allocation fallback may log
  `webgpu-msaa-runtime-fallback-1x` and retry at `1x`.

## Compatibility / Breaking Changes
This change is additive. Existing default behavior remains MSAA enabled unless
`enableMSAA` is explicitly set to `false`.
