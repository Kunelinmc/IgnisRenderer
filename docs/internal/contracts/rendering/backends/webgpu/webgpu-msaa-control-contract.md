# WebGPU MSAA Configuration Contract
## Scope
This document defines the constructor-only MSAA configuration contract for
`WebGPUBackend`.

## Background
WebGPU rendering uses single-sample rendering by default so deferred lighting
can activate when its other runtime requirements are available. Applications
may request MSAA when the backend is created, and the selected quality remains
internal to the WebGPU runtime thereafter.

## API/Contract
- `WebGPUBackendOptions.msaaSampleCount?: number`
	- Input contract: accepts a finite number. The value is floored and clamped
	  to at least `1`.
	- Behavior contract: omitted values request the default `1x` sample count.
	  Values greater than `1` request multisampling. The active count may be
	  lower than requested when device capabilities do not support the requested
	  count.
	- Error contract: non-finite values must throw a configuration error.
- MSAA runtime control is internal. `WebGPUBackend` must not expose
	`getMSAASampleCount()`, `setMSAAEnabled()`, or `setMSAASampleCount()`.
- The legacy `enableMSAA` option is removed. JavaScript callers that supply it
	must receive a deterministic error directing them to `msaaSampleCount`.

## Usage
```ts
import { WebGPUBackend } from "../src/backends/webgpu/WebGPUBackend";

const backend = new WebGPUBackend({ msaaSampleCount: 1 });
```

```bash
bun tests/static/webgpu/test_webgpu_backend_cache_and_dependency.mjs
```

## Errors & Diagnostics
- Requested MSAA counts that are unsupported must be clamped to the highest
  supported count that does not exceed the request, and may end at `1x`.
- A frame-target allocation failure may log `webgpu-msaa-runtime-fallback-1x`
  and retry at `1x` before recording render commands. The fallback remains
  active until the device runtime is reinitialized.

## Compatibility / Breaking Changes
This change is breaking. Runtime MSAA methods and `enableMSAA` are removed;
use `msaaSampleCount` when creating `WebGPUBackend`. The default sample count
is `1x`; applications that require MSAA must request it explicitly.
