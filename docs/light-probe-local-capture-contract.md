# LightProbe Local Capture Contract

## Scope
This document defines the runtime capture contract for `LightProbe` instances
with `source === "capturedScene"` and the shared `ProbeCaptureRuntime` path.

## Background
`LightProbe` previously stored SH coefficients that were authored manually or
filled from environment IBL baking. Localized indoor and mixed lighting scenes
need `LightProbe` instances to capture low-frequency scene radiance from their
own world positions.

## API/Contract
- `LightProbe.source` must support `"environment"`, `"capturedScene"`, and
  `"manual"`.
- `LightProbe.source` must default to `"environment"`.
- `LightProbe.captureUpdateMode` must support `"manual"`, `"onSceneDirty"`,
  and `"interval"`.
- `LightProbe.captureResolution` must default to `64x32`.
- `LightProbe.captureFar` must default to `200`.
- `LightProbe.includeEnvironment`, `includeMeshes`, `includeTransparent`,
  `includeParticles`, and `includeShadows` must default to `true`.
- `LightProbe.requestCapture()` must increment `captureRequestToken` and
  `captureRevision`.
- `ProbeCaptureRuntime` must capture `LightProbe` instances only when
  `source === "capturedScene"`.
- `ProbeCaptureRuntime` must write captured low-frequency radiance to
  `LightProbe.sh`.
- `ProbeCaptureRuntime` must store radiance SH coefficients and must not store
  pre-convolved irradiance coefficients.
- `ProbeCaptureRuntime` must share one capture between `LightProbe` and
  `ReflectionProbe` targets when capture origin, `captureFar`, and include flags
  match.
- Environment IBL warmup and runtime update must mutate only `LightProbe`
  instances with `source === "environment"`.

## Usage
```ts
import { LightProbe } from "../src/lights/LightProbe";

const probe = new LightProbe({
	source: "capturedScene",
	shape: "box",
	halfExtents: { x: 4, y: 3, z: 6 },
	captureUpdateMode: "manual",
	captureResolution: { width: 64, height: 32 },
	includeEnvironment: true,
	includeMeshes: true,
});

probe.requestCapture();
```

```bash
bun tests/static/lighting/test_probe_capture_runtime.mjs
```

## Errors & Diagnostics
- If `source === "manual"`, `ProbeCaptureRuntime` must not overwrite
  `LightProbe.sh`.
- If `source === "environment"`, environment IBL update may overwrite
  `LightProbe.sh`.
- If mesh capture is requested without a compatible GPU face capture source,
  runtime must emit `probe-mesh-capture-unsupported` and use analytical
  fallback capture.
- If a probe changes transform, source, include flags, or capture request token
  while a capture is in flight, stale results must not overwrite that probe.

## Compatibility / Breaking Changes
- Breaking change: `LightProbe` initialization now accepts only
  `new LightProbe({ ...params })`. Code using `new LightProbe(sh)` must migrate
  to `new LightProbe({ sh })`; code using `new LightProbe()` must migrate to
  `new LightProbe({})`.
- Renderer pipeline stage `reflection-probe-capture` has been replaced by
  `probe-capture`.
- `ReflectionProbeCaptureRuntime` remains as a compatibility export alias for
  `ProbeCaptureRuntime`.
