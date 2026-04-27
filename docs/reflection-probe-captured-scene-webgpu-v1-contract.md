# Reflection Probe Captured Scene WebGPU v1 Contract
## Scope
This document defines the v1 contract for `ReflectionProbe` scene capture when
`source = "capturedScene"`.
The contract must be treated as authoritative for runtime behavior and defaults
in WebGPU backend integration.

## Background
Historically, `capturedScene` capture used an analytical light and skybox
approximation and did not render scene `Mesh` content.
WebGPU v1 introduces per-face scene rendering for probe capture to produce
reflective environment maps from real scene geometry and effects.

## API/Contract
- `ReflectionProbe` must preserve `source` behavior and continue to support
  `source = "capturedScene"`.
- `ReflectionProbe` must expose `includeMeshes: boolean`.
- `ReflectionProbe` must expose `includeTransparent: boolean`.
- `ReflectionProbe` must expose `includeParticles: boolean`.
- `ReflectionProbe` must expose `includeShadows: boolean`.
- The new flags must default to `true`.
- `captureResolution` default must be `512x256` for v1 capture quality.
- Existing field `includeSkybox` must remain supported.
- Existing field `captureUpdateMode` must remain supported.
- Existing field `captureFar` must remain supported.
- Existing method `requestCapture()` must remain supported.
- When a `ReflectionProbe` has a parent `Node`, capture origin must resolve
  from the parent world position while the probe transform continues to define
  the influence volume and parallax proxy.
- Capture scheduler must prefer nearest probes first when
  `cameraWorldPosition` is available.
- Runtime capture budget must default to `4ms` per frame.
- Resolution adaptation must be temporary and must use scale steps
  `1.0 -> 0.75 -> 0.5` for in-flight capture tasks.
- During capture rendering, `enableReflection` must be `false`.
- During capture rendering, `enableSSR` must be `false`.
- Shadow capture must reuse current frame shadow results and must not trigger a
  dedicated shadow recapture pass.
- Mesh capture path must be active only on WebGPU v1.
  Non-WebGPU backends may continue to use analytical fallback capture.

## Usage
```ts
import { ReflectionProbe } from "../src/lights/ReflectionProbe";

const probe = new ReflectionProbe({
	source: "capturedScene",
	captureUpdateMode: "manual",
	captureResolution: { width: 512, height: 256 },
	captureFar: 200,
	includeSkybox: true,
	includeMeshes: true,
	includeTransparent: true,
	includeParticles: true,
	includeShadows: true,
});

// Request capture explicitly for manual mode.
probe.requestCapture();
```

```ts
model.addChild(probe);
probe.position.set(0, 1.5, 0);
```

If the probe is parented under `model`, capture should originate from
`model` world position while `probe.position` may still offset the probe
volume for blending and parallax fit.

The renderer should execute probe capture in the
`reflection-probe-capture` stage and should pass frame context and active camera
world position into `ReflectionProbeCaptureRuntime`.

## Errors & Diagnostics
- If a probe never updates in `manual` mode, verify `requestCapture()` is
  called and `captureRequestToken` increments.
- If capture appears low resolution, verify runtime budget pressure is not
  forcing temporary scale downgrade to `0.75` or `0.5`.
- If non-WebGPU backend is active, mesh capture is not expected and analytical
  fallback behavior is expected.
- If capture contains recursive reflections or instability, verify capture
  features disable `enableReflection` and `enableSSR`.
- If reflections appear projected from the probe gizmo instead of the parent
  model origin, verify the `ReflectionProbe` is attached under the intended
  model `Node` and that world matrices are updated before rendering.

## Compatibility / Breaking Changes
- Behavior change: default `captureResolution` is now `512x256` (previously
  lower defaults).
- Behavior change: `includeMeshes`, `includeTransparent`,
  `includeParticles`, and `includeShadows` now exist and default to `true`.
- Backend compatibility: WebGPU path includes mesh capture; other backends keep
  fallback capture behavior.
- Behavior change: parented probes now capture and project from the parent
  world position instead of the probe node origin.
