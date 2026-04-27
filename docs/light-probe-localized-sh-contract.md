# LightProbe Localized SH Contract

## Scope
This document defines the public contract for localized `LightProbe` spherical
harmonics in IgnisRenderer, including `shape`, `blendDistance`, `priority`, and
backend-specific behavior.

## Background
`LightProbe` previously behaved as a global SH accumulator. Indoor scenes,
multi-room transitions, caves, and indoor/outdoor boundaries require localized
probe influence with deterministic overlap resolution.

## API/Contract
- `LightProbe` must accept both `new LightProbe(sh, intensity)` and
  `new LightProbe({ ...params })`.
- `LightProbe.shape` must support `"global"`, `"sphere"`, and `"box"`.
- `LightProbe.shape` must default to `"global"`.
- `LightProbe.radius` must be finite and must be sanitized to a positive value.
- `LightProbe.halfExtents` must be finite and each component must be sanitized
  to a positive value.
- `LightProbe.blendDistance` must be finite and must be sanitized to `>= 0`.
- `LightProbe.priority` must be finite and must be sanitized to an integer.
- `LightProbe.copy()` and `LightProbe.clone()` must preserve SH coefficients and
  localized probe properties.
- Localized `LightProbe` selection must evaluate only probes in the highest
  active `priority` group.
- Higher numeric `priority` values must win over lower numeric `priority`
  values.
- Within the winning `priority` group, rendering backends must normalize the
  top two active probe weights and must blend only those two probes.
- Weight tie-breaks must be deterministic and must fall back to `LightProbe.id`.
- `blendDistance` must use normalized metric fade semantics consistent with
  `ReflectionProbe`, including an effective minimum floor derived from probe
  size.
- WebGPU and WebGL must treat `shape="global"` probes as contributors to the
  global SH buffer and must evaluate localized probes per-fragment.
- The Software backend must remain compatible by treating localized probes as
  global SH contributors.
- WebGPU and WebGL must clamp localized probe collection to `8` probes per
  frame.

## Usage
```ts
import { LightProbe } from "../src/lights/LightProbe";
import { SH } from "../src/maths/SH";

const hallwayProbe = new LightProbe({
	shape: "box",
	halfExtents: { x: 4, y: 3, z: 6 },
	blendDistance: 0.2,
	priority: 20,
	sh: SH.empty(),
	intensity: 1,
});

const courtyardProbe = new LightProbe({
	shape: "sphere",
	radius: 12,
	blendDistance: 0.35,
	priority: 5,
	sh: SH.empty(),
	intensity: 1,
});

const fallbackProbe = new LightProbe(SH.empty(), 0.5);
fallbackProbe.shape = "global";
```

```bash
bun tests/test_light_probe_runtime.mjs
```

## Errors & Diagnostics
- Non-finite `radius`, `halfExtents`, `blendDistance`, or `priority` values
  will be sanitized during construction and runtime cache refresh.
- If more than `8` localized probes are available, WebGPU and WebGL will select
  the camera-relevant subset for the current frame.
- If a localized probe appears inactive at a fragment, its weight will resolve
  to `0` and it will not participate in blending.
- If no localized probe is active at a fragment, WebGPU and WebGL will fall
  back to the global SH buffer.
- If no global SH data is available, localized probe coverage may blend against
  zero SH outside probe influence regions.

## Compatibility / Breaking Changes
- Existing code using `new LightProbe(sh, intensity)` remains supported.
- Existing scenes that do not set `shape` remain behaviorally compatible because
  `shape` defaults to `"global"`.
- WebGPU and WebGL scene lighting may change when localized probes are added
  because only `shape="global"` probes contribute to global SH accumulation on
  those backends.
- The Software backend intentionally preserves the previous global accumulation
  behavior for localized probes.
