# WebGPU Tone Mapping Contract
## Scope
This document defines the post-process tone mapping contract for the WebGPU backend.

## Background
WebGPU scene color is stored in HDR-capable intermediate textures.
Without tone mapping, highlights may be hard-clipped during final
presentation.

## API/Contract
- The default WebGPU post-process graph must register a `tonemap` pass.
- The `tonemap` pass must run after `interaction-outline` and before `gamma`.
- The `tonemap` pass must be enabled when `enableGamma` is `true`.
- The `tonemap` pass must read from `targets.sceneColor` and write to
  ping-pong post targets, then update `targets.sceneColor` to the
  written target.
- The shader `src/shaders/webgpu/postprocess/toneMapping.wgsl` must
  implement ACES-fitted mapping on linear RGB and preserve alpha.
- Warmup planning should include `tonemap` whenever `gamma` is enabled
  so shader compilation can happen before frame rendering.

## Usage
Use the existing WebGPU post-process runtime path. No additional user-side API call is required.

```bash
bun -e "import('./tests/test_webgpu_postprocess_runtime_screen.mjs').then((m) => m.run())"
```

The command above should pass and verify that `tonemap` compiles, dispatches, and updates `sceneColor`.

## Errors & Diagnostics
- If tone mapping shader compilation fails, warmup/reporting must
  surface a shader compile error in the WebGPU warmup phase.
- If post-process execution fails before `gamma`, presentation may fall
  back to direct scene color output, which can look overexposed.

## Compatibility / Breaking Changes
WebGPU output with `enableGamma=true` now includes tone mapping before
gamma encoding. This is a behavior change and may alter perceived
brightness/contrast compared with previous output.
