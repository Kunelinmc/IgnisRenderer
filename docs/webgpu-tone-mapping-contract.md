# WebGPU Tone Mapping Contract
## Scope
This document defines the `tonemap` post-process contract for the WebGPU backend.

## Background
WebGPU scene color is stored in HDR-capable intermediate textures. Without tone mapping, highlights may be hard-clipped during final presentation.

## API/Contract
- The default WebGPU post-process graph must register a `tonemap` pass.
- `renderer.postProcess` must enable `tonemap` by default.
- `renderer.postProcess.getPass<ToneMappingPass>("tonemap")?.disable()` must skip the pass.
- `renderer.postProcess.getPass<ToneMappingPass>("tonemap")?.enable()` must re-enable the pass.
- `ToneMappingPass` must expose a WebGPU implementation.
- `WebGPUBackend` must not expose `postProcessCapabilities`.
- The `tonemap` pass must run after `bloom` and before `color-filter`.
- The `tonemap` pass must read from `targets.sceneColor`, write to a ping-pong post target, and update `targets.sceneColor` to the written target.
- `src/shaders/webgpu/postprocess/toneMapping.wgsl` must implement ACES-fitted mapping on linear RGB and preserve alpha.
- Warmup planning should include `tonemap` whenever resolved post-process state returns `true` from `postProcess.isEnabled("tonemap")`.

## Usage
```ts
import { Renderer, ToneMappingPass, WebGPUBackend } from "ignis-renderer";

const renderer = new Renderer(new WebGPUBackend(), canvas, camera);
renderer.postProcess.getPass<ToneMappingPass>("tonemap")?.disable();
renderer.postProcess.getPass<ToneMappingPass>("tonemap")?.enable();
renderer.requestRender("postfx");
```

```bash
bun -e "import('./tests/static/webgpu/test_webgpu_postprocess_runtime_screen.mjs').then((m) => m.run())"
```

## Errors & Diagnostics
- If tone mapping shader compilation fails, warmup reporting must surface a shader compile error in the WebGPU warmup phase.
- If post-process execution fails before `gamma`, presentation may fall back to direct scene color output, which can look overexposed.

## Compatibility / Breaking Changes
`Renderer.features.enableToneMapping` and backend `postProcessCapabilities` are removed. Code must look up `ToneMappingPass` from `renderer.postProcess` and mutate the pass instance.
