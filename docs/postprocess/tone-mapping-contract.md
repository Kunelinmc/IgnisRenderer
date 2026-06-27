# Post-Process Tone Mapping Contract
## Scope
This document defines the `tonemap` post-process contract across Software, WebGL, and WebGPU backends.

## Background
IgnisRenderer's main scene color is rendered and stored in HDR formats (e.g. `rgba16float` in WebGPU and WebGL when float attachments are supported). Without tone mapping, HDR highlights exceeding `1.0` will hard-clip during final SDR presentation. The `tonemap` pass maps linear HDR values to the standard SDR range `[0.0, 1.0]` using ACES-fitted tone mapping.

## API/Contract
- The engine-provided `ToneMappingPass` must use pass id `"tonemap"`.
- `ToneMappingPass` is a built-in pass that reports `PostProcessPass.builtIn === true`.
- `Renderer` must automatically register `ToneMappingPass` by default, enabled initially.
- The `tonemap` pass must execute after `bloom` (when enabled) and before `color-filter` (when enabled).
- `ToneMappingPass` must support Software, WebGL, and WebGPU implementations:
  - **WebGPU**: Computes tone mapping using compute shader `src/shaders/webgpu/postprocess/toneMapping.wgsl`. It reads from `targets.sceneColor` and writes to the resolved ping-pong post-process target.
  - **WebGL**: Executes tone mapping using vertex shader `webgl.part.presentVertex` and fragment shader `src/shaders/webgl/parts/toneMappingFragment.glsl`.
  - **Software**: Modifies canvas pixels directly in CPU memory.
- All three backends must apply the same ACES-fitted mapping curve:
  $$\text{ACES}(x) = \frac{x \cdot (2.51 \cdot x + 0.03)}{x \cdot (2.43 \cdot x + 0.59) + 0.14}$$
  clamped to `[0.0, 1.0]` for the RGB channels, preserving the original alpha value.
- Warmup planning must compile the tone mapping pipelines/programs on GPU backends whenever the post-process registry snapshot resolves `postProcess.isEnabled("tonemap") === true`.

## Usage
```ts
import { Renderer, ToneMappingPass, WebGPUBackend } from "ignisrenderer";

const renderer = new Renderer(new WebGPUBackend(), canvas, camera);

// Tone mapping is enabled by default. To disable:
const toneMapPass = renderer.postProcess.getPass<ToneMappingPass>("tonemap");
toneMapPass?.disable();

// To enable again:
toneMapPass?.enable();
```

```bash
# Run backend validation tests
bun tests/static/webgpu/test_webgpu_postprocess_runtime_screen.mjs
```

## Errors & Diagnostics
- If tone mapping shader compilation fails on GPU backends, warmup/compilation reporting must surface the shader compile error.
- If post-process execution fails before `gamma`, presentation may fall back to direct scene color output, causing highlights to look overexposed or clipped.

## Compatibility / Breaking Changes
- `Renderer.features.enableToneMapping` and backend `postProcessCapabilities` are removed.
- Code must query `ToneMappingPass` from `renderer.postProcess` registry and invoke `.enable()` or `.disable()` on the pass instance.
