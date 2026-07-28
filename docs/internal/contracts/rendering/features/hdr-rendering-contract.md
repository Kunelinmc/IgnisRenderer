# HDR Rendering Contract
## Scope
This document defines the cross-backend HDR rendering contract for IgnisRenderer.
It covers internal HDR scene storage, SDR presentation, backend fallback behavior,
and the reserved design space for future Display HDR output.

## Background
IgnisRenderer lighting calculations are linear and may produce values above `1.0`.
GPU backends must preserve those values through scene rendering, OIT, bloom, and
tone mapping whenever the runtime supports floating-point color attachments.
Final canvas presentation remains SDR unless a future public Display HDR API is
introduced.

## API/Contract
- `WebGPUBackend` must store main scene color, post-process ping-pong targets,
  and OIT color resolve inputs in `rgba16float` textures when rendering through
  the render graph.
- WebGPU environment background rendering must not clamp linear HDR radiance when
  writing to an offscreen HDR scene target.
- WebGPU environment background rendering must clamp only when the shader writes
  directly to a gamma-encoded single/canvas target.
- WebGPU present output must remain SDR and must clamp the final display color to
  `[0.0, 1.0]`.
- `WebGLBackend` must use `rgba16float` scene, motion-depth, post-process, TAA,
  SSAO, and OIT accumulation attachments only when `EXT_color_buffer_float` is
  available.
- `WebGLBackend` must fall back to `rgba8unorm` color, motion-depth, and
  post-process attachments when `EXT_color_buffer_float` is unavailable.
- `WebGLBackend` must report the actual runtime attachment formats in
  `LogicalGBufferBridge` channels.
- WebGL post-process resources requested as `rgba16float` must return a handle
  with `format: "rgba8unorm"` when the float color attachment extension is
  unavailable.
- `SoftwareBackend` is an SDR fallback. It may run tone mapping and gamma passes
  over byte color buffers, but it must not claim internal HDR preservation.
- Display HDR output is not a public API. A future implementation may add a
  renderer-level display output contract, but it must keep SDR presentation as
  the default compatibility path.

## Usage
```ts
import { Renderer, ToneMappingPass, WebGPUBackend } from "../src";

const renderer = new Renderer({
	backend: new WebGPUBackend(),
	canvas,
	camera,
});
renderer.postProcess.getPass<ToneMappingPass>("tonemap")?.enable();
renderer.requestRender("hdr-internal");
```

```bash
bun tests/static/webgpu/test_webgpu_bridge.mjs
bun tests/static/webgl/test_webgl_frame_executor_fxaa.mjs
```

## Errors & Diagnostics
- `webgl-hdr-float-unsupported`: triggered when `EXT_color_buffer_float` is not
  available and WebGL falls back from HDR-capable float attachments to `RGBA8`.
- If `tonemap` is disabled, HDR highlights may hard-clip during SDR presentation.
- If a backend loses its frame targets, post-process histories must be recreated
  before they are reused.

## Compatibility / Breaking Changes
- SDR canvas presentation remains the default behavior.
- No public Display HDR API is introduced by this contract.
- WebGL bridge format reporting changes from hardcoded `rgba16float` values to
  actual runtime attachment formats.
