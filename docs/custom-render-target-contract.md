# Custom Render Target Contract

## Scope

This document defines the public custom render target and custom render pass
contract exposed through `Renderer.renderTargets` and `Renderer.renderPasses`.

## Background

Applications need offscreen render targets for tools, capture, intermediate
effects, and diagnostics. These targets must remain backend-agnostic: user code
must interact with IgnisRenderer descriptors, handles, command encoders, and
readback results instead of native `GPUTexture`, `GPUDevice`,
`WebGLFramebuffer`, or `WebGL2RenderingContext` objects.

## API/Contract

- `Renderer.renderTargets.register(descriptor)` must register one persistent
  custom render target.
- `RenderTargetDescriptor.id` must be unique within the renderer.
- `RenderTargetDescriptor.size` must use `canvas-scale` or `fixed`.
- `RenderTargetDescriptor.color` must contain at least one color attachment.
- `RenderTargetDescriptor.depth` may define one depth attachment.
- `RenderTargetDescriptor.sampleCount` defaults to `1`.
- `Renderer.renderTargets.readColor(id, attachmentIndex, options)` must return a
  `Promise<TextureReadbackResult>`.
- `readColor` must read only the most recent successfully completed frame.
- `readColor` must reject before the first successful frame, after an invalid
  target id, or for an invalid color attachment index.
- `Renderer.renderPasses.register(descriptor)` must register a backend pass stage
  with the same `id`.
- `CustomRenderPassDescriptor.target` must reference a registered custom render
  target.
- `CustomRenderPassDescriptor.execute(context)` must receive an
  `ICommandEncoder`, `CustomRenderTargetExecutionTarget`, `FrameContext`,
  backend id, dimensions, and a backend-owned resource facade.
- Custom render pass callbacks must not receive native backend handles.
- WebGPU and WebGL backends must support custom render targets, custom render
  passes, and color readback.
- SoftwareBackend must report custom render target support as unavailable and
  must skip custom render passes without failing the frame.

## Usage

```ts
import {
	Renderer,
	TextureFormat,
	type CustomRenderPassContext,
} from "ignisrenderer";

renderer.renderTargets.register({
	id: "inspect",
	size: { mode: "fixed", width: 256, height: 256 },
	color: [
		{ format: TextureFormat.RGBA8Unorm },
		{ format: TextureFormat.RGBA8Unorm },
	],
	depth: { format: TextureFormat.Depth32Float },
});

renderer.renderPasses.register({
	id: "inspect-pass",
	target: "inspect",
	dependsOn: ["main-opaque"],
	execute(context: CustomRenderPassContext) {
		context.encoder.beginRenderPass({
			label: "InspectClear",
			colorAttachments: context.target.color.map((attachment) => ({
				view: attachment.texture,
				loadOp: "clear",
				storeOp: "store",
				clearValue: { r: 0, g: 0, b: 0, a: 1 },
			})),
			depthStencilAttachment: context.target.depth ? {
				view: context.target.depth.texture,
				depthLoadOp: "clear",
				depthStoreOp: "store",
				depthClearValue: 1,
			} : undefined,
		});
		context.encoder.endRenderPass();
	},
});

await renderer.renderFrame(performance.now());
const readback = await renderer.renderTargets.readColor("inspect", 0);
```

```bash
bun tests/static/renderer/test_renderer_custom_render_targets.mjs
```

## Errors & Diagnostics

- `Render target "<id>" is already registered.` is thrown for duplicate target
  ids.
- `Custom render pass "<id>" is already registered.` is thrown for duplicate pass
  ids.
- `Render target "<id>" cannot be read before a successful frame completes.` is
  thrown when readback is requested too early.
- `software-custom-render-targets-unsupported` is logged when SoftwareBackend
  skips a custom render pass.
- `<backend>-custom-render-target-msaa-unsupported-<id>` is logged when a backend
  cannot allocate a requested sample count.

## Compatibility / Breaking Changes

N/A

