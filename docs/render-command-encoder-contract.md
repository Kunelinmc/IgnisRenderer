# Render Command Encoder Contract
## Scope
This document defines the backend-agnostic command recording contract for
`ICommandEncoder`.

## Background
`ICommandEncoder` records ordered draw, compute, and copy commands for one
command stream. Backend code may require a texture copy between render passes,
such as copying scene color before resolving order-independent transparency.
That copy must stay in the same command stream when pass ordering is observable.

## API/Contract
- `ICommandEncoder.beginRenderPass(desc)` must begin a render pass.
- `ICommandEncoder.beginComputePass(desc?)` must begin a compute pass.
- `ICommandEncoder.copyTextureToTexture(source, destination, copySize)` may
  record an in-frame texture copy.
- `copyTextureToTexture` must be called only when no render or compute pass is
  active.
- `source.texture` and `destination.texture` must be owned by the same backend
  implementation as the encoder.
- `source.texture` must have copy-source usage, and `destination.texture` must
  have copy-destination usage.
- `copySize.width` and `copySize.height` must be positive integers.
- Backends must preserve command order between render passes, compute passes,
  and `copyTextureToTexture` calls recorded on the same `ICommandEncoder`.
- Backend-level copy helpers that allocate or batch separate command encoders
  must not be used when an in-frame pass depends on copy ordering.
- `ICommandEncoder` must not expose native backend command encoders through the
  shared contract.

## Usage
```ts
encoder.beginRenderPass({
	label: "TransparentAccumulation",
	colorAttachments: [
		{
			view: accumTarget,
			loadOp: "load",
			storeOp: "store",
		},
	],
});
encoder.draw(3);
encoder.endRenderPass();

encoder.copyTextureToTexture?.(
	{ texture: sceneColor },
	{ texture: sceneColorCopy },
	{ width, height, depthOrArrayLayers: 1 }
);

encoder.beginRenderPass({
	label: "TransparentResolve",
	colorAttachments: [
		{
			view: sceneColor,
			loadOp: "load",
			storeOp: "store",
		},
	],
});
encoder.draw(3);
encoder.endRenderPass();
```

```bash
bun tests/test_webgpu_frame_executor_resilience.mjs
bun tests/test_webgpu_bridge.mjs
```

## Errors & Diagnostics
- `Cannot copy textures while a render or compute pass is active.` is thrown
  when WebGPU records `copyTextureToTexture` during an active pass.
- `Render texture is not backed by a WebGPU texture.` is thrown when WebGPU
  receives a texture that is not owned by the WebGPU backend.
- `webgpu-oit-disabled-runtime` is logged when WebGPU OIT is enabled but the
  active encoder cannot record in-frame texture copies.

## Compatibility / Breaking Changes
`getNativeWebGPUCommandEncoder` is removed from `ICommandEncoder`. Code that
requires ordered texture copies must use `copyTextureToTexture`. WebGPU-internal
passes that need native WebGPU objects must resolve them through WebGPU-owned
helpers instead of the shared renderer contract.
