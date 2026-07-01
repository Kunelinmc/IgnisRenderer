# Renderer and Backend Core Contract

## Scope
This document defines the core lifecycle, frame scheduling, command recording, extension registry, and custom render targets contract for `Renderer`, `IRenderBackend`, and their related interfaces.

## Background
In IgnisRenderer, the `Renderer` acts as the main application-facing facade. It coordinates with an attached backend instance implementing `IRenderBackend`.
The core rendering loop requires deterministic frame lifecycle hooks (`beginFrame`, `executePass`, `endFrame`, `abortFrame`) managed by a `FrameCoordinator`.
GPU devices may be lost or restored at runtime, necessitating a robust, backend-agnostic device lifecycle recovery contract.
To keep the main interfaces clean, optional features are exposed through a typed extension registry.
Furthermore, custom rendering workflows can register custom render targets and passes without accessing native backend handles.
All graphics commands are recorded through a backend-agnostic `ICommandEncoder`.

## API/Contract

### 1. Frame Scheduling & Loop (`Renderer`)
- `Renderer.renderLoop()`
  - Behavior contract: must schedule frames through `requestAnimationFrame`.
  - Behavior contract: must await each `renderFrame(nowMs)` call before scheduling the next frame.
  - Behavior contract: must log frame failures through `Logger.error` and continue scheduling later frames.
  - Output contract: must return an idempotent function that stops the loop and cancels a pending animation frame request when possible.
  - Constraint: repeated calls while the loop is active must return the same stop function and must not create another loop.
- `Renderer.renderFrame(nowMs)`
  - Input contract: `nowMs` must use the animation-frame timestamp time base in milliseconds.
  - Behavior contract: must render at most one frame and must not schedule a later frame.
  - Constraint: concurrent calls must reject.
- `Renderer.renderScene(nowMs)`
  - Compatibility contract: must remain a deprecated alias of `renderFrame(nowMs)`.
  - Constraint: new application code must use `renderFrame(nowMs)` for manual rendering or `renderLoop()` for automatic scheduling.
- `Renderer.destroy()`
  - Behavior contract: must stop the active render loop before waiting for an in-progress frame and destroying the attached backend.

### 2. Device Lifecycle (`Renderer` & `IRenderBackend`)
- `IRenderBackend.attach(context)`
  - Behavior contract: must bind the backend instance to one renderer-owned surface and event sink.
  - Constraint: each backend instance may be attached only once.
  - Constraint: a second call must throw, including after `destroy()`.
- `IRenderBackend.profile.id`
  - Output contract: must identify the backend implementation.
- `RenderBackendDeviceLostInfo`
  - Input contract: `reason` may contain a backend-specific loss reason.
  - Input contract: `message` may contain a diagnostic message.
- `RenderBackendEvent`
  - Must be a discriminated union representing backend state transitions.
  - `type: "device-lost"`: emitted when the backend observes device loss. Must carry `info` payload when diagnostics are available.
  - `type: "device-restored"`: emitted when the backend finishes context restoration.
  - `type: "render-invalidated"`: emitted when the backend invalidates visual state. Must carry a semantic `reason` of type `RenderDirtyReason`.
  - `type: "resource-lifecycle"`: emitted when backend-owned resources require renderer-side invalidation or destruction.
- `RenderBackendEventSink`
  - `emit(event: RenderBackendEvent): void`: method called by attached backends to dispatch events.
- `IRenderBackend.initialize()`
  - Behavior contract: must initialize the graphics context and acquire device resources.
  - Constraint: must throw when called before `attach(context)`, after the backend is destroyed, or when already initialized.
- `IRenderBackend.getDebugInfo()`
  - Output contract: must return a `RenderBackendDebugInfo` snapshot.
  - Behavior contract: must not initialize backend resources or change renderer lifecycle state.
  - Behavior contract: must return `available: false` with `unavailableReason` before backend initialization.
  - Behavior contract: may omit or redact `device`, `limits`, and `features` fields when browser privacy policy or runtime support prevents collection.
  - Constraint: callers must not use `driverVersion` or device identifier strings for feature gating. Feature decisions must use `IRenderBackend.profile.capabilities`.
- `RenderBackendDebugInfo`
  - `backend`: must identify the backend implementation.
  - `api`: must identify the graphics API surface as `"software"`, `"webgpu"`, or `"webgl2"`.
  - `available`: must be `true` only when the backend has collected runtime diagnostics.
  - `unavailableReason`: should describe why diagnostics are unavailable when `available` is `false`.
  - `device`: may contain best-effort adapter identifiers including `vendor`, `renderer`, `architecture`, `device`, `description`, `isFallbackAdapter`, `driverVersion`, and `raw`.
  - `limits`: may contain selected numeric API limits.
  - `features`: may contain sorted backend feature or extension names.
- `IRenderBackend.restore()`
  - Behavior contract: must rebuild the graphics context and device resources after loss.
  - Behavior contract: must trigger resource recovery and emit `device-restored` when complete.
- `IRenderBackend.destroy()`
  - Behavior contract: must release all device contexts, textures, buffers, and cached post-process implementations.
  - Constraint: must be idempotent.
- `Renderer.initialize()`
  - Behavior contract: must call `IRenderBackend.initialize()`.
  - Constraint: must throw if already initialized.
- `Renderer.restore()`
  - Behavior contract: must call `IRenderBackend.restore()`.
  - Behavior contract: resets the prepared-scene cache and marks the next frame dirty.
- `Renderer.destroy()`
  - Behavior contract: must wait for the active frame to finish, then call `IRenderBackend.destroy()`.
  - Constraint: must be idempotent.
- `RendererEvents.devicelost`
  - Output contract: emitted to the application after renderer-owned device-loss bookkeeping completes.
- `WebGPUBackend`
  - Must listen to `GPUDevice.lost`.
  - Must perform internal rollback, mark device as lost, and then emit `device-lost` event through `RenderBackendEventSink`.
- `WebGLBackend`
  - Must handle `webglcontextlost` by marking context as lost and emitting `device-lost`.
  - Must handle `webglcontextrestored` by restoring state and emitting `device-restored`.

### 3. Frame Execution Lifecycle (`IRenderBackend`)
- `IRenderBackend.beginFrame(context: FrameContext)`
  - Behavior contract: must prepare command encoders, bind presentation attachments, and transition frame state.
  - Constraint: must throw if another frame is already active or if the backend is uninitialized.
- `IRenderBackend.executePass(pass: FramePass, context: FrameContext)`
  - Behavior contract: must execute the commands for the given `FramePass`.
  - Constraint: must throw if no frame is active.
- `IRenderBackend.skipPass(pass: FramePass)`
  - Behavior contract: called when a pass is disabled in the frame plan, allowing the backend to release/transition dependencies.
- `IRenderBackend.endFrame()`
  - Behavior contract: must finalize command encoders, submit command buffers, and present the frame.
  - Constraint: must throw if no frame is active.
- `IRenderBackend.abortFrame(error?: unknown)`
  - Behavior contract: must cancel/release active encoders and discard command buffers.
  - Constraint: must be idempotent and must not throw if no frame is active.
  - Constraint: must not present to the canvas, commit temporal history, or submit work.
- Deferred flushing:
  - Backends must defer resize, MSAA, and shader runtime compilation updates while a frame is active.
  - Deferred updates must be flushed immediately after `endFrame` or `abortFrame` clears the active frame state.

### 4. Extension Registry
- `IRenderBackend`
  - Must expose a `RenderBackendExtensionRegistry`.
  - Must preserve extension object identity for the lifetime of the backend.
  - Must not expose feature-specific optional APIs as direct properties when a typed extension key exists.
- `RenderBackendExtensionKey<TApi>`
  - Must carry a unique string `id`.
- `RenderBackendExtensionRegistry.getBackendExtension(key)`
  - Behavior contract: must return the extension API for the specified `key`, or `null` if the backend does not implement it.
- `RenderBackendExtensionRegistry.requireBackendExtension(key)`
  - Behavior contract: must return the extension API or throw a deterministic error if the extension is unavailable.
- `OCCLUSION_CULLING_EXTENSION`
  - Must expose an `OcclusionCullingBackendAdapter` API.
- `PROBE_CAPTURE_EXTENSION`
  - Must expose a `ProbeWebGPUCaptureSource` API.
- `WEBGPU_COMPUTE_EXTENSION`
  - Must expose an `IWebGPUComputeFacade` API.
- Identity Persistence:
  - Extension API objects must maintain the same object identity for the lifetime of the backend runtime.
- Device Loss Behavior:
  - During a device-lost state, invoking operations on extension APIs must throw a clear, descriptive error.
  - After `restore()` completes, the existing extension API objects must resume normal operation.

### 5. Command Recording (`ICommandEncoder`)
- `ICommandEncoder.beginRenderPass(desc)` must begin a render pass.
- `ICommandEncoder.beginComputePass(desc?)` must begin a compute pass.
- `ICommandEncoder.copyTextureToTexture(source, destination, copySize)` may record an in-frame texture copy.
- `copyTextureToTexture` must be called only when no render or compute pass is active.
- `source.texture` and `destination.texture` must be owned by the same backend implementation as the encoder.
- `source.texture` must have copy-source usage, and `destination.texture` must have copy-destination usage.
- `copySize.width` and `copySize.height` must be positive integers.
- Backends must preserve command order between render passes, compute passes, and `copyTextureToTexture` calls recorded on the same `ICommandEncoder`.
- Backend-level copy helpers that allocate or batch separate command encoders must not be used when an in-frame pass depends on copy ordering.
- `ICommandEncoder` must not expose native backend command encoders through the shared contract.

### 6. Custom Render Targets (`Renderer.renderTargets` & `Renderer.renderPasses`)
- `Renderer.renderTargets.register(descriptor)` must register one persistent custom render target.
- `RenderTargetDescriptor.id` must be unique within the renderer.
- `RenderTargetDescriptor.size` must use `canvas-scale` or `fixed`.
- `RenderTargetDescriptor.color` must contain at least one color attachment.
- `RenderTargetDescriptor.depth` may define one depth attachment.
- `RenderTargetDescriptor.sampleCount` defaults to `1`.
- `Renderer.renderTargets.readColor(id, attachmentIndex, options)` must return a `Promise<TextureReadbackResult>`.
- `readColor` must read only the most recent successfully completed frame.
- `readColor` must reject before the first successful frame, after an invalid target id, or for an invalid color attachment index.
- `Renderer.renderPasses.register(descriptor)` must register a backend pass stage with the same `id`.
- `CustomRenderPassDescriptor.target` must reference a registered custom render target.
- `CustomRenderPassDescriptor.execute(context)` must receive an `ICommandEncoder`, `CustomRenderTargetExecutionTarget`, `FrameContext`, backend id, dimensions, and a backend-owned resource facade.
- Custom render pass callbacks must not receive native backend handles.
- WebGPU and WebGL backends must support custom render targets, custom render passes, and color readback.
- SoftwareBackend must report custom render target support as unavailable and must skip custom render passes without failing the frame.

## Usage

### 1. Initialization and Lifecycle
```ts
import { Renderer, WebGPUBackend } from "ignisrenderer";

const backend = new WebGPUBackend();
console.info(`Using ${backend.id} backend`);
const renderer = new Renderer({
	canvas,
	backend,
	camera,
});

await renderer.initialize();

renderer.on("devicelost", ({ info }) => {
	console.warn(`Device lost: ${info?.message ?? "unknown"}`);
});

// Manual recovery
await renderer.restore();
```

### 2. Backend Debug Info
```ts
const debugInfo = renderer.getBackendDebugInfo();
if (debugInfo.available) {
	console.info(debugInfo.backend, debugInfo.api, debugInfo.device?.vendor);
	console.info(debugInfo.limits?.maxTextureDimension2D);
} else {
	console.info(debugInfo.unavailableReason);
}
```

### 3. Frame Loop Control
```ts
// Start automatic rendering loop
const stopRenderLoop = renderer.renderLoop();

// Stop automatic rendering when the application no longer needs it.
stopRenderLoop();

// Render one frame manually (e.g. for tools or tests)
await renderer.renderFrame(performance.now());
```

### 4. Frame Coordinator Execution (Internal / Backend Implementation)
```ts
// Inside FrameCoordinator execution loop
try {
	await backend.beginFrame(frameContext);
	for (const pass of framePlan.backendPasses) {
		if (pass.enabled) {
			await backend.executePass(pass, frameContext);
		} else {
			backend.skipPass?.(pass);
		}
	}
	await backend.endFrame();
} catch (error) {
	await backend.abortFrame(error);
	throw error;
}
```

### 5. Querying Extensions
```ts
import { WEBGPU_COMPUTE_EXTENSION } from "ignisrenderer";

const compute = renderer.getBackendExtension(WEBGPU_COMPUTE_EXTENSION);
if (compute) {
	const buffer = compute.createBuffer({ size: 1024, usage: BufferUsage.Storage });
}
```

### 6. In-Frame Command Recording and Texture Copying
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

### 7. Custom Render Targets & Passes
```ts
import { TextureFormat, type CustomRenderPassContext } from "ignisrenderer";

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

## Errors & Diagnostics
- `Renderer render loop frame failed.`: logged when a frame rejects. The original error must be included in the diagnostic, and the loop must continue.
- `Renderer.renderFrame() cannot run concurrently.`: returned when another frame is still active.
- Errors from manually awaited `renderFrame()` calls must continue to reject to the caller without automatic logging by `Renderer`.
- `Renderer.initialize() cannot be called multiple times.`: triggered when `initialize` is invoked on an already initialized renderer.
- `RenderBackendDebugInfo.available === false`: returned before initialization, for the software backend, or when diagnostics cannot be collected. This is not an error.
- `device-lost` event triggers warning logging with backend-supplied details.
- Context restoration failures must log `WebGL context restore failed` or throw appropriate errors.
- `beginFrame` called while a frame is active must throw an error.
- If aborting fails, the backend must catch the error, log a critical diagnostic, and rethrow the original frame error.
- `Render backend extension "<id>" is unavailable.`: thrown when calling `requireBackendExtension` for an unsupported extension.
- Operations on extension APIs during device loss must throw an error with the message prefix `Device lost: `.
- `Cannot copy textures while a render or compute pass is active.` is thrown when WebGPU records `copyTextureToTexture` during an active pass.
- `Render texture is not backed by a WebGPU texture.` is thrown when WebGPU receives a texture that is not owned by the WebGPU backend.
- `webgpu-oit-disabled-runtime` is logged when WebGPU OIT is enabled but the active encoder cannot record in-frame texture copies.
- `Render target "<id>" is already registered.` is thrown for duplicate target ids.
- `Custom render pass "<id>" is already registered.` is thrown for duplicate pass ids.
- `Render target "<id>" cannot be read before a successful frame completes.` is thrown when readback is requested too early.
- `software-custom-render-targets-unsupported` is logged when SoftwareBackend skips a custom render pass.
- `<backend>-custom-render-target-msaa-unsupported-<id>` is logged when a backend cannot allocate a requested sample count.

## Compatibility / Breaking Changes
- `Renderer.renderScene()` is deprecated and retained only for compatibility. Applications must use `Renderer.renderFrame()` for manual rendering or `Renderer.renderLoop()` for automatic scheduling. Neither `renderFrame()` nor the deprecated alias schedules subsequent frames based on backend `frameScheduling`.
- `IRenderBackend.createSession(context)` and public backend session APIs are removed.
- Backend instances are one-shot renderer runtimes and must not be reused across renderers.
- `Renderer.onDeviceLost` and `Renderer.onBackendResourceEvent` are removed.
- `RendererBackendBridge` is removed.
- Backends must route all lifecycle notifications as events through `RenderBackendEventSink` instead of direct callbacks.
- `IRenderBackend` instances are one-shot renderer runtimes. Applications must create a new backend instance for each `Renderer`.
- Backend profile, capability, extension, and frame lifecycle methods are read from the attached backend runtime.
- `IRenderBackend.executeSharedPass` is removed.
- Extensions must be queried via typed keys rather than raw string identifiers.
- `getNativeWebGPUCommandEncoder` is removed from `ICommandEncoder`. Code that requires ordered texture copies must use `copyTextureToTexture`. WebGPU-internal passes that need native WebGPU objects must resolve them through WebGPU-owned helpers instead of the shared renderer contract.
- SoftwareBackend does not support custom render targets.
