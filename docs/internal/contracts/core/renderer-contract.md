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

- `new Renderer(options)`
  - Input contract: must accept exactly one `RendererOptions` object containing
    `backend`, `canvas`, and the optional `camera`.
  - Constraint: positional `backend`, `canvas`, and `camera` constructor
    arguments are not supported.
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
  - Output contract: must return an `IncrementalFrameStatus` in every result branch, including an on-demand clean result.
  - Output contract: `IncrementalFrameStatus` must distinguish planned tile coverage from verified final-output coverage.
- `Renderer.renderScene(nowMs)`
  - Compatibility contract: must remain a deprecated alias of `renderFrame(nowMs)`.
  - Constraint: new application code must use `renderFrame(nowMs)` for manual rendering or `renderLoop()` for automatic scheduling.
- `Renderer.updateSH()`
  - Behavior contract: must delegate spherical harmonics updates to the renderer-owned `FrameCoordinator`.
  - Constraint: must throw when the renderer-owned `FrameCoordinator` is unavailable and must not execute a fallback SH implementation.
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

- `FrameContext.viewCamera`
  - Output contract: must provide the active view/projection camera for the
    current frame or secondary capture context.
  - Behavior contract: renderer-driven frames must set `viewCamera` to the same
    camera used to build `FrameContext.scene`.
  - Constraint: secondary capture contexts, such as reflection probe captures,
    must rebuild `FrameContext.scene` for `viewCamera` before backend execution.
  - Compatibility contract: Software planar reflections may instead use an
    internal immutable mirrored view over the prepared main-view packets. That
    view must not mutate the application `Camera` or commit main-view temporal
    history.
- `IRenderBackend.beginFrame(context: FrameContext)`
  - Behavior contract: must prepare command encoders, bind presentation attachments, and transition frame state.
  - Constraint: must throw if another frame is already active or if the backend is uninitialized.
- `IRenderBackend.executePass(pass: FramePass, context: FrameContext)`
  - Behavior contract: must execute the commands for the given `FramePass`.
  - Constraint: must throw if no frame is active or `context` is not the active
    frame context.
- `IRenderBackend.skipPass(pass: FramePass)`
  - Behavior contract: called when a pass is disabled in the frame plan, allowing the backend to release/transition dependencies.
- `IRenderBackend.endFrame()`
  - Behavior contract: must finalize command encoders, submit command buffers in
    their recorded order, and present the frame.
  - Behavior contract: temporal histories, completed coverage, and custom render
    target publication must commit only after every frame command buffer and
    post-submit hook succeeds.
  - Error contract: a backend that submits more than one command buffer must
    distinguish a failure before any submission from a failure after partial
    submission. WebGPU must expose structured submitted and pending command
    labels for the latter case.
  - Constraint: must throw if no frame is active.
- `IRenderBackend.getCompletedFrameCoverage()`
  - Internal contract: must return `"dirty-tiles"` only when the completed output preserved every non-dirty tile.
  - Internal contract: must otherwise return `"full-frame"`.
- `IRenderBackend.abortFrame(error?: unknown)`
  - Behavior contract: must cancel/release active encoders and discard command buffers.
  - Constraint: must be idempotent and must not throw if no frame is active.
  - Constraint: before commit begins, must not present to the canvas, commit
    temporal history, or submit work. After a partial-submit failure, abort may
    discard only the remaining work and must not claim that submitted work was
    rolled back.
	- Backends must defer resize and shader runtime compilation updates while a frame is active.
	- Backend-internal frame-target recovery may synchronously select a safe
	  rendering configuration before the first render command is recorded.
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
  - Must expose a backend-agnostic `ProbeCaptureSource` API.
- `WEBGPU_COMPUTE_EXTENSION`
  - Must expose an `IWebGPUComputeFacade` API.
  - WebGPU compute callers must obtain buffer, texture, sampler, shader,
    binding, upload, and command capabilities through this extension.
  - `WebGPUBackend` must not expose backend-specific resource, pipeline,
    binding-group, or command-scheduler forwarding methods.
  - `WebGPUBackend` must not expose native `GPUDevice` or `GPUQueue` handles.
- `IBL_PREFILTER_EXECUTOR_EXTENSION`
  - Must expose a backend-owned generic executor API consumed by
    `IBLPrefilter`.
  - WebGPU and WebGL implementations must use the same request, availability,
    and CPU-backed mip result contract.
  - The WebGL implementation must delegate to context-scoped fragment-pass
    services without exposing native `WebGLTexture` handles.
  - The extension API object must remain identity-stable across WebGL context
    restoration.
- Identity Persistence:
  - Extension API objects must maintain the same object identity for the lifetime of the backend runtime.
- Device Loss Behavior:
  - During a device-lost state, extension operations must follow their
    documented retry policy. Explicit WebGL IBL work may wait for context
    restoration; extensions without a retry policy must throw a clear,
    descriptive error.
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
- Custom render targets currently support only `sampleCount = 1`; registration
  with another value must throw.
- Color attachments must use color formats, and depth attachments must use
  depth-only formats.
- `Renderer.renderTargets.readColor(id, attachmentIndex, options)` must return a
  `Promise<RenderTargetReadbackResult>`.
- `RenderTargetReadbackOptions` may restrict `width` and `height`, but must not
  reinterpret the attachment format or bytes-per-pixel layout.
- `RenderTargetReadbackResult.origin` must be `"top-left"` for WebGPU and
  `"bottom-left"` for WebGL.
- `readColor` must read only the most recent successfully completed frame.
- WebGL `readColor` must serialize through the backend context work queue and
  must reject while a renderer frame is active.
- `readColor` must reject before the first successful frame, after an invalid
  target id, for an invalid color attachment index, or when the requested
  dimensions exceed the target.
- `Renderer.renderPasses.register(descriptor)` must register a backend pass stage with the same `id`.
- `CustomRenderPassDescriptor.target` must reference a registered custom render target.
- Custom pass registration must be transactional. Failed target validation or
  pipeline-stage registration must not retain the pass descriptor or emit a
  registry change.
- A custom pass id must not replace a built-in pipeline stage.
- A render target referenced by a registered custom pass must not be
  unregistered until the dependent pass is unregistered.
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

### 4. Incremental Frame Coverage

```ts
const result = await renderer.renderFrame(performance.now());
const { plannedCoverage, finalOutputCoverage } = result.incremental;

// Ranges are row-major half-open tile indices.
console.info(plannedCoverage.updatedTileRanges);
console.info(finalOutputCoverage.reusableTileRanges);
```

### 5. Frame Coordinator Execution (Internal / Backend Implementation)

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

### 6. Querying Extensions

```ts
import { WEBGPU_COMPUTE_EXTENSION } from "ignisrenderer";

const compute = renderer.getBackendExtension(WEBGPU_COMPUTE_EXTENSION);
if (compute) {
	const buffer = compute.createBuffer({ size: 1024, usage: BufferUsage.Storage });
}
```

### 7. In-Frame Command Recording and Texture Copying

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

### 8. Custom Render Targets & Passes

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
- A failed or aborted frame must not publish a new incremental status or replace the last successful incremental stats snapshot.
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
- `Custom render pass "<id>" target "<target>" is not registered.` is thrown
  when registering a pass before its target.
- `Render target "<id>" is referenced by custom render pass "<pass>".` is
  thrown when unregistering a target that is still in use.
- `Render target "<id>" sampleCount must be 1.` is thrown for unsupported MSAA.
- `Render target "<id>" cannot be read before a successful frame completes.` is thrown when readback is requested too early.
- `software-custom-render-targets-unsupported` is logged when SoftwareBackend skips a custom render pass.

## Compatibility / Breaking Changes

- `Renderer` construction now requires a single `RendererOptions` object.
  The positional `new Renderer(backend, canvas, camera)` form is removed.
- `RenderTargetReadbackOptions.format` and `bytesPerPixel` are removed.
  `readColor` always returns the attachment's actual storage format.
- `Renderer.renderTargets` now rejects unsupported sample counts and invalid
  color/depth format kinds during registration instead of disabling the target
  during frame execution.
- `Renderer.renderPasses` now requires target-before-pass registration, and
  targets with dependent passes must be unregistered after those passes.
- `RenderFrameResult` requires an `incremental` property on both rendered and clean result branches. This is a breaking TypeScript API change.
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
- `WebGPUBackend.createBuffer`, `createTexture`, `createSampler`,
  `createShaderModule`, `createPipeline`, `createComputePipeline`,
  `createBindingGroup`, `createTextureView`, `createCommandEncoder`,
  `writeBuffer`, `copyTextureToTexture`, `submit`, `getTextureForSlot`,
  `registerExternalTexture`, and `unregisterExternalTexture` are removed.
  Applications requiring WebGPU compute or texture-bridge operations must use
  `WEBGPU_COMPUTE_EXTENSION`.
- `WebGPUBackend.device` and `WebGPUBackend.queue` are removed. Native WebGPU
  handles remain backend-private.
- `WebGPUBackend.getShaderDirectiveCacheTag`,
  `isOcclusionCullingEnabled`, `onDeviceLost`, `getFrameSceneTargetMode`,
  `captureProbeFace`, `getCurrentColorView`, `getCurrentDepthView`,
  `getTimestampDurationsMs`, and `createPassTimestampWrites` are removed.
  Backend lifecycle and command helpers remain internal, and probe capture must
  be requested through `PROBE_CAPTURE_EXTENSION`.
- `getNativeWebGPUCommandEncoder` is removed from `ICommandEncoder`. Code that requires ordered texture copies must use `copyTextureToTexture`. WebGPU-internal passes that need native WebGPU objects must resolve them through WebGPU-owned helpers instead of the shared renderer contract.
- SoftwareBackend does not support custom render targets.
- `SoftwareBackend` exposes only its scanline rasterization path.
  `SoftwareRasterMode`, `SoftwareTileOptions`, the `rasterMode` and `tile`
  backend options, and the `requestedRasterMode` and `activeRasterMode`
  properties are removed. Callers must construct `SoftwareBackend` without
  raster-mode configuration.
