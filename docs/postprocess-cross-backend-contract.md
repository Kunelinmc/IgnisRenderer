# Post-Process Cross-Backend Contract
## Scope
This document defines the cross-backend post-process abstraction used by `Renderer`, `PostProcessPipeline`, and render backends.

## Background
The renderer exposes a single `postprocess` frame stage. `PostProcessPipeline` schedules logical passes inside that stage using placement buckets and stable ordering, owns temporal history handles, validates logical G-buffer requirements, and dispatches work through pass-owned implementations when available. Backends own concrete GPU or CPU resources but do not own pass scheduling or history validity.

## API/Contract
- `PostProcessPassDescriptor.id` must identify one logical pass.
- `PostProcessPassDescriptor.placement` should identify where a custom pass enters the fixed post-process pipeline.
- `PostProcessPassDescriptor.placement` may be `"spatial"`, `"temporal"`, `"atmosphere"`, `"camera"`, `"hdr"`, `"ldr"`, `"overlay"`, or `"present"`.
- Custom passes that omit `PostProcessPassDescriptor.placement` must execute in the default `"overlay"` placement before `gamma`.
- `PostProcessPassDescriptor.order` may refine ordering within a placement bucket. It must not be used as a cross-placement dependency mechanism.
- `PostProcessPassDescriptor.requirements.gBuffer` must list required `LogicalGBufferSemantic` channels.
- `PostProcessPassDescriptor.history` must list temporal resources owned by `PostProcessPipeline`.
- `PostProcessPassDescriptor.resolveHistory(request)` may compute temporal resources owned by `PostProcessPipeline` for the current frame.
- If `PostProcessPassDescriptor.resolveHistory(request)` is present, `PostProcessPipeline` must use its returned descriptors instead of `PostProcessPassDescriptor.history`.
- `PostProcessHistoryResolveRequest` must include `frameContext`, resolved `postProcess` state, executor `backend`, `gBuffer`, and frame `width` and `height`.
- `PostProcessPassDescriptor.implementations` must map backend kinds to backend-specific implementation metadata or pass-owned implementations.
- `PostProcessPassImplementation.execute(request, context)` may execute a pass directly when backend-specific logic is owned by the logical pass.
- `PostProcessPassImplementation.warmup(context)` may allocate backend resources required by a pass-owned implementation.
- `PostProcessPassImplementation.invalidate()` may release frame-size dependent implementation resources.
- `PostProcessPassImplementation.destroy()` may release all implementation-owned resources for one backend implementation.
- `PostProcessPipeline` must call `PostProcessPassImplementation.execute(request, context)` when it is present.
- `PostProcessPipeline` must fall back to `IPostProcessExecutor.executePass(passId, request)` when `PostProcessPassImplementation.execute` is absent.
- Backend warmup must call `PostProcessPassImplementation.warmup(context)` for planned pass-owned implementations when the method is present.
- `PostProcessPassRegistry.invalidatePasses(backend)` must call `PostProcessPass.invalidate(backend)` on registered passes without changing pass enabled state, options, or ordering.
- `PostProcessPassRegistry.destroyPasses(backend)` must call `PostProcessPass.destroy(backend)` on registered passes without changing pass enabled state, options, or ordering.
- `PostProcessPassRegistry.unregisterPass(id)` must destroy the removed pass implementations after detaching change listeners.
- Built-in post-process order must be `ssao`, `ssgi`, `taa`, `ssr`, `volumetric`, `fog`, `motion-blur`, `dof`, `bloom`, `tonemap`, `color-filter`, `fxaa`, `interaction-outline`, `gamma`.
- `IPostProcessExecutor.backend` must identify the active backend kind.
- `IPostProcessExecutor.capabilities` must expose the same capability set used by `resolvePostProcessState`.
- `IPostProcessExecutor.createResource(desc)` must allocate a concrete resource and return a `PostProcessResourceHandle`.
- `IPostProcessExecutor.destroyResource(handle)` must release resources allocated by `createResource(desc)`.
- `IPostProcessExecutor.getPassExecutionContext(passId, request)` may return backend-specific low-level helpers for pass-owned implementations.
- `IPostProcessExecutor.executePass(passId, request)` must execute one high-level logical pass when no pass-owned implementation handles it.
- `PostProcessPassRequest.implementation` must contain the implementation metadata selected for `IPostProcessExecutor.backend`.
- WebGPU executors should expose WebGPU context helpers and may dispatch WGSL compute or render work through `executePass(passId, request)` only for non-pass-owned fallback passes.
- WebGL executors should expose WebGL context helpers and may dispatch GLSL fullscreen work through `executePass(passId, request)` only for non-pass-owned fallback passes.
- Software executors should expose CPU post-process helpers and may dispatch optimized CPU loops through `executePass(passId, request)` only for non-pass-owned fallback passes.
- `LogicalGBufferBridge` must describe semantic channels and must not expose a cross-backend low-level read/write API.
- Software `FrameAttachments.motionBuffer` must store `motion-depth` data as `float32x4` when a pass requires the `motion` semantic.
- `LogicalGBufferBridge.worldPosition.source` must be `"derived"` unless a future contract explicitly defines a physical world-position channel.
- `PostProcessPipeline` must invalidate temporal histories on camera signature changes, feature signature changes, explicit temporal resets, and resize.
- `PostProcessPipeline` must recreate temporal resources only when dimensions, format, usage, or backend kind changes.
- The built-in `taa` pass must own its WebGPU, WebGL, and Software implementations under `src/postprocess/passes/`.
- The built-in `fxaa` pass must own its WebGPU, WebGL, and Software implementations under `src/postprocess/passes/`.
- The built-in `ssao` pass must own its WebGPU, WebGL, and Software implementations under `src/postprocess/passes/`.
- The built-in `ssgi` pass must own its WebGPU implementation under `src/postprocess/passes/`.
- The built-in `ssr` pass must own its WebGPU implementation under `src/postprocess/passes/`.
- The built-in `volumetric` pass must own its WebGPU and Software implementations under `src/postprocess/passes/`.
- The built-in `fog` pass must own its WebGPU and WebGL implementations under `src/postprocess/passes/`.
- The built-in `bloom` pass must own its WebGPU and WebGL implementations under `src/postprocess/passes/`.
- The built-in `motion-blur` pass must own its WebGPU and WebGL implementations under `src/postprocess/passes/`.
- The built-in `dof` pass must own its WebGPU and WebGL implementations under `src/postprocess/passes/`.
- The built-in `tonemap` pass must own its WebGPU, WebGL, and Software implementations under `src/postprocess/passes/`.
- The built-in `color-filter` pass must own its WebGPU, WebGL, and Software implementations under `src/postprocess/passes/`.
- The built-in `interaction-outline` pass must own its WebGPU, WebGL, and Software implementations under `src/postprocess/passes/`.
- The built-in `gamma` pass must own final presentation for WebGPU and WebGL and gamma encoding for Software under `src/postprocess/passes/`.
- Backend executor fallback dispatch and runtime pass registration must not contain backend-private `ssao`, `ssgi`, `taa`, `fxaa`, `ssr`, `volumetric`, `fog`, `bloom`, `motion-blur`, `dof`, `tonemap`, `color-filter`, `interaction-outline`, or `gamma` kernel orchestration.
- The frame-level incremental planner must return `firstPass: "postprocess"` for post-process-only work and must store the internal starting pass in `postProcessStartPass`.

## Usage
```ts
const descriptor = {
	id: "custom-soft-glow",
	placement: "hdr",
	order: 10,
	requirements: {
		gBuffer: ["color", "depth"],
	},
	history: [
		{
			id: "custom-soft-glow",
			format: "rgba16float",
			usage: ["sampled", "storage", "render-target"],
		},
	],
	isEnabled(state) {
		return state.enabled["custom-soft-glow"] === true;
	},
	implementations: {
		webgpu: { id: "custom-soft-glow" }, // falls back to executor.executePass
		webgl: { id: "custom-soft-glow" },
		software: { id: "custom-soft-glow" },
	},
};

renderer.postProcess.registerPass(descriptor);
renderer.postProcess.enable("custom-soft-glow");
```

```ts
const dynamicHistoryDescriptor = {
	id: "custom-half-res-history",
	placement: "temporal",
	resolveHistory(request) {
		const halfRes =
			request.postProcess.options["custom-half-res-history"]?.halfRes === true;
		return [{
			id: "custom-half-res-history",
			widthScale: halfRes ? 0.5 : 1,
			heightScale: halfRes ? 0.5 : 1,
			format: "rgba16float",
			usage: ["sampled", "storage", "render-target"],
		}];
	},
	isEnabled(state) {
		return state.enabled["custom-half-res-history"] === true;
	},
	implementations: {
		webgpu: { id: "custom-half-res-history" },
	},
};
```

```bash
bun tests/test_postprocess_public_api.mjs
bun tests/test_screen_space_ambient_occlusion_pass.mjs
bun tests/test_screen_space_global_illumination_pass.mjs
bun tests/test_temporal_anti_aliasing_pass.mjs
```

## Errors & Diagnostics
- `postprocess-requirement-missing-<passId>` must be emitted when required logical G-buffer channels are unavailable.
- `Unknown post-process pass "<id>".` must be thrown when a caller enables an unregistered custom pass id.
- `Cannot register built-in post-process pass "<id>" as a custom pass.` must be thrown when a custom descriptor uses a built-in id.

## Compatibility / Breaking Changes
- Backend-specific public post-process graph registration is removed.
- `PostProcessPassDescriptor.dependsOn` is removed. Custom passes must use `placement` and optional `order`.
- `IPostProcessExecutor.getPassExecutionContext(passId, request)` is added for pass-owned implementations.
- `PostProcessPassImplementation.execute(request, context)` is added and takes precedence over backend executor dispatch.
- `PostProcessPassImplementation.warmup(context)` is added for pass-owned warmup.
- `PostProcessPassRegistry.invalidatePasses(backend)` is added for pass-owned implementation invalidation.
- `PostProcessPassRegistry.destroyPasses(backend)` is added for pass-owned implementation destruction.
- `PostProcessPassDescriptor.resolveHistory(request)` is added and takes precedence over static `history`.
- `WebGPUPostProcessPassPlugin` is no longer a public extension type.
- `WebGLPostProcessPassPlugin` is no longer a public extension type.
- Code that previously depended on per-pass frame stages such as `ssao`, `taa`, or `gamma` must use the single `postprocess` frame stage and inspect `IncrementalFrameContext.postProcessStartPass` for the internal pass start.
