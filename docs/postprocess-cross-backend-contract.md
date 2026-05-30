# Post-Process Cross-Backend Contract
## Scope
This document defines the cross-backend post-process abstraction used by `Renderer`, `PostProcessPipeline`, and render backends.

## Background
The renderer exposes a single `postprocess` frame stage. `PostProcessPipeline` schedules logical passes inside that stage using placement buckets, pass-owned frame predicates, and stable ordering, owns temporal history handles, validates logical G-buffer requirements, and dispatches work through pass-owned implementations when available. Backends own concrete GPU or CPU resources but do not own pass scheduling or history validity.

## API/Contract
- `PostProcessPass.id` must identify one logical pass.
- `PostProcessPass.placement` should identify where a custom pass enters the fixed post-process pipeline.
- `PostProcessPass.placement` may be `"spatial"`, `"temporal"`, `"atmosphere"`, `"camera"`, `"hdr"`, `"ldr"`, `"overlay"`, or `"present"`.
- Custom passes that omit `PostProcessPass.placement` must execute in the default `"overlay"` placement before `gamma`.
- `PostProcessPass.order` may refine ordering within a placement bucket. It must not be used as a cross-placement dependency mechanism.
- `PostProcessPass.getRequirements(request).gBuffer` must list required `LogicalGBufferSemantic` channels.
- `PostProcessPass.getHistoryDescriptors(request)` must list temporal resources owned by `PostProcessPipeline`.
- `PostProcessPass.getTransientResourceDescriptors(request)` must list single-frame resources owned by `PostProcessPipeline`.
- `PostProcessPass.getTransientResourceDescriptors(request)` must return an empty list when a pass does not require transient resources for `request.backend`.
- `PostProcessPass.shouldExecute(request)` may exclude an enabled snapshot pass from a specific frame without changing registry enabled state.
- `PostProcessPass.shouldExecute(request)` must be deterministic for the supplied `request` and must not allocate backend resources.
- `PostProcessPass.shouldExecute(request)` must return `true` by default for custom passes that do not override it.
- `resolvePostProcessExecutionOrder(postProcess, context)` must apply `PostProcessPass.shouldExecute(request)` before sorting passes.
- `hasPostProcessExecutionPasses(postProcess, context)` must use the same frame predicate as `resolvePostProcessExecutionOrder(postProcess, context)`.
- Renderer frame planners and backend prevalidation planners must use `hasPostProcessExecutionPasses(postProcess, context)` or `resolvePostProcessExecutionOrder(postProcess, context)` and must not duplicate built-in post-process pass id enablement lists.
- `PostProcessHistoryResolveRequest` must include `frameContext`, resolved `postProcess` state, executor `backend`, `gBuffer`, and frame `width` and `height`.
- `PostProcessPassConfig.implementations` must map backend kinds to backend-specific pass-owned implementations.
- `PostProcessPassImplementation.metadata.context` may declare backend-specific context requirements for pass-owned implementations.
- `PostProcessPassImplementation.metadata.warmupHints` may declare backend-specific runtime warmup hint ids.
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
- The built-in `fog` pass must return `false` from `shouldExecute(request)` when `request.options.application` is `"scene"`.
- The built-in `interaction-outline` pass must return `false` from `shouldExecute(request)` when `request.frameContext` exists and no entity is selected.
- `IPostProcessExecutor.backend` must identify the active backend kind.
- `IRenderBackend` must define only the core render backend lifecycle and pass execution surface.
- `IRenderBackend` must not expose `postProcessExecutor` or `createPostProcessGBufferBridge(context)`.
- `PostProcessBackendAdapter.backend` must identify the backend kind used for pass implementation resolution.
- `PostProcessBackendAdapter.executor` must supply the `IPostProcessExecutor` for `PostProcessPipeline`.
- `PostProcessBackendAdapter.createGBufferBridge(context)` must create the logical G-buffer view consumed by cross-backend passes.
- `registerPostProcessBackendAdapter(owner, adapter)` must associate one adapter with a backend or host object.
- `resolvePostProcessBackendAdapter(owner)` must return the registered adapter or `null`.
- `unregisterPostProcessBackendAdapter(owner)` must remove the registered adapter.
- `Renderer` must accept any `IRenderBackend` and must resolve post-process execution through the adapter registry.
- Built-in Software, WebGL, and WebGPU backends must register their post-process adapters during construction.
- When enabled post-process work exists but no adapter is registered, `Renderer` must emit `"<backend>-postprocess-adapter-missing"` once and skip the `postprocess` stage.
- Backend support for a built-in pass must be derived from `PostProcessPassConfig.implementations`, not from a backend-owned capability map.
- Backends must not expose `postProcessCapabilities`.
- `IPostProcessExecutor.createResource(desc)` must allocate a concrete resource and return a `PostProcessResourceHandle`.
- `IPostProcessExecutor.destroyResource(handle)` must release resources allocated by `createResource(desc)`.
- `IPostProcessExecutor.invalidateResourceBindings()` may invalidate backend bind-group or descriptor caches after transient resources are recreated.
- `IPostProcessExecutor.getPassExecutionContext(request)` may return backend-specific low-level helpers for pass-owned implementations based on `PostProcessPassImplementation.metadata.context`.
- `IPostProcessExecutor.executePass(passId, request)` must execute one high-level logical pass when no pass-owned implementation handles it.
- `IPostProcessExecutor.completePass(request, result)` may apply backend-owned side effects recorded during one logical pass.
- `IPostProcessExecutor.completePass(request, result)` must validate backend-owned resources before publishing them into frame state.
- `PostProcessPipeline` must call `IPostProcessExecutor.completePass(request, result)` after each pass returns, including passes that return `ran: false`.
- `IPostProcessExecutor.abortFrame(request)` may release backend post-process frame state after a failed logical pass or failed renderer frame.
- `IPostProcessExecutor.abortFrame(request)` must be idempotent and must not present, copy history, or commit temporal histories.
- `PostProcessPassRequest.implementation` must contain the implementation metadata selected for `IPostProcessExecutor.backend`, or `null` when the pass falls back to `IPostProcessExecutor.executePass(passId, request)`.
- `PostProcessPassRequest.transients` must contain the current frame's transient slots keyed by transient id.
- `PostProcessPassExecutionContextRequest` must contain the full `PostProcessPassRequest` contract and a non-null `implementation`.
- `PostProcessPipeline` must call `IPostProcessExecutor.getPassExecutionContext(request)` only when the selected implementation exposes `execute()`.
- Backends must use `PostProcessPassExecutionContextRequest.implementation` and its `metadata.context` as the execution context contract; they must not infer context shape from pass id strings.
- `PostProcessPassExecutionContextRequest.pass.builtIn` must classify engine-owned passes and must not be required for metadata-driven backend context packing.
- WebGPU executors should expose WebGPU context helpers and may dispatch WGSL compute or render work through `executePass(passId, request)` only for non-pass-owned fallback passes.
- WebGL executors should expose WebGL context helpers and may dispatch GLSL fullscreen work through `executePass(passId, request)` only for non-pass-owned fallback passes.
- Software executors should expose CPU post-process helpers and may dispatch optimized CPU loops through `executePass(passId, request)` only for non-pass-owned fallback passes.
- `LogicalGBufferBridge` must describe semantic channels and must not expose a cross-backend low-level read/write API.
- Software `FrameAttachments.motionBuffer` must store `motion-depth` data as `float32x4` when a pass requires the `motion` semantic.
- `LogicalGBufferBridge.worldPosition.source` must be `"derived"` unless a future contract explicitly defines a physical world-position channel.
- `PostProcessPipeline` must invalidate temporal histories on camera signature changes, feature signature changes, explicit temporal resets, and resize.
- `PostProcessPipeline` must recreate temporal resources only when dimensions, format, usage, or backend kind changes.
- `PostProcessPipeline` must recreate transient resources only when dimensions, format, usage, mip mode, or backend kind changes.
- `PostProcessPipeline` must collect history and transient descriptors only from passes whose G-buffer requirements are satisfied.
- `PostProcessPipeline` must keep temporal history validity separate from transient resource lifetime.
- `PostProcessPipeline.execute()` must commit temporal history automatically unless `historyFinalization` is `"manual"`.
- `PostProcessPipeline.commitFrame()` must swap pending updated histories after a successful renderer frame.
- `PostProcessPipeline.abortFrame(error?)` must clear pending history updates without invalidating previously valid histories.
- `PostProcessPipeline` must call `IPostProcessExecutor.abortFrame(request)` when a logical post-process frame fails after frame state has been prepared.
- `PostProcessPipeline.destroy(executor)` must clear pending frame state, destroy
  active temporal history handles, destroy active transient handles, and reset
  history signatures for backend lifecycle resets.
- Backends that release a graphics device or context must notify
  `RendererBackendBridge.onBackendResourceEvent({ resource: "postprocess", action: "destroy" })`
  before destroying the executor resources that own post-process handles.
- Backends that recreate frame targets without destroying the graphics context must notify
  `RendererBackendBridge.onBackendResourceEvent({ resource: "postprocess", action: "invalidate" })`.
- `PostProcessResourceDescriptor.mipMode` may be `"single"` or `"full-chain"`, and omitted values must behave as `"single"`.
- `PostProcessTransientManager` must destroy transient resources that are not requested by the current eligible pass set.
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
- Software built-in screen pass CPU runtimes must live under `src/postprocess/passes/` and must not be owned by `src/renderers/software/`.
- Backend executor fallback dispatch and runtime pass registration must not contain backend-private `ssao`, `ssgi`, `taa`, `fxaa`, `ssr`, `volumetric`, `fog`, `bloom`, `motion-blur`, `dof`, `tonemap`, `color-filter`, `interaction-outline`, or `gamma` kernel orchestration.
- The frame-level incremental planner must return `firstPass: "postprocess"` for post-process-only work and must store the internal starting pass in `postProcessStartPass`.

## Usage
```ts
import {
	PostProcessPass,
	type PostProcessHistoryDescriptor,
	type PostProcessPassResolveRequest,
	type PostProcessTransientDescriptor,
} from "ignisrenderer";

interface SoftGlowOptions {
	halfRes?: boolean;
	strength?: number;
}

class CustomSoftGlowPass extends PostProcessPass<
	SoftGlowOptions,
	Required<SoftGlowOptions>
> {
	public constructor() {
		super({
			id: "custom-soft-glow",
			placement: "hdr",
			order: 10,
			enabled: true,
			options: {
				halfRes: true,
				strength: 0.75,
			},
			implementations: {
				webgpu: { id: "custom-soft-glow:webgpu" },
				webgl: { id: "custom-soft-glow:webgl" },
				software: { id: "custom-soft-glow:software" },
			},
		});
	}

	public override normalizeOptions(): Required<SoftGlowOptions> {
		const raw = this.getRawOptions();
		return {
			halfRes: raw.halfRes === true,
			strength: raw.strength ?? 0.75,
		};
	}

	public override shouldExecute(
		request: PostProcessPassResolveRequest<Required<SoftGlowOptions>>
	): boolean {
		return request.options.strength > 0;
	}

	public override getRequirements() {
		return { gBuffer: ["color", "depth"] as const };
	}

	public override getHistoryDescriptors(
		request: PostProcessPassResolveRequest<Required<SoftGlowOptions>>
	): readonly PostProcessHistoryDescriptor[] {
		return [{
			id: "custom-soft-glow",
			widthScale: request.options.halfRes ? 0.5 : 1,
			heightScale: request.options.halfRes ? 0.5 : 1,
			format: "rgba16float",
			usage: ["sampled", "storage", "render-target"],
		}];
	}

	public override getTransientResourceDescriptors(
		request: PostProcessPassResolveRequest<Required<SoftGlowOptions>>
	): readonly PostProcessTransientDescriptor[] {
		return [{
			id: "custom-soft-glow:temp",
			widthScale: request.options.halfRes ? 0.5 : 1,
			heightScale: request.options.halfRes ? 0.5 : 1,
			format: "rgba16float",
			usage: ["sampled", "storage", "render-target"],
		}];
	}
}

renderer.postProcess.registerPass(new CustomSoftGlowPass());
```

```bash
bun tests/test_postprocess_public_api.mjs
bun tests/test_screen_space_ambient_occlusion_pass.mjs
bun tests/test_screen_space_global_illumination_pass.mjs
bun tests/test_temporal_anti_aliasing_pass.mjs
bun tests/test_webgpu_postprocess_runtime_temporal.mjs
```

## Errors & Diagnostics
- `postprocess-requirement-missing-<passId>` must be emitted when required logical G-buffer channels are unavailable.
- `renderer.postProcess.registerPass(pass)` must throw when `pass` is not a `PostProcessPass`.
- `renderer.postProcess.registerPass(pass)` must throw when `pass.id` is already registered.
- `postprocess-transient-conflict-<transientId>` must be emitted when eligible passes request incompatible descriptors for the same transient id.

## Compatibility / Breaking Changes
- Backend-specific public post-process graph registration is removed.
- `IPostProcessExecutor.capabilities`, `PostProcessCapabilities`, and backend `postProcessCapabilities` are removed. Backend post-process support must be declared through pass-owned implementations.
- `PostProcessPassDescriptor.dependsOn` is removed. Custom passes must use `placement` and optional `order`.
- `IPostProcessExecutor.getPassExecutionContext(request)` is added for pass-owned implementations.
- `PostProcessPassImplementation.execute(request, context)` is added and takes precedence over backend executor dispatch.
- `IPostProcessExecutor.completePass(request, result)` is added for backend-owned pass-boundary side effects.
- `PostProcessPassImplementation.warmup(context)` is added for pass-owned warmup.
- `PostProcessPassRegistry.invalidatePasses(backend)` is added for pass-owned implementation invalidation.
- `PostProcessPassRegistry.destroyPasses(backend)` is added for pass-owned implementation destruction.
- `PostProcessPass.getHistoryDescriptors(request)` replaces static descriptor `history` and `resolveHistory` fields.
- `PostProcessPass.getTransientResourceDescriptors(request)` is added for single-frame pipeline-owned resources.
- `PostProcessPassRequest.transients` is added for transient resource access.
- `PostProcessResourceDescriptor.mipMode` is added for single-mip and full-chain resources.
- `PostProcessPass.shouldExecute(request)` is added for pass-owned frame-level execution predicates.
- `PostProcessBackendSupport` and `PostProcessCapableRenderBackend` are removed. Code that needs backend post-process execution must use `PostProcessBackendAdapter` and the adapter registry.
- Public `backend.postProcessExecutor` and `backend.createPostProcessGBufferBridge(context)` are removed from built-in backends.
- `PostProcessor` is removed from the public API. Software built-in post-process behavior is owned by pass implementations under `src/postprocess/passes/`.
- `WebGPUPostProcessPassPlugin` is no longer a public extension type.
- `WebGLPostProcessPassPlugin` is no longer a public extension type.
- Code that previously depended on per-pass frame stages such as `ssao`, `taa`, or `gamma` must use the single `postprocess` frame stage and inspect `IncrementalFrameContext.postProcessStartPass` for the internal pass start.
