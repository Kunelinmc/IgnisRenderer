# Post-Process Public API Migration Contract
## Scope
This document defines the breaking migration from request-map post-processing to pass-owned post-processing.

## Background
Post-process state previously lived in `renderer.postProcess` as enable and option maps. The renderer now exposes `PostProcessPassRegistry` as a registry and lookup surface. Each `PostProcessPass` instance owns enabled state, raw options, option normalization, requirements, frame-level execution predicates, history descriptors, warmup metadata, backend implementation selection, execution, invalidation, and destruction.

## API/Contract
- `renderer.postProcess` must be a `PostProcessPassRegistry`.
- `renderer.postProcess.registerPass(pass)` must accept only `PostProcessPass` instances.
- `renderer.postProcess.unregisterPass(id)` must remove the registered pass for `id`.
- `renderer.postProcess.getPass(id)` must return the registered pass instance or `null`.
- `renderer.postProcess.getPasses()` must return registered pass instances.
- `renderer.postProcess` must not expose `enable`, `disable`, `setOptions`, `reset`, or `getState`.
- Pass mutation must use `pass.enable(options)`, `pass.disable()`, `pass.setEnabled(enabled)`, `pass.setOptions(options)`, or `pass.resetOptions()`.
- `PostProcessPass.builtIn` must identify engine-provided built-in passes.
- `PostProcessPass.warningLabel` must provide the human-readable pass name used in diagnostics.
- `PostProcessPass.shouldExecute(request)` may exclude an enabled snapshot pass from a specific frame without changing registry enabled state.
- `PostProcessPass.shouldExecute(request)` must be deterministic for the supplied `request` and must not allocate backend resources.
- Built-in pass classes must define their ids, ordering, implementations, and diagnostic labels internally.
- Custom `PostProcessPassConfig.warningLabel` may provide a human-readable custom pass name; when omitted, diagnostics must use `PostProcessPass.id`.
- `Renderer` must auto-register only `ToneMappingPass` and `GammaPass`, both enabled by default.
- Other built-in passes must be explicitly registered before they can run.
- `FrameContext.postProcess` must be a `PostProcessPassRegistrySnapshot`.
- Snapshot consumers must use `postProcess.isEnabled(id)` and `postProcess.getOptions(id)`.
- Snapshot consumers must not use `postProcess.enabled` or `postProcess.options` maps.
- `PostProcessPipeline` must execute enabled snapshot passes in built-in placement order and custom placement order.
- Built-in pass classes must expose pass-owned normalization, requirements, history descriptors, and implementations when migrated.
- Backends used with `Renderer` must satisfy `IRenderBackend`.
- Backends that support post-processing must register a `PostProcessBackendAdapter` with `registerPostProcessBackendAdapter(owner, adapter)`.
- `PostProcessBackendAdapter` must expose the `IPostProcessExecutor` methods and `createGBufferBridge(context)`.
- `Renderer` must resolve post-process execution with `resolvePostProcessBackendAdapter(backend)`.
- `IRenderBackend` must not expose `postProcessExecutor` or `createPostProcessGBufferBridge(context)`.
- Backends must not expose `postProcessCapabilities`.
- Backends must not expose public post-process graph registration APIs.
- Unsupported enabled built-in passes must be determined by missing pass-owned backend implementations and must emit warning key `"<backend>-postprocess-unsupported-<passId>"`.

## Usage
```ts
import {
	GammaPass,
	ScreenSpaceAmbientOcclusionPass,
	ToneMappingPass,
} from "ignisrenderer";

const ssao = new ScreenSpaceAmbientOcclusionPass({
	enabled: true,
	options: { samples: 24, radius: 3 },
});

renderer.postProcess.registerPass(ssao);
ssao.setOptions({ radius: 4 });
ssao.disable();

renderer.postProcess.getPass<GammaPass>("gamma")?.disable();
renderer.postProcess.getPass<ToneMappingPass>("tonemap")?.enable();
```

```ts
import {
	PostProcessPass,
	type PostProcessPassResolveRequest,
} from "ignisrenderer";

class CustomEdgePass extends PostProcessPass<
	{ strength?: number },
	{ strength: number }
> {
	public constructor() {
		super({
			id: "custom-edge",
			placement: "ldr",
			warningLabel: "custom edge",
			order: 5,
			enabled: true,
			options: { strength: 0.75 },
			implementations: {
				webgpu: { id: "custom-edge:webgpu" },
			},
		});
	}

	public override normalizeOptions(): { strength: number } {
		return { strength: this.getRawOptions().strength ?? 0.75 };
	}

	public override shouldExecute(
		request: PostProcessPassResolveRequest<{ strength: number }>
	): boolean {
		return request.options.strength > 0;
	}
}

renderer.postProcess.registerPass(new CustomEdgePass());
```

```bash
bun tests/static/postprocess/test_postprocess_public_api.mjs
```

## Errors & Diagnostics
- `renderer.postProcess.registerPass(pass)` must throw when `pass` is not a `PostProcessPass`.
- `renderer.postProcess.registerPass(pass)` must throw when `pass.id` is already registered.
- `"<backend>-postprocess-unsupported-<passId>"` must be emitted when an enabled built-in pass has no implementation for the active backend.
- `postprocess-history-conflict-<historyId>` must be emitted when enabled passes request incompatible descriptors for the same history id.
- `"<backend>-postprocess-adapter-missing"` must be emitted once when enabled post-process work exists but the active backend has no registered adapter.

## Compatibility / Breaking Changes
- Plain object pass descriptors are no longer accepted.
- `PostProcessPassDescriptor` is removed.
- `PostProcessController` is removed.
- `PostProcessCapabilities` and backend `postProcessCapabilities` are removed. Pass support is derived from pass-owned backend implementations.
- `POST_PROCESS_PASS_IDS` is removed. Code must inspect registered `PostProcessPass` instances instead of relying on a global id list.
- `PostProcessOptionsMap` is removed. Code must read typed options through the concrete pass or `PostProcessPassRegistrySnapshot.getOptions<TOptions>(id)`.
- `getPostProcessWarningLabel(id)` is removed. Diagnostics must use `PostProcessPass.warningLabel`.
- `resolvePostProcessState()` is removed.
- `renderer.postProcess.enable(id, options)` is removed; construct or look up a pass and call `pass.enable(options)`.
- `renderer.postProcess.disable(id)` is removed; call `renderer.postProcess.getPass(id)?.disable()`.
- `renderer.postProcess.setOptions(id, options)` is removed; call `renderer.postProcess.getPass(id)?.setOptions(options)`.
- `renderer.postProcess.reset(id)` and `renderer.postProcess.reset()` are removed.
- `renderer.postProcess.getState()` is removed.
- Only `tonemap` and `gamma` are auto-registered.
- `FrameContext.postProcess.enabled` and `FrameContext.postProcess.options` are removed.
- `PostProcessBackendSupport` and `PostProcessCapableRenderBackend` are removed.
- `PostProcessBackendAdapter.executor` is removed. Use the resolved adapter object as the executor.
- Public `backend.postProcessExecutor` and `backend.createPostProcessGBufferBridge(context)` are removed; use `resolvePostProcessBackendAdapter(backend)` for backend-owned post-process execution internals.
- `renderer.features.enableSSAO = true` must migrate to `renderer.postProcess.registerPass(new ScreenSpaceAmbientOcclusionPass({ enabled: true }))`.
- `renderer.features.enableSSGI = true` must migrate to `renderer.postProcess.registerPass(new ScreenSpaceGlobalIlluminationPass({ enabled: true }))`.
- `renderer.features.enableTAA = true` must migrate to `renderer.postProcess.registerPass(new TemporalAntiAliasingPass({ enabled: true }))`.
- `renderer.features.enableSSR = true` must migrate to `renderer.postProcess.registerPass(new ScreenSpaceReflectionsPass({ enabled: true }))`.
- `renderer.features.enableFXAA = true` must migrate to `renderer.postProcess.registerPass(new FastApproximateAntiAliasingPass({ enabled: true }))`.
- `renderer.features.enableGamma = false` must migrate to `renderer.postProcess.getPass("gamma")?.disable()`.
