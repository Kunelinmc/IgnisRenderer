# Post-Process Public API Migration Contract
## Scope
This document defines the breaking migration from request-map post-processing and
renderer-owned post-process execution to pass-owned logical post-processing with
backend-owned runtime lifecycle.

## Background
`renderer.postProcess` remains the public pass registry. Each `PostProcessPass`
instance owns enabled state, raw options, option normalization, requirements,
frame predicates, history descriptors, warmup metadata, backend implementation
selection, invalidation, and destruction.

`Renderer` no longer owns post-process execution. The logical `postprocess`
stage is now a backend pass. Backends opt in through
`BackendCapabilities.postProcess` and execute the graph through their internal
`BackendPostProcessRuntime`.

## API/Contract
- `renderer.postProcess` must be a `PostProcessPassRegistry`.
- `renderer.postProcess.registerPass(pass)` must accept only
  `PostProcessPass` instances.
- `renderer.postProcess.unregisterPass(id)` must remove the registered pass for
  `id`.
- `renderer.postProcess.getPass(id)` must return the registered pass instance or
  `null`.
- `renderer.postProcess.getPasses()` must return registered pass instances.
- `renderer.postProcess` must not expose `enable`, `disable`, `setOptions`,
  `reset`, or `getState`.
- Pass mutation must use `pass.enable(options)`, `pass.disable()`,
  `pass.setEnabled(enabled)`, `pass.setOptions(options)`, or
  `pass.resetOptions()`.
- `PostProcessPass.builtIn` must be `true` only for renderer-default built-in
  passes: `tonemap` and `gamma`.
- `PostProcessPass.warningLabel` must provide the human-readable pass name used
  in diagnostics.
- `PostProcessPass.shouldExecute(request)` may exclude an enabled snapshot pass
  from a specific frame without changing registry enabled state.
- `PostProcessPass.shouldExecute(request)` must be deterministic and must not
  allocate backend resources.
- Engine-provided pass classes must define their ids, ordering, incremental
  metadata, implementations, and diagnostic labels internally.
- Custom `PostProcessPassConfig.warningLabel` may provide a human-readable
  custom pass name.
- `Renderer` must auto-register only `ToneMappingPass` and `GammaPass`, both
  enabled by default.
- Other engine-provided pass classes must be explicitly registered before they
  can run and must be treated as manually registered passes.
- `FrameContext.postProcess` must be a `PostProcessPassRegistrySnapshot`.
- Snapshot consumers must use `postProcess.isEnabled(id)` and
  `postProcess.getOptions(id)`.
- Snapshot consumers must not use `postProcess.enabled` or `postProcess.options`
  maps.
- Enabled logical passes must execute through the `"postprocess"` backend pass
  when the backend supports post-processing.
- Backends used with `Renderer` must satisfy `IRenderBackend`.
- Backends that support post-processing must set
  `BackendCapabilities.postProcess = true`.
- Backends that support post-processing must handle
  `executePass({ stage: "postprocess" }, context)`.
- Backends that do not support post-processing must set
  `BackendCapabilities.postProcess = false`.
- `IRenderBackend` must not expose `postProcessExecutor`,
  `createPostProcessGBufferBridge(context)`, or `postProcessAdapter`.
- Backends must not expose public post-process graph registration APIs.
- Unsupported enabled renderer-default built-in passes must be determined by
  missing pass-owned backend implementations and must emit warning key
  `"<backend>-postprocess-unsupported-<passId>"`.

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
bun tests/static/postprocess/test_renderer_postprocess_registry.mjs
```

## Errors & Diagnostics
- `renderer.postProcess.registerPass(pass)` must throw when `pass` is not a
  `PostProcessPass`.
- `renderer.postProcess.registerPass(pass)` must throw when `pass.id` is already
  registered.
- `"<backend>-postprocess-unsupported-<passId>"` must be emitted when an enabled
  renderer-default built-in pass has no implementation for the active backend.
- `postprocess-history-conflict-<historyId>` must be emitted when eligible passes
  request incompatible descriptors for the same history id.
- `postprocess-transient-conflict-<transientId>` must be emitted when eligible
  passes request incompatible descriptors for the same transient id.

## Compatibility / Breaking Changes
- Plain object pass descriptors are no longer accepted.
- `PostProcessPassDescriptor` is removed.
- `PostProcessController` is removed.
- `RendererPostProcessController` is removed.
- `PostProcessCapabilities` and backend `postProcessCapabilities` are removed.
- `BackendCapabilities.postProcess` is added.
- `POST_PROCESS_PASS_IDS` is removed. Code must inspect registered
  `PostProcessPass` instances instead of relying on a global id list.
- `PostProcessOptionsMap` is removed. Code must read typed options through the
  concrete pass or `PostProcessPassRegistrySnapshot.getOptions<TOptions>(id)`.
- `getPostProcessWarningLabel(id)` is removed. Diagnostics must use
  `PostProcessPass.warningLabel`.
- `resolvePostProcessState()` is removed.
- `renderer.postProcess.enable(id, options)` is removed; construct or look up a
  pass and call `pass.enable(options)`.
- `renderer.postProcess.disable(id)` is removed; call
  `renderer.postProcess.getPass(id)?.disable()`.
- `renderer.postProcess.setOptions(id, options)` is removed; call
  `renderer.postProcess.getPass(id)?.setOptions(options)`.
- `renderer.postProcess.reset(id)` and `renderer.postProcess.reset()` are
  removed.
- `renderer.postProcess.getState()` is removed.
- Only `tonemap` and `gamma` are auto-registered and report
  `PostProcessPass.builtIn === true`.
- Engine-provided post-process classes other than `ToneMappingPass` and
  `GammaPass` report `PostProcessPass.builtIn === false` and may be
  unregistered after manual registration.
- `FrameContext.postProcess.enabled` and `FrameContext.postProcess.options` are
  removed.
- `PostProcessBackendSupport` and `PostProcessCapableRenderBackend` are removed.
- `renderer.postprocess` backend extension APIs are removed, including
  `RENDERER_POST_PROCESS_EXTENSION_ID`,
  `RENDERER_POST_PROCESS_INSERTION_POINT`, and
  `resolvePostProcessBackendExtension(backend)`.
- Public `backend.postProcessAdapter`, `backend.postProcessExecutor`, and
  `backend.createPostProcessGBufferBridge(context)` are removed with no public
  replacement.
- `renderer.features.enableSSAO = true` must migrate to
  `renderer.postProcess.registerPass(new ScreenSpaceAmbientOcclusionPass({
  enabled: true }))`.
- `renderer.features.enableSSGI = true` must migrate to
  `renderer.postProcess.registerPass(new ScreenSpaceGlobalIlluminationPass({
  enabled: true }))`.
- `renderer.features.enableTAA = true` must migrate to
  `renderer.postProcess.registerPass(new TemporalAntiAliasingPass({
  enabled: true }))`.
- `renderer.features.enableSSR = true` must migrate to
  `renderer.postProcess.registerPass(new ScreenSpaceReflectionsPass({
  enabled: true }))`.
- `renderer.features.enableFXAA = true` must migrate to
  `renderer.postProcess.registerPass(new FastApproximateAntiAliasingPass({
  enabled: true }))`.
- `renderer.features.enableGamma = false` must migrate to
  `renderer.postProcess.getPass("gamma")?.disable()`.
