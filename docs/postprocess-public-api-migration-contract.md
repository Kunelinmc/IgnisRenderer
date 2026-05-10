# Post-Process Public API Migration Contract
## Scope
This document defines the breaking migration from post-process flags on `Renderer.features` and `BackendCapabilities` to the dedicated public post-process API.

## Background
Post-process controls were previously mixed with backend-agnostic renderer feature flags. The renderer now separates core rendering features from image-space post-process passes through `renderer.postProcess`, `FrameContext.postProcess`, and `backend.postProcess.capabilities`.

## API/Contract
- `Renderer.features` must contain only non-post-process feature requests.
- `RendererFeatureRequest` must not expose `enableSSAO`, `enableSSGI`, `enableTAA`, `enableSSR`, `enableVolumetric`, `enableFog`, `enableMotionBlur`, `enableDOF`, `enableBloom`, `enableToneMapping`, `enableColorFilter`, `enableFXAA`, or `enableGamma`.
- `RendererFeatureRequest` must not expose post-process option fields such as `ssaoOptions`, `ssgiOptions`, `taaOptions`, `ssrOptions`, `volumetricOptions`, `fogOptions`, `motionBlurOptions`, `dofOptions`, `bloomOptions`, or `colorFilterOptions`.
- `renderer.postProcess.enable(id, options)` must enable a post-process pass and merge pass options.
- `renderer.postProcess.disable(id)` must disable a post-process pass.
- `renderer.postProcess.setOptions(id, options)` must merge pass options without enabling the pass.
- `renderer.postProcess.reset(id)` must reset one pass to default request state.
- `renderer.postProcess.reset()` must reset all post-process request state.
- `renderer.postProcess.getState()` must return a cloned `PostProcessRequest`.
- `renderer.postProcess.registerPass(pass)` must register a custom backend post-process pass through the active backend post-process registry.
- `renderer.postProcess.unregisterPass(id)` must unregister a custom backend post-process pass through the active backend post-process registry.
- Registered custom pass ids must be accepted by `renderer.postProcess.enable(id, options)`, `renderer.postProcess.disable(id)`, `renderer.postProcess.setOptions(id, options)`, and `renderer.postProcess.reset(id)`.
- `FrameContext.postProcess` must contain the resolved `ResolvedPostProcessState` for the current frame.
- `BackendCapabilities` must not expose post-process capability fields.
- `backend.postProcess.capabilities` must expose `PostProcessCapabilities`.
- `WebGPUBackend.postProcess.registerPass(pass)` and `WebGLBackend.postProcess.registerPass(pass)` must replace the removed top-level backend registration APIs.
- Default enabled passes must be `tonemap`, `gamma`, and `interaction-outline`.
- Unsupported explicit enables must emit warning key `"<backend>-postprocess-unsupported-<passId>"`.

## Usage
```ts
renderer.postProcess.enable("ssao", {
	radius: 6,
	intensity: 1.2,
});
renderer.postProcess.setOptions("fog", {
	application: "scene",
	density: 0.02,
});
renderer.postProcess.enable("fog");
renderer.postProcess.disable("gamma");
```

```ts
renderer.postProcess.registerPass({
	id: "custom-edge",
	dependsOn: ["tonemap"],
	isEnabled(postProcess) {
		return postProcess.enabled["custom-edge"];
	},
	execute(context) {
		context.executeRuntimePass({
			passId: "custom-edge",
			encoder: context.encoder,
			targets: context.targets,
			frameContext: context.frameContext,
		});
	},
});
renderer.postProcess.enable("custom-edge", {
	strength: 0.75,
});
```

```ts
const supportsSSR = renderer.backend.postProcess.capabilities.ssr;
if (supportsSSR) {
	renderer.postProcess.enable("ssr");
}
```

```bash
bun tests/test_postprocess_public_api.mjs
```

## Errors & Diagnostics
- `"<backend>-postprocess-unsupported-<passId>"` must be emitted when `renderer.postProcess.enable(passId)` requests a pass unsupported by `backend.postProcess.capabilities`.
- `renderer.postProcess.enable(id)` must throw `Unknown post-process pass "<id>".` when `id` is neither a built-in pass id nor a registered custom pass id.
- `renderer.postProcess.registerPass(pass)` must throw when the active backend does not expose a post-process pass registry.
- A custom backend that omits `postProcess.capabilities` must fail during render setup because post-process resolution requires backend support metadata.
- Invalid option values must be handled by the pass implementation according to that pass contract.

## Compatibility / Breaking Changes
- `renderer.features.enableSSAO = true` must migrate to `renderer.postProcess.enable("ssao")`.
- `renderer.features.enableSSGI = true` must migrate to `renderer.postProcess.enable("ssgi")`.
- `renderer.features.enableTAA = true` must migrate to `renderer.postProcess.enable("taa")`.
- `renderer.features.enableSSR = true` must migrate to `renderer.postProcess.enable("ssr")`.
- `renderer.features.enableVolumetric = true` must migrate to `renderer.postProcess.enable("volumetric")`.
- `renderer.features.enableFog = true` must migrate to `renderer.postProcess.enable("fog")`.
- `renderer.features.enableMotionBlur = true` must migrate to `renderer.postProcess.enable("motion-blur")`.
- `renderer.features.enableDOF = true` must migrate to `renderer.postProcess.enable("dof")`.
- `renderer.features.enableBloom = true` must migrate to `renderer.postProcess.enable("bloom")`.
- `renderer.features.enableToneMapping = false` must migrate to `renderer.postProcess.disable("tonemap")`.
- `renderer.features.enableColorFilter = true` must migrate to `renderer.postProcess.enable("color-filter")`.
- `renderer.features.enableFXAA = true` must migrate to `renderer.postProcess.enable("fxaa")`.
- `renderer.features.enableGamma = false` must migrate to `renderer.postProcess.disable("gamma")`.
- `renderer.features.<pass>Options = options` must migrate to `renderer.postProcess.setOptions("<pass>", options)` or `renderer.postProcess.enable("<pass>", options)`.
- `backend.capabilities.<postProcessField>` must migrate to `backend.postProcess.capabilities.<postProcessField>`.
- `backend.registerPostProcessPass(pass)` must migrate to `backend.postProcess.registerPass(pass)`.
- `backend.unregisterPostProcessPass(id)` must migrate to `backend.postProcess.unregisterPass(id)`.
