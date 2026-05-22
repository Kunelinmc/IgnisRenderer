# Post-Process Public API Migration Contract
## Scope
This document defines the breaking migration from backend-specific post-process plugin APIs to the cross-backend logical post-process API.

## Background
Post-process pass registration previously flowed through backend-owned WebGPU and WebGL graph registries. The renderer now owns a single public `postprocess` frame stage, and `PostProcessPipeline` owns logical pass ordering, G-buffer requirements, history handles, and history validity. Backends expose only `RenderBackendPostProcessSupport.executor` and `RenderBackendPostProcessSupport.createGBufferBridge(context)`.

## API/Contract
- `Renderer.features` must contain only non-post-process feature requests.
- `RendererFeatureRequest` must not expose post-process enable flags or pass option fields.
- `renderer.postProcess.enable(id, options)` must enable a built-in or registered logical post-process pass and merge pass options.
- `renderer.postProcess.disable(id)` must disable a built-in or registered logical post-process pass.
- `renderer.postProcess.setOptions(id, options)` must merge pass options without enabling the pass.
- `renderer.postProcess.reset(id)` must reset one pass to default request state.
- `renderer.postProcess.reset()` must reset all post-process request state.
- `renderer.postProcess.getState()` must return a cloned `PostProcessRequest`.
- `renderer.postProcess.registerPass(pass)` must register a `PostProcessPassDescriptor`.
- `renderer.postProcess.unregisterPass(id)` must remove the logical descriptor and any stored request state for `id`.
- A custom `PostProcessPassDescriptor` must declare backend implementations through `implementations`.
- A custom pass without an implementation for the active `IPostProcessExecutor.backend` must be treated as unsupported for that backend.
- `RenderBackendPostProcessSupport` must expose only `capabilities`, `executor`, and `createGBufferBridge(context)`.
- `RenderBackendPostProcessSupport` must not expose `registerPass` or `unregisterPass`.
- `FrameContext.postProcess` must contain the resolved `ResolvedPostProcessState` for the current frame.
- `BackendCapabilities` must not expose post-process capability fields.
- `backend.postProcess.capabilities` must expose `PostProcessCapabilities`.
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
import type { PostProcessPassDescriptor } from "ignisrenderer";

const customEdgePass: PostProcessPassDescriptor = {
	id: "custom-edge",
	dependsOn: ["tonemap"],
	incremental: {
		firstPass: "tonemap",
		grade: "light",
		inflationRadius: 2,
	},
	isEnabled(state) {
		return state.enabled["custom-edge"] === true;
	},
	implementations: {
		webgpu: { id: "custom-edge" },
		webgl: { id: "custom-edge" },
		software: { id: "custom-edge" },
	},
};

renderer.postProcess.registerPass(customEdgePass);
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
- `renderer.postProcess.registerPass(pass)` must throw when `pass.id` is empty.
- `renderer.postProcess.registerPass(pass)` must throw when `pass.id` is a built-in pass id.
- `renderer.postProcess.registerPass(pass)` must throw when `pass.id` is already registered.
- A backend that omits `postProcess.executor` or `postProcess.createGBufferBridge(context)` violates `IRenderBackend` and must not be used with `Renderer`.
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
- `backend.postProcess.registerPass(pass)` is removed.
- `backend.postProcess.unregisterPass(id)` is removed.
- `WebGPUPostProcessPassPlugin` and `WebGLPostProcessPassPlugin` are removed from the public API.
