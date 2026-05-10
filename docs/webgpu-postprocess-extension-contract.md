# WebGPU Post-Process Extension Contract
## Scope
This document defines the public WebGPU post-process extension contract for custom graph passes and optional runtime pass handlers.

## Background
`renderer.postProcess.registerPass(pass)` registers a custom pass through the active WebGPU backend and makes the custom id available to `renderer.postProcess.enable(id, options)`. `WebGPUBackend.postProcess.registerPass(pass)` remains the lower-level backend entry point. A pass may include `runtime` metadata when it needs the shared WebGPU post-process runtime to own shader, pipeline, buffer, bind group, or warmup lifecycle work.

## API/Contract
- `renderer.postProcess.registerPass(pass)` must register a custom WebGPU post-process graph pass when the renderer uses `WebGPUBackend`.
- `renderer.postProcess.enable(id, options)` must enable a registered custom pass id and expose custom options through `ResolvedPostProcessState.options[id]`.
- `renderer.postProcess.unregisterPass(id)` must unregister the custom pass from the active backend and remove stored request state for that id.
- `WebGPUPostProcessPassPlugin.incremental` may define `firstPass`, `grade`, `inflationRadius`, and `fallbackScale` for incremental rendering.
- If `WebGPUPostProcessPassPlugin.incremental` is omitted, incremental rendering must treat the custom pass as `firstPass: "gamma"`, `grade: "light"`, and `inflationRadius: 2`.
- `WebGPUBackend.postProcess.registerPass(pass)` must register a custom post-process graph pass.
- `WebGPUBackend.postProcess.unregisterPass(id)` must unregister a custom post-process graph pass.
- `WebGPUPostProcessPassPlugin.id` must be a non-empty custom id.
- `WebGPUPostProcessPassPlugin.isEnabled(postProcess)` must receive the current `ResolvedPostProcessState`.
- `WebGPUPostProcessPassPlugin.execute(context)` must receive `context.frameContext` and `context.postProcess`.
- `WebGPUPostProcessPassPlugin.runtime` may define a `WebGPUPostProcessRuntimePass`.
- `WebGPUPostProcessRuntimePass.id` must be a non-empty custom id and should match the graph pass id unless one graph pass intentionally dispatches a separately named runtime handler.
- `WebGPUPostProcessPassContext.executeRuntimePass(request)` must dispatch to the registered runtime handler whose `id` matches `request.passId`.
- `WebGPUPostProcessRuntimePass.warmupHints` lists runtime hints that may be precompiled during WebGPU warmup.
- `WebGPUPostProcessRuntimePass.warmup(hint, context)` may create shaders, pipelines, buffers, or other reusable resources.
- `WebGPUPostProcessRuntimePass.execute(request, context)` must encode the runtime work for `request.passId`. Returning `void` is treated as `{ ran: true }`.
- `WebGPUPostProcessRuntimePass.invalidateBindings(context)` must release cached texture-dependent bindings when frame targets are rebuilt.
- `WebGPUPostProcessRuntimePass.onShaderRuntimeChanged(context)` must release shader-runtime-dependent resources.
- The ids `ssao`, `ssgi`, `taa`, `ssr`, `volumetric`, `fog`, `motion-blur`, `dof`, `bloom`, `tonemap`, `color-filter`, `fxaa`, `interaction-outline`, and `gamma` are reserved and must not be registered or unregistered by custom code.

## Usage
```ts
import type {
	WebGPUPostProcessPassPlugin,
	WebGPUPostProcessRuntimePass,
} from "ignisrenderer";
import { Camera, Renderer, WebGPUBackend } from "ignisrenderer";

const runtime: WebGPUPostProcessRuntimePass = {
	id: "custom-edge",
	warmupHints: ["postprocess:custom-edge"],
	async warmup(_hint, context) {
		await context.ensureCommonResources();
	},
	async execute(request, context) {
		await context.ensureCommonResources();
		const pass = request.encoder.beginComputePass({ label: "CustomEdge" });
		pass.end();
		return { ran: true };
	},
};

const pass: WebGPUPostProcessPassPlugin = {
	id: "custom-edge",
	kind: "compute",
	dependsOn: ["tonemap"],
	incremental: {
		firstPass: "tonemap",
		grade: "light",
		inflationRadius: 2,
	},
	runtime,
	isEnabled(postProcess) {
		return postProcess.enabled["custom-edge"];
	},
	async execute(context) {
		await context.executeRuntimePass({
			passId: "custom-edge",
			encoder: context.encoder,
			targets: context.targets,
			frameContext: context.frameContext,
		});
	},
};

const canvas = document.querySelector("canvas") as HTMLCanvasElement;
const camera = new Camera();
const backend = new WebGPUBackend();
const renderer = new Renderer(backend, canvas, camera);
renderer.postProcess.registerPass(pass).enable("custom-edge");
```

## Errors & Diagnostics
- Calling `renderer.postProcess.enable(id)` before `renderer.postProcess.registerPass(pass)` for a custom id must throw `Unknown post-process pass "<id>".`.
- Calling `renderer.postProcess.registerPass(pass)` on a renderer whose backend does not expose a post-process pass registry must throw a backend registry availability error.
- Registering an empty graph id must throw `Post-process pass id is required.`.
- Registering an empty runtime id must throw `WebGPU post-process runtime pass id is required.`.
- Registering a reserved graph id must throw a built-in pass error.
- Registering a reserved runtime id must throw a built-in runtime pass error.
- Registering a duplicate custom graph id must throw an already registered graph pass error.
- Registering a duplicate custom runtime id must throw an already registered runtime pass error.
- Unregistering a reserved id must throw a built-in unregister error.

## Compatibility / Breaking Changes
`WebGPUBackend.registerPostProcessPass(pass)` and `WebGPUBackend.unregisterPostProcessPass(id)` are removed. Code must use `backend.postProcess.registerPass(pass)` and `backend.postProcess.unregisterPass(id)`.
