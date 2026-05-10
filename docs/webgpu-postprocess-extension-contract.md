# WebGPU Post-Process Extension Contract

## Scope

This document defines the public WebGPU post-process extension contract for
custom graph passes and optional runtime pass handlers.

## Background

`WebGPUBackend.registerPostProcessPass(pass)` registers a pass in the WebGPU
post-process graph. A pass may now include `runtime` metadata when it needs the
shared WebGPU post-process runtime to own shader, pipeline, buffer, bind group,
or warmup lifecycle work.

## API/Contract

- `WebGPUPostProcessPassPlugin.id` must be a non-empty custom id.
- `WebGPUPostProcessPassPlugin.runtime` may define a
  `WebGPUPostProcessRuntimePass`.
- `WebGPUPostProcessRuntimePass.id` must be a non-empty custom id and should
  match the graph pass id unless one graph pass intentionally dispatches a
  separately named runtime handler.
- `WebGPUPostProcessPassContext.executeRuntimePass(request)` must dispatch to
  the registered runtime handler whose `id` matches `request.passId`.
- `WebGPUPostProcessRuntimePass.warmupHints` lists runtime hints that may be
  precompiled during WebGPU warmup.
- `WebGPUPostProcessRuntimePass.warmup(hint, context)` may create shaders,
  pipelines, buffers, or other reusable resources. Returning `false` means the
  hint did not compile work; returning `true` or `void` means it did.
- `WebGPUPostProcessRuntimePass.execute(request, context)` must encode the
  runtime work for `request.passId`. Returning `void` is treated as `{ ran:
  true }`.
- `WebGPUPostProcessRuntimePass.invalidateBindings(context)` must release
  cached texture-dependent bindings when frame targets are rebuilt.
- `WebGPUPostProcessRuntimePass.onShaderRuntimeChanged(context)` must release
  shader-runtime-dependent resources.
- The ids `ssao`, `ssgi`, `taa`, `ssr`, `volumetric`, `fog`, `motion-blur`,
  `dof`, `bloom`, `color-filter`, `fxaa`, `interaction-outline`, `tonemap`,
  and `gamma` are reserved and must not be registered or unregistered by custom
  code.

## Usage

```ts
import type {
	WebGPUPostProcessPassPlugin,
	WebGPUPostProcessRuntimePass,
} from "ignisrenderer";
import { WebGPUBackend } from "ignisrenderer";

const runtime: WebGPUPostProcessRuntimePass = {
	id: "custom-edge",
	warmupHints: ["postprocess:custom-edge"],
	async warmup(_hint, context) {
		await context.ensureCommonResources();
	},
	async execute(request, context) {
		await context.ensureCommonResources();
		request.encoder.beginComputePass({ label: "CustomEdge" });
		request.encoder.endComputePass();
		return { ran: true };
	},
};

const pass: WebGPUPostProcessPassPlugin = {
	id: "custom-edge",
	kind: "compute",
	dependsOn: ["tonemap"],
	runtime,
	isEnabled: () => true,
	async execute(context) {
		await context.executeRuntimePass({
			passId: "custom-edge",
			encoder: context.encoder,
			targets: context.targets,
			frameContext: context.frameContext,
		});
	},
};

const backend = new WebGPUBackend();
backend.registerPostProcessPass(pass);
```

## Errors & Diagnostics

- Registering an empty id must throw `Post-process pass id is required.` or
  `WebGPU post-process runtime pass id is required.`
- Registering a reserved graph id must throw a built-in pass error.
- Registering a reserved runtime id must throw a built-in runtime pass error.
- Registering a duplicate custom graph id must throw an already registered
  graph pass error.
- Registering a duplicate custom runtime id must throw an already registered
  runtime pass error.
- Unregistering a reserved id must throw a built-in unregister error.

## Compatibility / Breaking Changes

Custom integrations must not replace built-in WebGPU post-process passes by
registering the same id. Code that previously replaced `tonemap`, `fxaa`,
`gamma`, or another built-in id must register a new custom id and position it
with `dependsOn`.
