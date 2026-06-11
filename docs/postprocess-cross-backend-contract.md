# Post-Process Cross-Backend Contract
## Scope
This document defines the cross-backend post-process abstraction used by
`Renderer`, `BackendPostProcessRuntime`, logical post-process passes, and render
backends.

## Background
`Renderer` owns the public `renderer.postProcess` registry only. The logical
`postprocess` frame stage is a `backend-pass` in `FrameContext.framePlan`.
Backends that declare `BackendCapabilities.postProcess = true` must execute that
stage inside `IRenderBackend.executePass({ stage: "postprocess" })`.

`PostProcessGraphCompiler` owns deterministic logical graph compilation.
`PostProcessResourcePool` owns temporal history and transient resource lifetime.
`BackendPostProcessRuntime` composes both pieces with an `IPostProcessExecutor`
and is owned by each backend instance.

## API/Contract
- `BackendCapabilities.postProcess` must be `true` only when the backend handles
  the `"postprocess"` backend pass.
- Built-in Software, WebGL, and WebGPU backends must set
  `BackendCapabilities.postProcess = true`.
- Custom backends that set `BackendCapabilities.postProcess = false` must not
  receive an enabled `"postprocess"` backend pass from the default frame plan.
- `BUILTIN_FRAME_PASS_STAGES` must include `"postprocess"`.
- The default pipeline must define `"postprocess"` as `kind: "backend-pass"`.
- The `"postprocess"` stage must depend on `"particles"`.
- The `"sync-out"` stage must depend on `"postprocess"`.
- `Renderer` must pass `backendType` and `backendCapabilities` into frame plan
  creation.
- `Renderer` must not own post-process graph execution, resource allocation,
  history commit, or history abort.
- `Renderer` must not resolve post-process execution through backend extensions.
- `BackendPostProcessRuntime` must be the only runtime execution owner for the
  cross-backend post-process graph.
- `PostProcessGraphCompiler` must be the only source of graph ordering,
  incremental slicing, eligibility, and resource descriptor compilation.
- `renderer.postProcess.registerPass(pass)` must remain the public registration
  API for logical passes.
- `PostProcessPass.placement` may be `"spatial"`, `"temporal"`,
  `"atmosphere"`, `"camera"`, `"hdr"`, `"ldr"`, `"overlay"`, or `"present"`.
- `PostProcessPass.order` may refine ordering inside one placement bucket.
- `PostProcessPass.shouldExecute(request)` must be deterministic for the
  supplied frame and must not allocate backend resources.
- `PostProcessGraphCompiler` must apply `shouldExecute(request)` before sorting.
- `PostProcessGraphCompiler` must apply incremental slicing from
  `IncrementalFrameContext.postProcessStartPass` when the frame starts at
  `"postprocess"`.
- `PostProcessGraphCompiler` must filter passes whose required
  `LogicalGBufferSemantic` channels are unavailable.
- `PostProcessGraphCompiler` must collect history and transient descriptors only
  from eligible passes.
- `PostProcessGraphCompiler` must keep the first descriptor when two eligible
  passes request incompatible descriptors for the same history or transient id.
- `PostProcessResourcePool` must recreate temporal resources only when
  dimensions, format, usage, backend kind, reset state, or graph signature
  requires it.
- `PostProcessResourcePool` must recreate transient resources only when
  dimensions, format, usage, mip mode, or backend kind changes.
- `PostProcessResourcePool.commitFrame()` must swap only histories marked as
  updated by the executed frame.
- `PostProcessResourcePool.abortFrame()` must clear pending updates without
  invalidating previously valid histories.
- `BackendPostProcessRuntime.execute(context)` must compile the graph, prepare
  resources, call executor frame hooks, execute eligible passes, and record
  updated histories.
- `BackendPostProcessRuntime.execute(context)` must not commit histories.
- Backend `endFrame()` must call `BackendPostProcessRuntime.commitFrame()` only
  after backend frame execution succeeds.
- Backend `abortFrame(error)` must call `BackendPostProcessRuntime.abortFrame()`
  before clearing other backend frame state.
- Backend resize, device loss, shader runtime change, and destroy paths must call
  `BackendPostProcessRuntime.invalidateFrameSized()` or `destroy()` directly.
- Backends must not emit `RendererBackendBridge.onBackendResourceEvent()` for
  post-process resources.
- `IPostProcessExecutor.createGBufferBridge(context)` must create the logical
  G-buffer view consumed by post-process passes during execution.
- `IPostProcessExecutor.createResource(desc)` and
  `IPostProcessExecutor.destroyResource(handle)` must own concrete backend
  resources for histories and transients.
- `IPostProcessExecutor.invalidateResourceBindings()` may invalidate backend
  binding caches after transients are recreated.
- `IPostProcessExecutor.getPassExecutionContext(request)` may return
  backend-specific helpers for pass-owned implementations.
- `IPostProcessExecutor.executePass(passId, request)` must execute fallback
  backend-owned logical passes when no pass-owned implementation exists.
- `IPostProcessExecutor.completePass(request, result)` may publish validated
  backend-owned side effects for one pass.
- `IPostProcessExecutor.abortFrame(request)` must be idempotent and must not
  commit histories.
- Backend warmup must use `BackendPostProcessRuntime.compileWarmupGraph(context)`
  to obtain ordered pass descriptors and implementation warmup hints.
- Warmup code must not depend on `WARMUP_POST_PROCESS_*` transient keys.
- Backend support for engine-provided logical passes must be derived from
  pass-owned implementations, not from a public backend pass registry.
- Backends must not expose public post-process graph registration APIs.

## Usage
```ts
import { PostProcessPass, Renderer, WebGPUBackend } from "ignisrenderer";

class CustomOverlayPass extends PostProcessPass {
	public constructor() {
		super({
			id: "custom-overlay",
			placement: "overlay",
			enabled: true,
			implementations: {
				webgpu: { id: "custom-overlay:webgpu" },
				webgl: { id: "custom-overlay:webgl" },
				software: { id: "custom-overlay:software" },
			},
		});
	}
}

const backend = new WebGPUBackend();
const renderer = new Renderer(backend, canvas, camera);
renderer.postProcess.registerPass(new CustomOverlayPass());
await renderer.renderScene(0);
```

```bash
bun tests/static/postprocess/test_postprocess_graph_compiler.mjs
bun tests/static/postprocess/test_postprocess_resource_pool.mjs
bun tests/static/postprocess/test_postprocess_public_api.mjs
```

## Errors & Diagnostics
- `postprocess-requirement-missing-<passId>` must be emitted when required
  logical G-buffer channels are unavailable during execution graph compilation.
- `postprocess-history-conflict-<historyId>` must be emitted when eligible
  passes request incompatible descriptors for the same history id.
- `postprocess-transient-conflict-<transientId>` must be emitted when eligible
  passes request incompatible descriptors for the same transient id.
- `renderer.postProcess.registerPass(pass)` must throw when `pass` is not a
  `PostProcessPass`.
- `renderer.postProcess.registerPass(pass)` must throw when `pass.id` is already
  registered.

## Compatibility / Breaking Changes
- The `renderer.postprocess` backend extension is removed.
- `RENDERER_POST_PROCESS_EXTENSION_ID`,
  `RENDERER_POST_PROCESS_INSERTION_POINT`, and
  `resolvePostProcessBackendExtension(backend)` are removed.
- `RendererPostProcessController` is removed.
- `Renderer` no longer owns post-process history commit, abort, invalidation, or
  destruction.
- Custom backends that support post-processing must set
  `BackendCapabilities.postProcess = true` and handle
  `executePass({ stage: "postprocess" })`.
- Custom backends that do not support post-processing must set
  `BackendCapabilities.postProcess = false`; no adapter-missing warning is
  emitted by `Renderer`.
- Public `backend.postProcessAdapter`, `backend.postProcessExecutor`,
  `backend.postProcess`, and `backend.createPostProcessGBufferBridge(context)`
  remain unsupported.
- Code that previously depended on per-pass frame stages such as `ssao`, `taa`,
  or `gamma` must use the single `"postprocess"` backend pass and inspect
  `IncrementalFrameContext.postProcessStartPass` for the internal logical start.
- `PostProcessPipeline`, `PostProcessPipelineExecuteRequest`, and
  `PostProcessPipelineExecuteResult` are removed. Tests, tools, and custom
  backends must use `PostProcessGraphCompiler`, `PostProcessResourcePool`, or
  `BackendPostProcessRuntime` according to whether they need graph metadata,
  resource lifetime, or full backend execution.
