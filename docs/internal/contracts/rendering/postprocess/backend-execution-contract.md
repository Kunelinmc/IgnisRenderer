# Post-Process Backend Execution Contract

## Scope

This document defines planning, subgraph composition, resource binding, and
execution rules for Software, WebGL, and WebGPU post-processing.

## Background

GPU post-processing is a backend-owned `"postprocess"` stage composed into the
authoritative whole-frame Render Graph. Software uses the same logical plan but
executes it directly. A post-process runtime must not compile a nested GPU
Render Graph.

## API/Contract

### Common Planning

- `PostProcessPlanner` must resolve enabled passes, deterministic schedule
  order, incremental start, active-backend implementation, execution
  declaration, eligibility, and resource descriptors exactly once per frame.
- GPU backends that allocate frame targets from post-process requirements must
  retain a declaration plan before allocation, then finalize resource
  availability from that retained plan after allocation.
- Availability finalization must reuse the retained execution declarations. It
  must not resolve pass order, instantiate implementations, or call
  `describeExecution()` again.
- Finalization must aggregate pre-scene frame requirements from eligible
  passes. Scene preparation must consume that retained aggregate and must not
  query pass IDs or pass-specific option types.
- An implementation's `describeExecution()` must be called at most once for
  one planned frame. The immutable returned declaration must be retained by the
  plan and reused by subgraph building and execution.
- Missing active-backend implementations must be skipped with a stable warning.
- Required G-buffer and shared-resource availability must be evaluated from the
  declaration. Optional unavailable entries must remain accessible as `null`.
- History and transient descriptor conflicts must fail planning.
- `PostProcessSubgraphBuilder` must expose named imports and a named `color`
  export. GPU backends must compose it under `"postprocess"` and flatten each
  retained pass to one always-retained `"post-process-pass"` outer node.
- The whole-frame compiler must be the only shared GPU compiler invocation.
- Software must consume `PostProcessPlan` directly without constructing or
  compiling a GPU Render Graph.

### Resource Binding and Lifecycle

- The runtime must prepare only descriptors retained in the current plan.
- Backend managers and post-process pools must retain allocation, aliasing,
  native-handle, resize, device-loss, and destruction ownership.
- Before execution, the backend binding must create one fixed execution context
  with a declaration-checked resource accessor.
- For `"new-version"`, `color.input` and `color.output` must be different
  physical resources. For `"preserve"`, `color.output` must be `null`.
- A successful result must advance the backend color target automatically.
  A skipped result must retain the previous physical target and record the
  planned-to-resolved logical alias.
- History writes must remain pending until the enclosing backend frame commits.
  Abort, device loss, execution failure, and `{ ran: false }` must not commit
  them.
- Camera jitter and previous transform state must use the same enclosing frame
  transaction boundary. A failed or aborted frame must not advance either
  state.
- Software, WebGL, and WebGPU temporal camera services must delegate that
  transaction to `TemporalFrameState`. Backend binding caches must not
  duplicate jitter checkpoints or pending previous-view-projection ownership.
- Warmup must use the same planner and declaration validation with synthetic
  availability. It must not allocate frame resources.

### WebGPU

- The fixed WebGPU context must expose the command encoder, frame services,
  shared post-process services, and a typed resource accessor.
- `WebGPUPostProcessRuntime` must directly own common sampler, bind-group cache,
  Hi-Z service access, and one lazy `WebGPUDenoiser`. A second shared-context
  lifecycle owner must not exist.
- `WebGPUDenoiser` must own its shader modules, compute pipelines, uniform
  buffers, and binding cache. It must invalidate texture bindings on resize,
  invalidate shader-owned resources on shader runtime changes, and release all
  owned resources with `WebGPUPostProcessRuntime`.
- `WebGPUDenoiser` must not allocate graph textures, submit command buffers, or
  advance the color target. A consuming pass must declare every denoise source,
  scratch, and output use and pass the graph-owned textures to the service.
- A consuming pass may denoise back into its source only when it provides a
  distinct scratch texture and declares the source for both storage write and
  sampled read.
- Pass implementations should write final temporal results directly to their
  declared history write resource when that resource is also consumed later in
  the same pass. They must declare every storage-write and sampled-read use and
  must not introduce a shared copy service for pass-local transfers.
- Hi-Z must be declared as required shared resource
  `"backend:frame-hiz"` and obtained through `getShared()`.
- Frame bindings, lighting state, and feature data are backend services, not
  graph-resource metadata.
- Built-in shared-resource IDs must be resolved through one WebGPU catalog that
  defines graph bindings, allocation groups, availability, and execution-time
  texture access. Unknown custom IDs must remain unavailable.
- WebGPU frame-target requirements must be derived from retained execution
  declarations. Feature analysis must not infer consumers from built-in pass
  IDs. An optional planar-reflection-mask declaration must still request mask
  allocation so the pass may consume it when available.
- Motion history copies must use
  `copyGBufferToHistory("motion", historyId)` and must validate both the source
  read and history write declarations.

### WebGL

- The fixed WebGL context must expose GL, fullscreen draw helpers, diagnostics,
  and a typed resource accessor.
- G-buffer, history, transient, and shared textures must be obtained through
  the accessor, never through dynamically injected properties.
- The graph-assigned color output must be bound as the final framebuffer target
  and committed automatically when the implementation reports success.

### Software

- The fixed Software context must expose canvas, attachment, shadow-sampling
  services, and a CPU resource accessor.
- The Software executor must not switch on `passId` to construct different
  context shapes.
- In-place CPU effects must declare color `{ access: "read-write", output:
  "preserve" }`.

## Usage

```ts
class CustomPass extends PostProcessPass {
	public constructor() {
		super({
			id: "custom",
			schedule: { placement: "ldr", order: 5 },
			enabled: true,
			implementations: {
				webgpu: () => new CustomWebGPUImplementation(),
			},
		});
	}
}
```

## Errors & Diagnostics

- Undeclared accessor calls must throw and identify the pass and resource.
- `{ ran: false, updatedHistoryIds: [...] }` must abort the transaction.
- A history update not declared with a write use must abort the transaction.
- A missing required physical binding after successful planning is a backend
  invariant violation and must abort the frame.
- Runtime attempts must preserve `lastAttempt` and `lastSuccessful`; only a
  committed enclosing frame may update `lastSuccessful`.

## Compatibility / Breaking Changes

- `PostProcessSharedContext` and the `sharedContext` runtime indirection are
  removed. Built-in WebGPU contexts receive `WebGPUPostProcessRuntime` through
  the narrow `WebGPUPostProcessServices` contract.
- Nested `PostProcessGraphCompiler` execution and
  `PostProcessRenderGraphAdapter` are replaced by planning plus subgraph
  building.
- `IPostProcessExecutor.executePass(passId, request)` and pass-ID fallback
  execution are removed.
- Manual `publishColorTarget()` and `publishColorTexture()` callbacks are
  removed; assigned color output is committed from the pass result.
- Temporary and history resources migrate from separate descriptor and context
  metadata APIs into `describeExecution()`.
