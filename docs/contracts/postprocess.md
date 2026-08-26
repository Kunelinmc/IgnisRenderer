# Post-Process Contract

This document defines logical post-process declarations, planning, backend execution, resources, histories, and transaction ownership.

## Contract

### Cross-backend declarations

- `PostProcessPassConfig.alphaContract` may be `"opaque-only"` or
  `"premultiplied"` and must default to `"opaque-only"`.
- Transparent presentation must reject an enabled pass whose alpha contract is
  not `"premultiplied"` before graph resource allocation or command recording.
- Every built-in implementation must accept and produce premultiplied RGBA.
  Point operations must safely unpremultiply before nonlinear RGB transforms;
  spatial filters must filter coverage with the same support as RGB; effects
  that add visible radiance outside source coverage must add coverage.
- Alpha-zero post-process output must have zero RGB. Temporal color histories
  must preserve premultiplied alpha and reset transparent pixels to zero.
- `PostProcessPassConfig.schedule` must own placement, numeric order, and
  incremental metadata. Resource behavior must not appear in the schedule.
- Incremental pass registration must accept one logical `PostProcessPass` and
  derive its ID, built-in status, order, and incremental metadata from that
  pass. A separate ID-and-metadata registration form must not exist.
- `PostProcessPassConfig.label` may provide a human-readable pass name for
  diagnostics and consumer-facing metadata. It defaults to `id`.
- `PostProcessPassImplementation.describeExecution(request)` must return one
  complete `PostProcessExecutionDeclaration` for the active backend.
- Engine-owned implementations must compose explicit typed color and resource
  uses. A backend-name factory that infers access or usage must not be the
  source of an implementation declaration.
- The declaration must contain `color` and may contain `gBuffer`, `histories`,
  `transients`, backend `shared` resource entries, and `frameRequirements`.
- `frameRequirements` must use the pipeline-owned
  `FramePreparationRequirements` contract. Post-process declarations are
  requirement producers; backend frame preparation is the consumer.
- `frameRequirements.cameraJitter` must declare the pre-scene camera sampling
  sequence and normalized scale required by the implementation. Backends must
  consume the finalized aggregate instead of identifying a pass by ID.
- The planner must aggregate frame requirements only from eligible passes.
  Identical camera-jitter requirements may be coalesced; incompatible
  requirements must fail planning and identify every conflicting pass.
- History and transient entries must contain their allocation descriptor and
  all logical uses. A second descriptor API or graph-metadata overlay must not
  exist.
- Required G-buffer or shared resources must make a pass ineligible when they
  are unavailable. Optional resources must not affect eligibility.
- Implementations must receive a fixed backend execution context containing a
  `PostProcessResourceAccessor`. Backends must not synthesize pass-specific
  context properties from metadata.
- The accessor must expose assigned color input/output and typed getters for
  G-buffer, history, transient, and shared resources. Access to an undeclared
  resource must throw; a missing optional resource must return `null`.
- `LogicalGBufferBridge.normalSpace` must describe the physical normal channel
  encoding supplied by the active backend. An implementation that reconstructs
  positions or directions in another coordinate space must explicitly convert
  sampled normals before combining them with those values.
- `IPostProcessExecutor.createGBufferBridge(context, options)` must accept
  `resourceMode: "physical" | "synthetic"`; omitted mode must mean
  `"physical"`.
- A physical bridge must expose only channels backed by the active frame's
  allocated resources. A synthetic bridge must expose every logical G-buffer
  semantic with null resource handles while preserving the backend's real
  normal, depth, and motion metadata.
- Creating a synthetic bridge must not allocate CPU or GPU frame resources,
  require initialized device services, or mutate frame lifecycle state.
- Declaration planning and warmup must request a synthetic bridge through
  `createGBufferBridge()`. They must not duplicate backend metadata or infer it
  from a backend identifier.
- `color.output: "new-version"` must receive a backend-assigned output distinct
  from its input. `color.output: "preserve"` must not receive a new output.
- `{ ran: true }` must commit the assigned color output automatically.
  `{ ran: false }` must alias the planned output to its input and must not
  update history.
- A backend implementation may consume a history write resource after writing
  it in the same pass. The history write declaration must include every
  applicable use, including both storage write and sampled read.
- `PostProcessPassResult.updatedHistoryIds` must contain only declared history
  IDs with write uses. The runtime must reject updates reported with
  `{ ran: false }`.
- A missing active-backend implementation must skip the pass and emit
  `postprocess-implementation-missing-<passId>`. Backends must not dispatch a
  fallback kernel by pass ID.
- Backend runtime implementation instances must remain backend-local and must
  be invalidated or destroyed with their owning device lifecycle.
- Shared backend services may generalize backend-specific implementation
  details without changing the logical pass contract. The WebGPU denoiser is
  internal to WebGPU execution; Software and WebGL implementations must not be
  required to expose or emulate that service.

### Backend execution

#### Common Planning

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

#### Resource Binding and Lifecycle

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

#### WebGPU

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

#### WebGL

- The fixed WebGL context must expose GL, fullscreen draw helpers, diagnostics,
  and a typed resource accessor.
- G-buffer, history, transient, and shared textures must be obtained through
  the accessor, never through dynamically injected properties.
- The graph-assigned color output must be bound as the final framebuffer target
  and committed automatically when the implementation reports success.

#### Software

- The fixed Software context must expose frame attachments and a CPU resource
  accessor. It must not synthesize pass-specific canvas or shadow-sampling
  services.
- The Software executor must not switch on `passId` to construct different
  context shapes.
- The built-in volumetric lighting pass provides a WebGPU implementation only;
  Software and WebGL runtimes must skip it as a missing implementation and emit
  `postprocess-implementation-missing-volumetric`.
- In-place CPU effects must declare color `{ access: "read-write", output:
  "preserve" }`.

## Usage

### Cross-backend declarations

```ts
class CustomWebGPUImplementation {
	public describeExecution(): PostProcessExecutionDeclaration {
		return {
			color: { access: "read", output: "new-version" },
			transients: [{
				descriptor: { id: "custom:scratch", format: "rgba16float" },
				uses: [{ access: "write", usage: "storage" }],
			}],
		};
	}

	public execute(request, context): PostProcessPassResult {
		const source = context.resources.color.input;
		const target = context.resources.color.output;
		const scratch = context.resources.getTransient("custom:scratch");
		// Record commands from source through scratch into target.
		return { ran: true };
	}
}
```

### Backend execution

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

## Diagnostics

### Cross-backend declarations

- Malformed declarations must fail planning and identify the backend, pass ID,
  resource ID, and every detected violation.
- Duplicate history or transient IDs with incompatible descriptors must fail
  planning; the runtime must not select the first descriptor.
- Missing required G-buffer channels must retain the
  `postprocess-requirement-missing-<passId>` diagnostic.
- Missing required backend-shared resources must retain the
  `postprocess-backend-shared-unavailable-<passId>` diagnostic.
- Exceptions during execution must abort the active post-process transaction.
- Temporal camera state derived from frame requirements must remain tentative
  until the enclosing backend frame commits. Abort must restore the previous
  jitter sequence and view-projection state.

### Backend execution

- Undeclared accessor calls must throw and identify the pass and resource.
- `{ ran: false, updatedHistoryIds: [...] }` must abort the transaction.
- A history update not declared with a write use must abort the transaction.
- A missing required physical binding after successful planning is a backend
  invariant violation and must abort the frame.
- Runtime attempts must preserve `lastAttempt` and `lastSuccessful`; only a
  committed enclosing frame may update `lastSuccessful`.

## Verification

```bash
bun tests/static/webgpu/test_webgpu_post_graph.mjs
bun tests/static/postprocess/test_postprocess_execution_declarations.mjs
bunx tsc --noEmit
```

## Related Documents

- [Rendering architecture](../architecture/rendering.md)
- [Render Graph architecture](../architecture/render-graph.md)
- [WebGPU contract](webgpu.md)
- [WebGL contract](webgl.md)
- [Software backend contract](software.md)
