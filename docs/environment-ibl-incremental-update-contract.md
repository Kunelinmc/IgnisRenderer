# Environment IBL Incremental Update Contract
## Scope
This document defines the runtime contract for incremental environment IBL updates in `Renderer`.
The contract covers update triggering, per-frame incremental application, temporal blending, and dirty-reason behavior.

## Background
Before this contract, environment IBL baking was primarily driven by `warmup()` and produced one-shot updates.
Runtime scene changes to the environment did not provide a dedicated incremental environment IBL update pipeline.

## API/Contract
- `Renderer.pipeline` must expose the renderer `RenderPipelineRegistry`.
- `renderer.pipeline.registerDirtyReason(descriptor)` must register a custom dirty reason in the default incremental registry and must return the allocated mask.
- `renderer.pipeline.unregisterDirtyReason(id)` must unregister a custom dirty reason and must not reuse its allocated mask.
- `renderer.pipeline.registerPipelineStage(stage)` must register a renderer, backend-pass, or shared-pass stage descriptor.
- Custom backend or shared pass stages must set `kind` to `"backend-pass"` or `"shared-pass"` and must include incremental pass ordering metadata when they can be selected as an incremental first pass.
- `renderer.pipeline.unregisterPipelineStage(id)` must unregister a custom pipeline stage and must remove corresponding incremental pass metadata for backend or shared pass stages.
- `Renderer.setEnvironmentIBLUpdateOptions(options)` must accept partial `EnvironmentIBLUpdateOptions` and must normalize values before storing them.
- `Renderer.getEnvironmentIBLUpdateOptions()` must return a normalized snapshot.
- `Renderer.requestEnvironmentIBLUpdate()` must enqueue a manual update request token and must mark the frame dirty.
- `EnvironmentIBLUpdateOptions` must include:
	- `enabled`
	- `autoUpdate`
	- `mipsPerFrame`
	- `temporalBlendFactor`
	- `temporalBlendEpsilon`
	- `acceleration`
	- `prefilterMaxSampleWidth`
	- `prefilterMaxSampleHeight`
	- `prefilterMaxMipLevels`
	- `resetTemporalHistoryOnComplete`
- Runtime behavior requirements:
	- When `enabled` is `false`, the runtime updater must not schedule or apply new environment IBL work.
	- When `autoUpdate` is `true`, the updater must detect environment signature changes and schedule a new bake.
	- Manual requests via `requestEnvironmentIBLUpdate()` must schedule a new bake when a valid environment exists.
	- The updater must apply at most `mipsPerFrame` reflection-specular mip levels per frame.
	- Temporal blending must be applied to both `LightProbe.sh` and environment `ReflectionProbe.prefilteredMap`.
	- Environment IBL update must target:
		- only `LightProbe` instances with `source === "environment"` for SH coefficients
		- only `ReflectionProbe` instances with `source === "environment"` for specular maps
	- Runtime must emit dirty reason `environment-ibl` while update is in progress.
	- Runtime must emit dirty reason `environment-ibl-complete` when one update round converges and finishes.
- Incremental planner requirements:
	- `environment-ibl` must force full-frame rendering and must not reset temporal history.
	- `environment-ibl-complete` must force full-frame rendering and must reset temporal history.
	- Built-in dirty reason behavior must be registered through the default incremental registry so custom dirty reasons can coexist with environment IBL reasons.

## Usage
```ts
renderer.setEnvironmentIBLUpdateOptions({
	enabled: true,
	autoUpdate: true,
	mipsPerFrame: 1,
	temporalBlendFactor: 0.2,
	temporalBlendEpsilon: 1e-3,
	acceleration: "auto",
	resetTemporalHistoryOnComplete: true,
});

renderer.requestEnvironmentIBLUpdate();
```

```ts
const mask = renderer.pipeline.registerDirtyReason({
	id: "custom-post-update",
	firstPass: "gamma",
});
renderer.requestRender("custom-post-update");
```

```ts
renderer.pipeline.registerPipelineStage({
	id: "custom-plan-pass",
	kind: "backend-pass",
	dependsOn: ["main-opaque"],
	shouldRun: () => true,
	incremental: { order: 4.5 },
});
```

```bash
bun tests/test_environment_ibl_update_runtime.mjs
```

## Errors & Diagnostics
- `environment-ibl-update-bake-failed` must be emitted as a warning when runtime bake fails.
- When a bake task is aborted due to a newer request, the aborted task result must not be applied.
- When environment texture is unavailable or invalid, runtime update must skip scheduling and must not mutate probe data.

## Compatibility / Breaking Changes
`renderer.pipeline.registerBackendPass(pass)` and `renderer.pipeline.unregisterBackendPass(id)` are removed.
Consumers must use `renderer.pipeline.registerPipelineStage(stage)` and `renderer.pipeline.unregisterPipelineStage(id)`.
The runtime environment IBL update path remains opt-in via `EnvironmentIBLUpdateOptions.enabled`.
