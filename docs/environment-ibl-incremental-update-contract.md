# Environment IBL Incremental Update Contract
## Scope
This document defines the runtime contract for incremental environment IBL updates in `Renderer`.
The contract covers update triggering, per-frame incremental application, temporal blending, and dirty-reason behavior.

## Background
Before this contract, environment IBL baking was primarily driven by `warmup()` and produced one-shot updates.
Runtime scene changes to the skybox did not provide a dedicated incremental environment IBL update pipeline.

## API/Contract
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
	- When `autoUpdate` is `true`, the updater must detect skybox signature changes and schedule a new bake.
	- Manual requests via `requestEnvironmentIBLUpdate()` must schedule a new bake when a valid skybox exists.
	- The updater must apply at most `mipsPerFrame` reflection-specular mip levels per frame.
	- Temporal blending must be applied to both `LightProbe.sh` and skybox `ReflectionProbe.prefilteredMap`.
	- Skybox IBL update must target:
		- all `LightType.LightProbe` instances for SH coefficients
		- only `ReflectionProbe` instances with `source === "skybox"` for specular maps
	- Runtime must emit dirty reason `environment-ibl` while update is in progress.
	- Runtime must emit dirty reason `environment-ibl-complete` when one update round converges and finishes.
- Incremental planner requirements:
	- `environment-ibl` must force full-frame rendering and must not reset temporal history.
	- `environment-ibl-complete` must force full-frame rendering and must reset temporal history.

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

```bash
bun tests/test_environment_ibl_update_runtime.mjs
```

## Errors & Diagnostics
- `environment-ibl-update-bake-failed` must be emitted as a warning when runtime bake fails.
- When a bake task is aborted due to a newer request, the aborted task result must not be applied.
- When skybox texture is unavailable or invalid, runtime update must skip scheduling and must not mutate probe data.

## Compatibility / Breaking Changes
No breaking API changes are introduced.
The new runtime update path is opt-in via `EnvironmentIBLUpdateOptions.enabled`.
