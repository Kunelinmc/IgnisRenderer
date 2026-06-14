# Environment IBL Renderer Update Removal Contract
## Scope
This document defines the removal contract for Renderer-owned environment IBL
runtime updates. The contract covers removed APIs, renderer pipeline behavior,
and the replacement path through standalone `IBLPrefilter`.

## Background
Environment IBL update work previously lived in `Renderer` and could be driven
incrementally from scene environment changes. That ownership made prefiltering
depend on frame scheduling and backend selection inside `Renderer`. Environment
IBL prefiltering is now an explicit operation owned by applications or tooling.

## API/Contract
- `Renderer.setEnvironmentIBLUpdateOptions(options)` must not exist.
- `Renderer.getEnvironmentIBLUpdateOptions()` must not exist.
- `Renderer.requestEnvironmentIBLUpdate()` must not exist.
- `EnvironmentIBLUpdateRuntime` must not be part of the runtime source tree.
- The default renderer pipeline must not include an
  `environment-ibl-update` stage.
- `Renderer` must not import `EnvironmentIBLBaker` for warmup or runtime update
  orchestration.
- `Renderer` must not pass a WebGPU compute source to probe-capture baking.
- Applications must call `IBLPrefilter` or
  `bakeEnvironmentIBLFromEnvironmentMap` directly when environment IBL data
  needs to be regenerated.
- Applications must assign generated SH coefficients and prefiltered maps to
  the target probes explicitly.

## Usage
```ts
const baked = await bakeEnvironmentIBLFromEnvironmentMap(environmentTexture, {
	backend: renderer,
	acceleration: "auto",
});

for (const lightProbe of environmentLightProbes) {
	lightProbe.sh = baked.sh.map((coefficient) => ({ ...coefficient }));
}

for (const reflectionProbe of environmentReflectionProbes) {
	reflectionProbe.prefilteredMap = baked.prefilteredMap;
	reflectionProbe.markRuntimeDirty();
}

renderer.requestRender("lighting");
```

```bash
bun tests/static/lighting/test_ibl_prefilter.mjs
```

## Errors & Diagnostics
- Renderer frame execution must not emit `environment-ibl` or
  `environment-ibl-complete` dirty reasons.
- Environment IBL bake failures must be surfaced by the direct prefilter or
  bake call.
- Applications that replace probe data must request a render with an existing
  dirty reason such as `lighting`.

## Compatibility / Breaking Changes
This change is breaking. Renderer-owned environment IBL update APIs and the
`environment-ibl-update` pipeline stage are removed. Consumers must migrate to
explicit `IBLPrefilter` or `bakeEnvironmentIBLFromEnvironmentMap` calls.
