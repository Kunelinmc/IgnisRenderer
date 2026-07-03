# Environment IBL Renderer Update Removal Contract
## Scope
This document defines the removal contract for Renderer-owned environment IBL
runtime updates. The contract covers removed APIs, renderer pipeline behavior,
and the replacement path through standalone SH projection and `IBLPrefilter`.

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
- `Renderer` must not import IBL prefilter or SH projection helpers for warmup
  or runtime update orchestration.
- `Renderer` must not pass a WebGPU compute source to probe-capture
  prefiltering.
- Applications must call `projectEnvironmentTextureToSH` directly when
  environment SH data needs to be regenerated.
- Applications must call `IBLPrefilter` or `prefilterEnvironmentIBL` directly
  when environment specular IBL data needs to be regenerated.
- Applications must assign generated SH coefficients and prefiltered maps to
  the target probes explicitly.

## Usage
```ts
const sh = projectEnvironmentTextureToSH(environmentTexture, {
	maxSampleWidth: 128,
	maxSampleHeight: 64,
});

for (const lightProbe of environmentLightProbes) {
	lightProbe.sh = sh.map((coefficient) => ({ ...coefficient }));
}

const prefilteredMap = await prefilterEnvironmentIBL(environmentTexture, {
	backend: renderer,
	acceleration: "auto",
	maxSampleWidth: 128,
	maxSampleHeight: 64,
	maxMipLevels: 5,
});

for (const reflectionProbe of environmentReflectionProbes) {
	reflectionProbe.prefilteredMap = prefilteredMap;
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
- Environment SH projection failures must be surfaced by
  `projectEnvironmentTextureToSH`.
- Environment specular prefilter failures must be surfaced by `IBLPrefilter` or
  `prefilterEnvironmentIBL`.
- Applications that replace probe data must request a render with an existing
  dirty reason such as `lighting`.

## Compatibility / Breaking Changes
This change is breaking. Renderer-owned environment IBL update APIs, the
`environment-ibl-update` pipeline stage, `bakeEnvironmentIBLFromEnvironmentMap`,
and `EnvironmentIBLBake*` types are removed. Consumers must migrate to explicit
`projectEnvironmentTextureToSH` and `IBLPrefilter` or `prefilterEnvironmentIBL`
calls.
