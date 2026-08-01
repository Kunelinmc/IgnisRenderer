# Warmup Environment IBL Contract

## Scope

This document defines the `Renderer.warmup()` contract for environment IBL.
The contract covers warmup side effects, fallback behavior, and migration to
standalone SH projection and `IBLPrefilter`.

## Background

Environment IBL prefiltering is now an explicit application or tooling step.
Warmup prepares renderer/backend resources only; it must not derive lighting
probe data from `Environment.iblTexture`.

## API/Contract

- `WarmupOptions` must not expose `allowEnvironmentSpecularFallback`.
- `WarmupOptions` must not expose `includeEnvironmentIBLBake`.
- `WarmupOptions` must not expose `environmentIBLBake`.
- `Renderer.warmup(options)` must not project environment SH or prefilter
  environment IBL.
- `Renderer.warmup(options)` must not synthesize IBL input from
  `environment.backgroundTexture`.
- `Renderer.warmup(options)` must not synthesize background input from
  `environment.iblTexture`.
- `Renderer.warmup(options)` must not create `LightProbe` instances.
- `Renderer.warmup(options)` must not mutate `LightProbe.sh`.
- `Renderer.warmup(options)` must not mutate
  `ReflectionProbe.prefilteredMap`.

## Usage

```ts
await renderer.warmup();
```

```ts
const sh = projectEnvironmentTextureToSH(environmentTexture, {
	maxSampleWidth: 128,
	maxSampleHeight: 64,
});

const prefilteredMap = await prefilterEnvironmentIBL(environmentTexture, {
	service: { backend: webgpuBackend },
	acceleration: "auto",
	maxSampleWidth: 128,
	maxSampleHeight: 64,
	maxMipLevels: 5,
});

lightProbe.sh = sh;
reflectionProbe.prefilteredMap = prefilteredMap;
```

```bash
bun tests/static/renderer/test_renderer_warmup_lightprobe.mjs
```

## Errors & Diagnostics

- `Renderer.warmup()` must not emit environment SH projection or IBL prefilter
  progress events.
- `Renderer.warmup()` must not warn when `environment.iblTexture` is missing,
  invalid, or a load-error fallback.
- Environment SH projection errors must come from `projectEnvironmentTextureToSH`,
  not from `Renderer.warmup()`.
- Environment specular prefilter errors must come from `IBLPrefilter` or
  `prefilterEnvironmentIBL`, not from `Renderer.warmup()`.

## Compatibility / Breaking Changes

This change is breaking.
`WarmupOptions.includeEnvironmentIBLBake` and
`WarmupOptions.environmentIBLBake` are removed. Consumers must invoke
`projectEnvironmentTextureToSH` and `IBLPrefilter` or
`prefilterEnvironmentIBL` explicitly before assigning probe data.
