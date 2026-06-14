# IBL Prefilter Contract
## Scope
This document defines the standalone environment IBL prefilter contract.
The contract covers `IBLPrefilter`, `prefilterEnvironmentIBL`, backend
selection, output texture shape, and ownership boundaries with `Renderer`.

## Background
Environment specular prefiltering is independent from frame rendering.
`Renderer` must not schedule, own, or configure IBL prefilter work. Applications
and tooling must invoke `IBLPrefilter` or `bakeEnvironmentIBLFromEnvironmentMap`
explicitly, then assign the resulting texture or SH coefficients to probes.

```mermaid
flowchart TD
	App[Application or Tooling] --> Prefilter[IBLPrefilter]
	App --> Baker[bakeEnvironmentIBLFromEnvironmentMap]
	Baker --> SH[SH Projection]
	Baker --> Prefilter
	Prefilter --> CPU[CPU or Worker Prefilter]
	Prefilter --> GPU[WebGPU ComputeRuntime]
	GPU --> Facade[WebGPU ComputeFacade]
	Facade --> Session[WebGPUBackendSession]
	Prefilter --> Texture[Prefiltered HDR Texture]
	App --> Probe[ReflectionProbe or LightProbe]
	Probe --> Renderer[Renderer Frame Rendering]
```

## API/Contract
- `IBLPrefilter` must accept an optional `Renderer`, backend session, or WebGPU
  compute source at construction time.
- `IBLPrefilter.prefilter(envMap, options)` must return an HDR `Texture` whose
  `mipmaps` encode roughness levels.
- `prefilterEnvironmentIBL(envMap, options)` must provide a one-shot helper with
  the same behavior as constructing `IBLPrefilter` and calling `prefilter`.
- `IBLPrefilterOptions` must use `maxSampleWidth`, `maxSampleHeight`, and
  `maxMipLevels` for output limits.
- `IBLPrefilterOptions.acceleration` must support `auto`, `worker`, `cpu`, and
  `webgpu`.
- If `acceleration` is `webgpu`, a WebGPU renderer, backend session, or compute
  source must be provided.
- If `acceleration` is `auto`, WebGPU may be used when a valid WebGPU source is
  available; otherwise worker or CPU fallback may be used.
- `Renderer` must not expose environment IBL bake or update methods.
- `Renderer.warmup()` must not create `LightProbe` instances, assign
  `LightProbe.sh`, or assign `ReflectionProbe.prefilteredMap`.

## Usage
```ts
const prefilter = new IBLPrefilter(renderer);
const prefilteredMap = await prefilter.prefilter(environmentTexture, {
	acceleration: "auto",
	maxSampleWidth: 128,
	maxSampleHeight: 64,
	maxMipLevels: 5,
});

reflectionProbe.prefilteredMap = prefilteredMap;
reflectionProbe.markRuntimeDirty();
```

```ts
const baked = await bakeEnvironmentIBLFromEnvironmentMap(environmentTexture, {
	backend: renderer,
	acceleration: "auto",
});

lightProbe.sh = baked.sh;
reflectionProbe.prefilteredMap = baked.prefilteredMap;
```

```bash
bun tests/static/lighting/test_ibl_prefilter.mjs
```

## Errors & Diagnostics
- `IBLPrefilter.prefilter()` must throw when `envMap` is not a valid 2D
  equirectangular texture or cubemap.
- `IBLPrefilter.prefilter()` must throw an `AbortError` when `signal` is
  aborted.
- Explicit `webgpu` acceleration must throw when no WebGPU renderer, backend
  session, or compute source is available.
- Explicit `worker` acceleration must throw when the Worker API is unavailable.

## Compatibility / Breaking Changes
This change is breaking.
`Renderer.setEnvironmentIBLUpdateOptions()`,
`Renderer.getEnvironmentIBLUpdateOptions()`,
`Renderer.requestEnvironmentIBLUpdate()`, `WarmupOptions.environmentIBLBake`,
and `WarmupOptions.includeEnvironmentIBLBake` are removed. Consumers must use
`IBLPrefilter` or `bakeEnvironmentIBLFromEnvironmentMap` directly.
`IRenderBackend` providers are no longer accepted as compute sources; use the
active `Renderer`, an explicit `IRenderBackendSession`, or a compute facade.
