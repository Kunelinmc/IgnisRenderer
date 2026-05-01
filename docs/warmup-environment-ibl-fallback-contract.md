# Warmup Environment IBL Fallback Contract
## Scope
This document defines the warmup-time contract for environment IBL baking and skybox specular fallback in `Renderer`.
The contract covers `WarmupOptions` semantics and runtime side effects that affect environment specular lighting.

## Background
Before this contract, `includeEnvironmentIBLBake` implicitly controlled whether skybox specular fallback was enabled.
That coupling made warmup configuration ambiguous because disabling bake also disabled fallback, even when users wanted fallback to remain enabled.

## API/Contract
- `WarmupOptions` must expose `allowSkyboxSpecularFallback?: boolean` as an independent control.
- `Renderer.warmup(options)` must apply `allowSkyboxSpecularFallback` only when the option is explicitly provided.
- `Renderer.warmup(options)` must not derive skybox fallback state from `includeEnvironmentIBLBake`.
- `includeEnvironmentIBLBake` must control only whether warmup performs environment IBL bake work.
- When `includeEnvironmentIBLBake` is `false`, warmup must skip environment IBL bake and must preserve the current fallback state unless `allowSkyboxSpecularFallback` is explicitly provided.
- To reproduce legacy behavior where disabling bake also disables fallback, callers should set:
	- `includeEnvironmentIBLBake: false`
	- `allowSkyboxSpecularFallback: false`

## Usage
```ts
await renderer.warmup({
	includeEnvironmentIBLBake: true,
	allowSkyboxSpecularFallback: false,
	environmentIBLBake: { acceleration: "cpu" },
});
```

```ts
await renderer.warmup({
	includeEnvironmentIBLBake: false,
	allowSkyboxSpecularFallback: false,
});
```

```bash
bun tests/test_renderer_warmup_lightprobe.mjs
```

## Errors & Diagnostics
- If `allowSkyboxSpecularFallback` is omitted, warmup should keep the current renderer fallback state.
- If `includeEnvironmentIBLBake` is `false`, no bake-specific progress events should be emitted.
- If bake is enabled but skybox data is unavailable or invalid, warmup must skip bake work without mutating fallback state unless explicitly requested by `allowSkyboxSpecularFallback`.

## Compatibility / Breaking Changes
Behavioral compatibility is changed:
- `includeEnvironmentIBLBake` no longer toggles skybox specular fallback.
- Existing integrations that relied on implicit coupling should pass `allowSkyboxSpecularFallback` explicitly.
