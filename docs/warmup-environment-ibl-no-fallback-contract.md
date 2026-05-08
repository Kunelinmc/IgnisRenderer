# Warmup Environment IBL No-Fallback Contract
## Scope
This document defines the warmup-time contract for environment IBL baking in
`Renderer`.
The contract covers `WarmupOptions` semantics and runtime side effects for the
`Environment` model.

## Background
The renderer now uses an explicit `Environment` object with independent
`backgroundTexture` and `iblTexture`.
Background rendering and IBL sampling must not fall back to each other.

## API/Contract
- `WarmupOptions` must not expose `allowEnvironmentSpecularFallback`.
- `Renderer.warmup(options)` must treat `includeEnvironmentIBLBake` as a
  bake-only switch.
- If `includeEnvironmentIBLBake` is `false`, warmup must skip environment IBL
  bake work.
- If `scene.environment.iblTexture` is `null` or `scene.environment.lightingEnabled`
  is `false`, warmup must skip environment IBL bake work.
- Warmup must not synthesize IBL input from `backgroundTexture`.
- Warmup must not synthesize background input from `iblTexture`.

## Usage
```ts
await renderer.warmup({
	includeEnvironmentIBLBake: true,
	environmentIBLBake: { acceleration: "cpu" },
});
```

```ts
await renderer.warmup({
	includeEnvironmentIBLBake: false,
});
```

```bash
bun tests/test_renderer_warmup_lightprobe.mjs
```

## Errors & Diagnostics
- If `includeEnvironmentIBLBake` is `false`, no bake-specific progress events should be emitted.
- If bake is enabled but `iblTexture` is unavailable or invalid, warmup must
  skip bake work.
- If bake is enabled but `iblTexture` is a load-error fallback texture, warmup
  should emit diagnostics and skip bake work.

## Compatibility / Breaking Changes
Behavioral compatibility is changed:
- `allowEnvironmentSpecularFallback` is removed.
- Integrations must configure `scene.environment.backgroundTexture` and
  `scene.environment.iblTexture` explicitly.
