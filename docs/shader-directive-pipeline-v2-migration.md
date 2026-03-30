# Shader Directive Pipeline V2 Migration

## Scope

This migration is **breaking** and intentionally removes the legacy
`preprocessEngineShaderDirectives(...)` path.

The compile flow is now:

1. Stage A (`ShaderDirectiveStage`): directives only (`#import`, `#inject`)
2. Stage B (`ShaderBackendCompileStage`): backend runtime rules + diagnostics

`mode` ownership moved to backends:

- `WebGPUBackend`: default `strict`
- `WebGLBackend`: default `warn`

## API Mapping

- `preprocessEngineShaderDirectives(...)` -> `ShaderBackendCompileStage.compile(...)`
- `ENGINE_DIRECTIVE_RUNTIME` -> backend-owned `ShaderBackendCompileStage`
- Loader preprocess output -> raw/composite source only, then compile-time A->B
- Global directive behavior -> backend profile registry
  (`webgpu/v1`, `webgl/v1`, `software/v1`)

## Required New Concepts

- `ShaderBackendId = "webgpu" | "webgl" | "software"`
- `ShaderDirectiveProfile` per backend (including explicit `software` no-op)
- `ShaderDirectiveProfileRegistry` completeness assertion at startup
- `ShaderDirectiveCompileHook` (backend-instance scope, async allowed)
- `directiveFingerprint = profile.id + "|rev:" + profile.revision + "|hook:" + token`

## Hook Contract Changes

- If a hook returns a patch, `token` is required.
- Token format: `^[A-Za-z0-9._:/-]{1,128}$`
- Missing/invalid token:
  - `strict`: fail compile
  - `warn`: warning + fallback to `token=base` and no patch
- Same token with different patch payload triggers
  `hook-token-collision` warning and disables that patch.

## WebGL Source Loading Changes

- Removed module-level eager shader constants.
- Shader sources now come from backend-owned `WebGLShaderSourceFactory`.
- Prepare factory during backend initialization (`prepareAll()`), then compile via
  A->B at shader compile time.

## Guardrails

- Static migration guard test now fails if code references:
  - `preprocessEngineShaderDirectives`
  - `ENGINE_DIRECTIVE_RUNTIME`

## Common Errors

- `Missing shader directive profile for backend "software"`:
  registry is incomplete; register explicit `software` profile.
- Hook token errors in strict mode:
  ensure hook returns stable token whenever it returns a patch.
