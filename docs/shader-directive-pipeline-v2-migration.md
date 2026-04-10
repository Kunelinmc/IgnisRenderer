# Shader Directive Pipeline V2 Migration

## Scope

This document defines the migration contract for Shader Directive Runtime v2 behavior in IgnisRenderer.
The contract applies to:

- `ShaderRuntime`
- `ShaderDirectiveStage`
- `ShaderBackendCompileStage`
- backend directive profiles (`webgpu`, `webgl`, `software`)

## Background

Legacy global preprocess entry points were removed in favor of a staged compile flow:

1. Stage A (`ShaderDirectiveStage`) preprocesses directives.
2. Stage B (`ShaderBackendCompileStage` + `ShaderRuntime`) applies rule rewrites, validation, and injection.

The runtime now includes:

- conditional directive execution (`#if/#ifdef/#ifndef/#elif/#else/#endif/#undef`)
- rule rewrite hooks (`transform`, `replace`)
- source map schema version `2` with column-span-aware mapping
- robust WGSL `afterBindings` anchor detection for multi-line binding declarations

## API/Contract

- Directive execution contract:
  - `#if`, `#ifdef`, `#ifndef`, `#elif`, `#else`, `#endif`, and `#undef` must execute, not just emit unsupported diagnostics.
  - Conditional branches must use one mutable macro table across the entire preprocess pass.
  - Inactive branches must not apply side effects from non-conditional directives.
  - `#define`/`#undef` in an active branch must persist after `#endif`.
  - `#undef` for a missing macro must be a no-op.
- Conditional expression contract:
  - `#if/#elif` expressions must use deterministic `BigInt` integer semantics.
  - Supported literals must include decimal, `0x` hex, `0b` binary, and `0o` octal.
  - `defined(name)` and `defined name` must evaluate to `1` or `0`.
  - Invalid expressions or divide/mod-by-zero must emit `directive-conditional-expression-invalid`, and the branch condition must resolve to false.
- Rule execution contract:
  - Rule matching must be computed once from preprocessed source.
  - For each matched rule, execution order must be:
    1. `transform`
    2. `replace`
    3. `validate`
    4. `inject`
  - `transform` and `replace` may return `null`/`undefined` to indicate no change.
  - `processAsync()` must fail fast when async `transform`/`replace` rejects or throws.
  - Fail-fast errors must include rule id and hook kind.
- Source map contract:
  - `ShaderSourceSegmentMap.schemaVersion` must be `2` for runtime-generated maps.
  - `ShaderSourceSegment` may include optional column fields:
    - `generatedColumnStart`
    - `generatedColumnEnd`
    - `sourceColumnStart`
    - `sourceColumnEnd`
  - Hash/fingerprint calculations must include schema version and optional column fields.
  - Line-only source maps must remain valid and map as before.
- WGSL anchor contract:
  - `afterBindings` must detect binding declarations across multi-line attributes and multi-line `var ... ;`.
  - When bindings exist, `afterBindings` must resolve after the last fully closed binding declaration.
  - Anchor precedence must be:
    - `afterEnable`
    - `afterAliases = max(lastAlias + 1, afterEnable)`
    - `afterStruct = max(lastStructEnd + 1, afterAliases)`
    - `afterBindings = max(lastBindingEnd + 1, afterStruct)` if bindings exist, else `afterStruct`
    - `beforeEntryPoint`

## Usage

```ts
import { ShaderRuntime } from "../src/shaders/runtime/index.ts";

const runtime = new ShaderRuntime({ mode: "warn" });

runtime.registerRule({
	id: "user/rewrite-example",
	transform(context) {
		return {
			code: context.source.replace("TOKEN_A", "TOKEN_B"),
		};
	},
	replace() {
		return [
			{
				pattern: "TOKEN_B",
				replacement: "TOKEN_C",
			},
		];
	},
});

const result = runtime.process({
	code: `#define ENABLE 1
#if ENABLE
// TOKEN_A
#endif
@vertex
fn vsMain() -> @builtin(position) vec4<f32> {
	return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}`,
	language: "wgsl",
	stage: "vertex",
	entryPoint: "vsMain",
	sourceKind: "custom-material",
});
```

```bash
bun tests/test_shader_runtime.mjs
bun tests/test_shader_directive_pipeline_v2.mjs
bun run test
```

## Errors & Diagnostics

- `directive-conditional-expression-invalid`:
  - Triggered when a conditional expression is malformed or evaluates divide/mod-by-zero.
- `directive-conditional-elif-without-if`:
  - Triggered when `#elif` appears without an open conditional block.
- `directive-conditional-else-without-if`:
  - Triggered when `#else` appears without an open conditional block.
- `directive-conditional-endif-without-if`:
  - Triggered when `#endif` appears without an open conditional block.
- `directive-conditional-unterminated`:
  - Triggered when preprocess reaches EOF with open conditional blocks.
- `directive-undef-invalid`:
  - Triggered when `#undef` does not receive exactly one macro identifier.

## Compatibility / Breaking Changes

- Breaking migration from legacy API:
  - `preprocessEngineShaderDirectives(...)` must not be used.
  - `ENGINE_DIRECTIVE_RUNTIME` must not be used.
- Additive runtime API changes:
  - `ShaderRule` now supports `transform` and `replace`.
  - `ShaderSourceSegmentMap` now supports `schemaVersion`.
  - `ShaderSourceSegment` now supports optional column span fields.
- Runtime caches are memory-only in current implementation, so no persisted cache migration is required.
