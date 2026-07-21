# Post-Process Execution Declaration Migration

## Scope

This document describes the one-time migration from graph/context metadata and
pass-level resource descriptor methods to execution declarations.

## Background

The old API represented one resource in several independent places. The new API
uses one backend implementation declaration and a fixed resource accessor.

## API/Contract

- Move `placement`, `order`, and `incremental` into `schedule`.
- Replace `metadata.graph` and pass descriptor methods with
  `describeExecution()`.
- Embed every history/transient descriptor beside its logical uses.
- Replace context property bindings with `context.resources.get*()` calls.
- Replace `requiresHiZ` with a required `"backend:frame-hiz"` shared entry.
- Replace publication callbacks with writes to assigned `color.output` and a
  successful `{ ran: true }` result.

## Usage

```ts
public describeExecution(): PostProcessExecutionDeclaration {
	return {
		color: { access: "read", output: "new-version" },
		gBuffer: [{
			semantic: "motion",
			access: "read",
			usage: "sampled",
		}],
		histories: [{
			descriptor: { id: "taa", usage: ["sampled", "render-target"] },
			read: [{ access: "read", usage: "sampled" }],
			write: [{ access: "write", usage: "storage" }],
		}],
	};
}
```

## Errors & Diagnostics

Migration is incomplete when old metadata fields or descriptor overrides remain,
when a resource getter names an undeclared ID, or when a reported history update
does not have a declared write use. These conditions must fail type checking or
runtime contract validation.

## Compatibility / Breaking Changes

No compatibility adapter is provided. All built-in and custom implementations
must migrate in the same release as the public contract change.
