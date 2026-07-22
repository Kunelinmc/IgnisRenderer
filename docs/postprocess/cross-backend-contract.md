# Post-Process Cross-Backend Contract

## Scope

This document defines logical post-process passes, backend implementation
declarations, fixed execution contexts, and backend-owned execution.

## Background

Post-process resource behavior previously appeared in pass descriptor methods,
implementation graph metadata, and backend context-binding metadata. Those
independent declarations could disagree. The execution declaration is now the
single source of truth for resource existence, access, usage, and optionality.

## API/Contract

- `PostProcessPassConfig.schedule` must own placement, numeric order, and
  incremental metadata. Resource behavior must not appear in the schedule.
- `PostProcessPassConfig.label` may provide a human-readable pass name for
  diagnostics and consumer-facing metadata. It defaults to `id`.
- `PostProcessPassImplementation.describeExecution(request)` must return one
  complete `PostProcessExecutionDeclaration` for the active backend.
- Engine-owned implementations must compose explicit typed color and resource
  uses. A backend-name factory that infers access or usage must not be the
  source of an implementation declaration.
- The declaration must contain `color` and may contain `gBuffer`, `histories`,
  `transients`, and backend `shared` resource entries.
- History and transient entries must contain their allocation descriptor and
  all logical uses. A second descriptor API or graph-metadata overlay must not
  exist.
- Required G-buffer or shared resources must make a pass ineligible when they
  are unavailable. Optional resources must not affect eligibility.
- Implementations must receive a fixed backend execution context containing a
  `PostProcessResourceAccessor`. Backends must not synthesize pass-specific
  context properties from metadata.
- The accessor must expose assigned color input/output and typed getters for
  G-buffer, history, transient, and shared resources. Access to an undeclared
  resource must throw; a missing optional resource must return `null`.
- `color.output: "new-version"` must receive a backend-assigned output distinct
  from its input. `color.output: "preserve"` must not receive a new output.
- `{ ran: true }` must commit the assigned color output automatically.
  `{ ran: false }` must alias the planned output to its input and must not
  update history.
- `PostProcessPassResult.updatedHistoryIds` must contain only declared history
  IDs with write uses. The runtime must reject updates reported with
  `{ ran: false }`.
- A missing active-backend implementation must skip the pass and emit
  `postprocess-implementation-missing-<passId>`. Backends must not dispatch a
  fallback kernel by pass ID.
- Backend runtime implementation instances must remain backend-local and must
  be invalidated or destroyed with their owning device lifecycle.

## Usage

```ts
class CustomWebGPUImplementation {
	public describeExecution(): PostProcessExecutionDeclaration {
		return {
			color: { access: "read", output: "new-version" },
			transients: [{
				descriptor: { id: "custom:scratch", format: "rgba16float" },
				uses: [{ access: "write", usage: "storage" }],
			}],
		};
	}

	public execute(request, context): PostProcessPassResult {
		const source = context.resources.color.input;
		const target = context.resources.color.output;
		const scratch = context.resources.getTransient("custom:scratch");
		// Record commands from source through scratch into target.
		return { ran: true };
	}
}
```

## Errors & Diagnostics

- Malformed declarations must fail planning and identify the backend, pass ID,
  resource ID, and every detected violation.
- Duplicate history or transient IDs with incompatible descriptors must fail
  planning; the runtime must not select the first descriptor.
- Missing required G-buffer channels must retain the
  `postprocess-requirement-missing-<passId>` diagnostic.
- Missing required backend-shared resources must retain the
  `postprocess-backend-shared-unavailable-<passId>` diagnostic.
- Exceptions during execution must abort the active post-process transaction.

## Compatibility / Breaking Changes

- `PostProcessPassConfig.warningLabel` and `PostProcessPass.warningLabel` are
  renamed to `label`. Consumers must migrate to the new generic label name.
- `createPostProcessExecutionDeclaration()` is removed. Built-in passes compose
  immutable backend color policies and typed resource-use records directly.
- `PostProcessPassImplementation.metadata`, `PostProcessGraphMetadata`, context
  binding metadata, and controlled publication callbacks are removed.
- `getRequirements()`, `getHistoryDescriptors()`,
  `getTransientResourceDescriptors()`, and `getHistorySignature()` are removed.
- `PostProcessPassConfig.placement`, `order`, and `incremental` move under
  `schedule`.
- Custom passes must migrate directly to `describeExecution()` and the fixed
  resource accessor. No compatibility adapter is provided.
