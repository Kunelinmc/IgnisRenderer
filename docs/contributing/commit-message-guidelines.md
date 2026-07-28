# Commit Message Guidelines

## Scope

This document defines commit message requirements for changes in the
IgnisRenderer repository.

## Background

This document is the canonical commit message contract for all contributors.
Automation-specific entrypoints, such as `AGENTS.md`, may summarize or reference
these rules, but this file owns the complete workflow requirement.

## API/Contract

### Header Format

- Commit headers must use Conventional Commits style.
- The full header shape must be `type(scope)!: subject`, where `scope` and `!`
  are used only when allowed by this document.
- `type` is required, lowercase, and must match one of the approved types below.
- `scope` is optional but should be used when a change is focused on one
  subsystem.
- Scopes should use repository terms such as `webgpu`, `webgl`, `software`,
  `ecs`, `pipeline`, `postprocess`, `shaders`, `physics`, `animation`, `docs`,
  `tests`, or `build`.
- `!` is required when the commit introduces a breaking API, behavior, file
  layout, shader contract, or backend contract change.
- `subject` must be concise, imperative, and must not end with a period.
- The full header should be 72 characters or fewer when practical.

### Approved Types

- `feat`: adds a user-visible feature, renderer capability, backend feature,
  public API, shader feature, simulation feature, or supported workflow.
- `fix`: corrects a bug, rendering artifact, crash, invalid state transition,
  incorrect math, race condition, resource leak, or test failure.
- `perf`: improves runtime performance, memory use, allocation behavior,
  scheduling, batching, shader cost, or frame execution without changing
  observable behavior.
- `refactor`: restructures implementation without changing public behavior.
- `docs`: updates documentation, examples, public API comments, migration
  notes, or contributor guidance.
- `test`: adds, updates, or removes tests, fixtures, snapshots, mocks, or test
  infrastructure without changing production behavior.
- `build`: changes package scripts, bundling, TypeScript configuration,
  dependency metadata, or generated build artifacts.
- `ci`: changes GitHub Actions, release automation, validation workflows, or
  other continuous integration configuration.
- `style`: changes formatting, lint-only concerns, whitespace, ordering, or
  naming that does not affect behavior.
- `chore`: performs repository maintenance that does not fit another type.
- `revert`: reverts a previous commit. The body should name the reverted commit
  hash and reason.

### Writing Rules

- Commit messages must use English.
- Commits must use the smallest accurate `type`.
- Commits must not use `chore` when `feat`, `fix`, `perf`, `docs`, `test`,
  `build`, or `ci` applies.
- Commits should prefer a precise scope over a broad scope.
- Subjects should describe the observable outcome, not the implementation step.
- Commit messages with body text must include a blank line after the header.
- Body and footer lines should wrap at 100 characters or fewer.
- Footers may include issue links, breaking changes, and co-authors.
- `BREAKING CHANGE:` must be present when `!` is present.
- Each commit should represent one logical change.
- Unrelated source, test, docs, and formatting changes should be split into
  separate commits when practical.
- Commit bodies should mention validation when it matters, using exact commands
  such as `bunx tsc --noEmit` or
  `bun tests/static/webgpu/test_webgpu_bridge.mjs`.
- Generated files must not be included unless the repository expects them for
  that change.

## Usage

Examples:

```text
feat(webgpu): add deferred resolve resource cache

fix(postprocess): clamp TAA history rect before sampling

perf(software): reuse tile buffers during rasterization

docs(workflow): document commit message conventions

feat(core)!: rename frame pass registry contract

BREAKING CHANGE: `RenderPipelineRegistry.registerStage` has been replaced by
`RenderPipelineRegistry.registerPass`.
```

## Errors & Diagnostics

- A commit hook or reviewer should reject headers with missing or unapproved
  `type` values.
- A commit hook or reviewer should reject breaking commits that use `!` without
  a `BREAKING CHANGE:` footer.
- Reviewers should request a more precise `scope` when a broad scope hides the
  affected subsystem.
- Reviewers should request a different `type` when `chore` hides an observable
  feature, fix, performance change, documentation update, test update, build
  change, or CI change.

## Compatibility / Breaking Changes

Changing this document changes repository workflow expectations. Such changes
should use the `docs` type and should update contributor-facing summaries such
as `AGENTS.md` when their guidance changes.
