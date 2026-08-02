# Contributing Documentation

This document defines repository commit messages and documentation authoring
rules. `AGENTS.md` summarizes the highest-priority requirements; this document
owns the complete contributor workflow.

## Commit Messages

Commit headers use Conventional Commits:

```text
type(scope)!: subject
```

- `type` is required, lowercase, and selected from the approved list below.
- `scope` is optional and should name the smallest affected subsystem.
- `!` is required for breaking API, behavior, file-layout, shader, backend, or
  documentation-path changes.
- `subject` is concise, imperative, and does not end with a period.
- Headers should remain within 72 characters when practical.
- Breaking commits include a `BREAKING CHANGE:` footer.

### Approved Types

| Type | Use |
| --- | --- |
| `feat` | User-visible engine, backend, shader, or simulation capability |
| `fix` | Incorrect behavior, state, rendering, math, race, or resource lifetime |
| `perf` | Runtime, memory, allocation, scheduling, batching, or shader cost |
| `refactor` | Implementation structure without observable behavior changes |
| `docs` | Documentation, examples, comments, migrations, or contributor guidance |
| `test` | Tests, fixtures, snapshots, mocks, or test infrastructure |
| `build` | Scripts, bundling, TypeScript configuration, or dependency metadata |
| `ci` | Continuous integration and release automation |
| `style` | Formatting or ordering without behavior changes |
| `chore` | Maintenance that does not fit a more precise type |
| `revert` | Reversal of a previous commit |

Commit messages are written in English. Use repository terms such as `webgpu`,
`webgl`, `software`, `ecs`, `pipeline`, `postprocess`, `shaders`, `physics`,
`animation`, `docs`, `tests`, or `build` as scopes. Body and footer lines remain
within 100 characters, and bodies include exact validation commands when they
matter.

Examples:

```text
feat(webgpu): add deferred resolve resource cache

fix(postprocess): clamp TAA history rect before sampling

docs(workflow): document commit message conventions

feat(core)!: rename frame pass registry contract

BREAKING CHANGE: `RenderPipelineRegistry.registerStage` has been replaced by
`RenderPipelineRegistry.registerPass`.
```

## Documentation Structure

Documentation paths use one purpose directory and one file:

```text
docs/<category>/<document>.md
```

The supported categories are:

| Directory | Content |
| --- | --- |
| `public/` | Consumer workflows based on package-root exports |
| `architecture/` | Responsibilities, data flow, and design boundaries |
| `contracts/` | Normative subsystem and backend requirements |
| `migrations/` | Aggregated upgrade actions that remain applicable |
| `reference/` | Maintainer lookup tables and generated-style references |
| `contributing/` | Repository workflows and documentation policy |

New Markdown files are not added directly under `docs/` except for the root
index. Nested subsystem directories are not used.

## Document Ownership

A standalone document represents one of these durable units:

- A complete consumer workflow.
- A cross-component architecture boundary.
- A stable subsystem or backend contract.
- A maintained reference whose table-oriented content would obscure a
  subsystem contract.
- Aggregated migration or contribution guidance.

A single rendering effect, capability flag, probe behavior, loader variant,
material property, shader option, or backend pass is a section in its owning
subsystem document. It does not receive a standalone document.

Architecture explains responsibility, flow, and design intent. Normative
`must`, `should`, lifecycle, diagnostic, fallback, and compatibility rules live
in contracts. Public documents describe package-root workflows without exposing
backend-private ownership.

## Document Shapes

Use only the sections that serve the document type:

- Architecture: system model, responsibilities, flow, boundaries, and related
  documents.
- Contract: contract sections by capability, usage, diagnostics,
  compatibility, verification, and related documents.
- Public guide: overview, API, usage, troubleshooting, compatibility, and
  related documents.
- Reference: scope, lookup tables, maintenance, diagnostics, and related
  documents.
- Migration: current replacement paths and required upgrade actions.

Every document begins with one `#` title. Code blocks include a language tag.
Contract names, types, functions, and parameters use backticks. RFC-style
requirements use `must`, `should`, and `may` only in normative contracts.

Do not add editorial feature versions such as `v<number>` or
`phase <number>` to filenames, titles, or prose. Preserve version text only
when it is part of an
actual API identifier, file format, protocol, shader directive, or compatibility
input. Current limitations are written in the present tense.

## Navigation and Maintenance

- `docs/README.md` lists every maintained document by audience and subsystem.
- Every non-index document links to related architecture, contracts, guides, or
  references.
- Repository links use relative Markdown targets; code references use
  repo-relative paths in backticks.
- Host-local paths are not written into repository documentation.
- Completed implementation history is removed after active upgrade guidance is
  captured in `migrations/README.md`; Git remains the historical archive.
- Public interface or behavior changes update public guidance and corresponding
  contracts in the same change.

## Diagnostics

Documentation validation rejects broken relative links, legacy directory
layouts, excessive path depth, unindexed files, and editorial version markers
in filenames. Run:

```bash
bun tests/static/docs/test_docs_structure.mjs
git diff --check
```

## Related Documents

- [Documentation index](../README.md)
- [Engine architecture](../architecture/engine.md)
- [Renderer contract](../contracts/renderer.md)
- [Migration guidance](../migrations/README.md)
