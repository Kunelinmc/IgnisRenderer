# Documentation

## Scope

This document is the entry point for IgnisRenderer documentation. The
documentation tree separates package-consumer guidance from contributor-facing
contracts, architecture, reference material, migrations, and workflows.

## Background

The supported public API boundary is the package root exported by
`src/index.ts`. Subsystem `index.ts` files are not automatically supported
package entry points.

Existing documents that combine public behavior with backend or implementation
requirements remain intact under `contracts/`. They may be split into
dedicated public documentation in a later documentation phase.

## API/Contract

Documentation must be placed according to its primary audience and purpose:

| Directory | Content |
| --- | --- |
| `public/` | Consumer-facing API documentation and usage guides |
| `architecture/` | Engine and rendering architecture |
| `contracts/` | Normative implementation and backend contracts |
| `reference/` | Maintainer reference material |
| `migrations/` | Historical and active migration contracts |
| `contributing/` | Repository contribution workflows |

New documents must not be added directly under `docs/` except for this entry
point.

## Usage

Package consumers should start with:

- [Renderer API](public/renderer.md)
- [Interaction API](public/interaction.md)
- [ComputeRuntime usage](public/compute-runtime-interface-usage.md)

Contributors should start with:

- [Engine architecture](architecture/engine-architecture.md)
- [Rendering contracts](architecture/rendering-pipeline-and-shader-contracts.md)
- [Renderer and backend core contract](contracts/renderer-contract.md)
- [Commit message guidelines](contributing/commit-message-guidelines.md)

## Errors & Diagnostics

If a documentation link fails after the directory migration, search for the
unchanged filename under `docs/public/`, `docs/contracts/`, or another
top-level documentation category. Repository references must use the new
canonical path.

Documents that primarily specify implementation behavior must be moved to the
appropriate `contracts/` subsystem instead of being added to
`public/`.

## Compatibility / Breaking Changes

The documentation paths were reorganized without compatibility stubs. Existing
filenames and document content remain unchanged, but links to the previous
locations must be updated.
