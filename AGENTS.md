# AGENTS.md

This file provides collaboration guidance for AI/code agents working in this repository.

## Scope

- Applies to the whole repo unless a subdirectory includes its own `AGENTS.md`.

## Build & Test Commands

- **Dev server**: `npm run dev` - Starts Vite dev server
- **Run all tests**: `npm test` or `npm run test`
- **Run single test**: `npx tsx tests/<test-file>.mjs` (e.g., `npx tsx tests/test_lighting.mjs`)
- **Available test suites**:
  - `npm run test:lighting` - Lighting system tests
  - `npm run test:pointspot` - Point/Spot light tests
  - `npm run test:sh` - Spherical Harmonics regression tests
  - `npm run test:winding` - Model factory winding tests
  - `npm run test:sparse` - Sparse accessor tests

## Code Style Guidelines

### Imports

- Prefer **type-only imports**: `import type { Foo } from './foo'`
- Group imports: 1) external libs, 2) internal modules (separate groups with blank line)
- Use ES modules with `.ts` extension in imports
- Example:
  ```typescript
  import { Vector3 } from "../maths/Vector3";
  import type { IVector3 } from "../maths/types";
  ```

### Formatting

- Use **tabs** for indentation
- No semicolons required (ASI-friendly code)
- Single quotes for strings
- 80-100 character line limit (soft)

### Types & Interfaces

- **Interfaces**: Use for object structures and API contracts
- **Types**: Use for unions, intersections, primitive aliases
- Avoid `any`; use `unknown` or proper generics
- Avoid type assertions (`as`) unless necessary
- Never use double assertions (`as unknown as`)
- Centralize common types in `types.ts` files

### Naming Conventions

| Category | Convention | Example |
|----------|------------|---------|
| Classes/Interfaces/Types | PascalCase | `Light`, `IVector3` |
| Functions/Methods/Variables | camelCase | `computeLight`, `intensity` |
| Constants | UPPER_SNAKE_CASE | `MAX_SAMPLES` |
| Private members | `private _prefix` | `private _intensity` |
| Files (classes) | PascalCase.ts | `Light.ts` |
| Files (utils) | camelCase.ts | `types.ts` |

### Error Handling

- **Never** swallow errors silently
- Prefer typed error objects over raw strings
- Use early returns for validation
- Example:
  ```typescript
  if (!json) throw new Error("Failed to parse glTF JSON")
  ```

### Documentation

- Use JSDoc for public classes and methods
- Keep comments concise; focus on intent and constraints
- Don't restate the code
- Document non-obvious decisions

## How To Work Here

- Prefer minimal, incremental changes
- Avoid unrelated refactors
- If changes affect runtime, performance, or APIs, explain the impact
- Run tests for non-trivial changes
- Don't introduce new patterns without justification
- Prefer consistency over theoretical improvements
- When requirements are ambiguous, ask focused questions

## Architecture Conventions

### Coordinate Systems

- **World Space**: Right-handed (X: Right, Y: Up, Z: Towards Viewer)
- **View Space**: Eye at origin, -Z is forward
- **Screen Space**: (0,0) at top-left, pixel centers at +0.5
- **Depth Buffer**: Linear camera-space depth (positive distance from camera)

### Constants

- Use named constants for magic numbers
- Group constants in domain-specific classes (see `src/core/Constants.ts`)
- Example: `CoreConstants.EPSILON`, `LightingConstants.PBR_MIN_NDOTV`

### Testing Policy

- Do not remove tests to make code pass
- Update tests only if behavior change is intentional
- Tests use Node.js `assert` module
- Test files are `.mjs` in `tests/` directory

## Design Philosophy

- Explicit over implicit
- Composition over inheritance
- Fail fast on invalid input
- Keep core modules dependency-light
- Use English for code and comments

## Contact/Approval

- If a task has unclear requirements, ask a brief clarifying question.
