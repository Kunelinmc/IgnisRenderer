# AGENTS.md

This file provides collaboration guidance for AI/code agents working in this repository.

## Scope

- Applies to the whole repo unless a subdirectory includes its own `AGENTS.md`.

## How To Work Here

- Prefer minimal, incremental changes that are easy to review.
- Avoid unrelated refactors.
- If a change could affect runtime behavior, explain the impact in the response.
- If tests exist, prefer running or updating them when changes are non-trivial.

## Conventions

- Use existing coding style and patterns.
- Avoid hardcoding magic numbers or strings; use named constants for values with specific significance.
- Keep comments concise and meaningful.
- Default to ASCII in new content unless the file already uses Unicode.
- Use English for code comments and commit messages.
- Prefer explicit type-only imports using `import type { Foo } from './foo'` to avoid unintended runtime dependencies.
- Avoid type assertions (as) unless necessary; never use double assertions (as unknown as).

## Naming & Type Standards

### Naming Conventions

- **Classes / Interfaces / Types**: `PascalCase`.
- **Functions / Methods / Variables / Properties**: `camelCase`.
- **Constants**: `UPPER_SNAKE_CASE`. Use constants to replace magic numbers and frequently used strings.
- **Private Properties/Methods**: Use `private` keyword and `_` prefix (e.g., `private _intensity`).
- **Files**:
  - Class/Component files: `PascalCase.ts` (e.g., `Light.ts`).
  - Logic/Utility/Type files: `camelCase.ts` (e.g., `types.ts`).

### Type Definition Rules

- **Interface**: Use for object structures and API contracts.
- **Type**: Use for union types, intersection types, or primitive aliases.
- **Strictness**: Avoid `any`. Use `unknown` if the type is truly unknown, or better, define a generic or a specific interface.
- **Centralization**: Common types should be defined in a module-level `types.ts` (e.g., `src/maths/types.ts` for engine-wide math types) to avoid redundant definitions.

### Documentation

- Use JSDoc for public classes and methods.
- Keep documentation up-to-date with code changes.

## Contact/Approval

- If a task has unclear requirements, ask a brief clarifying question.
