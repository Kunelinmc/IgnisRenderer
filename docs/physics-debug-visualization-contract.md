# Physics Debug Visualization Contract

## Scope
This document defines the data contract for future physics debug visualization.
It does not require any renderer overlay, shader, or draw-pass implementation.

## Background
Physics debugging needs a stable engine-facing data shape before renderer
integration. The first implementation may expose collider outlines, AABB lines,
and joint axes. Contact points require adapter contact-manifold support and may
be unavailable until adapters expose that data.

## API/Contract
`PhysicsDebugGeometry` must contain:

- `lines: PhysicsDebugLine[]`
- `points: PhysicsDebugPoint[]`

`PhysicsDebugLine` must include:

- `kind: "collider" | "aabb" | "joint-axis" | "contact-point"`
- `worldId: PhysicsWorldId`
- `from: IVector3`
- `to: IVector3`

`PhysicsDebugLine` may include `bodyId`, `colliderId`, `jointId`, and `color`.

`PhysicsDebugPoint` must include:

- `kind: "contact-point"`
- `worldId: PhysicsWorldId`
- `position: IVector3`

`PhysicsDebugPoint` may include `bodyAId`, `bodyBId`, `normal`, and `color`.

The debug geometry contract must be renderer-agnostic. Renderers may consume the
geometry in a later overlay implementation, but physics modules must not submit
rendering work directly.

## Usage
```ts
import type { PhysicsDebugGeometry } from "../src/physics/types";

const debugGeometry: PhysicsDebugGeometry = {
	lines: [
		{
			kind: "aabb",
			worldId: "main",
			bodyId: "physicsBody_1",
			from: { x: -1, y: 0, z: 0 },
			to: { x: 1, y: 0, z: 0 },
		},
	],
	points: [],
};
```

## Errors & Diagnostics
Debug geometry producers should omit data they cannot compute. Missing
contact-manifold support must produce an empty `points` array instead of
throwing.

## Compatibility / Breaking Changes
N/A
