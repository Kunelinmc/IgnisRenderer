# Irradiance Probe Grid

## Scope

This document defines the V1 `IrradianceProbeGrid` scene light contract.
`IrradianceProbeGrid` provides diffuse irradiance from a regular local box grid
of spherical harmonics probes. V1 supports one active grid per frame.

## Background

`LightProbe` provides global or localized SH irradiance from one probe volume.
`IrradianceProbeGrid` extends that diffuse irradiance path to a regular grid of
cells in a local box volume. The renderer samples the grid inside its coverage
volume before falling back to localized `LightProbe` and global SH lighting.

## API/Contract

`LightType.IrradianceProbeGrid` must identify grid lights as
`"irradianceProbeGrid"`. `IrradianceProbeGrid` is part of `SceneLight` and the
public `src/lights` exports.

`IrradianceProbeGridParams.dimensions` must be provided as `{ x, y, z }`. Each
axis is floored to an integer and clamped to at least `1`. The total cell count
must be less than or equal to `256`; construction throws `RangeError` when the
limit is exceeded.

`halfExtents` defaults to `{ x: 5, y: 5, z: 5 }`. The grid volume is the node
transform multiplied by this local box. `blendDistance` defaults to `0.15` and
controls edge fade outside the normalized box metric. `priority` defaults to
`0`; if multiple grids exist, the renderer selects the highest priority grid,
then the nearest grid center to the camera, then the lowest `id`.

`source` defaults to `"manual"`. `"capturedScene"` uses `ProbeCaptureRuntime`
to capture one cell at a time. Capture fields match `LightProbe`:
`captureUpdateMode`, `captureIntervalSeconds`, `captureResolution`,
`captureFar`, `includeEnvironment`, `includeMeshes`, `includeTransparent`,
`includeParticles`, and `includeShadows`.

Grid SH data must use the engine SH contract: L=3 with `16` coefficients per
cell. Cell indexing is `x` fastest, then `y`, then `z`.

`IrradianceProbeGrid.sh` and `getCellSH()` return mutable cell-owned SH
storage for backward compatibility. Direct mutation of `sh`, a cell SH array,
or a coefficient component must advance `textureRevision` and
`captureRevision`, mark the touched cell valid, and invalidate scene lighting.
Applications should use `setCellSH()` for authored writes and `clearCell()` for
invalidating a cell.

Public methods:

```ts
getCellIndex(x: number, y: number, z: number): number;
getCellSH(indexOrCoord: number | { x: number; y: number; z: number }): SHCoefficients;
setCellSH(indexOrCoord: number | { x: number; y: number; z: number }, coeffs: SHCoefficients): void;
clearCell(indexOrCoord: number | { x: number; y: number; z: number }): void;
requestCapture(indexOrCoord?: number | { x: number; y: number; z: number }): void;
getRuntimeCache(): IrradianceProbeGridRuntimeCache;
```

`setCellSH` must mark the cell valid and advance the texture revision.
`clearCell` must mark the cell invalid. `requestCapture()` without an argument
must request all cells.

`IrradianceProbeGridRuntimeCache.worldToGrid3x3` must be a `Matrix3` instance.
Consumers must read its row-major values from `worldToGrid3x3.elements`.

## Usage

```ts
import { IrradianceProbeGrid } from "../src/lights";
import { SH } from "../src/maths/SH";

const grid = scene.add(new IrradianceProbeGrid({
	dimensions: { x: 4, y: 2, z: 4 },
	halfExtents: { x: 8, y: 3, z: 8 },
	source: "capturedScene",
	captureUpdateMode: "manual",
	captureResolution: { width: 64, height: 32 },
}));

grid.position.set(0, 2, 0);
grid.requestCapture();

const authored = SH.empty();
authored[0] = { r: 12, g: 10, b: 8 };
grid.setCellSH({ x: 0, y: 0, z: 0 }, authored);
```

## Errors & Diagnostics

`RangeError` is thrown when `dimensions.x * dimensions.y * dimensions.z > 256`.

`RangeError` is thrown when `getCellIndex`, `getCellSH`, `setCellSH`,
`clearCell`, or `requestCapture` receives a cell outside the grid.

WebGPU and WebGL emit a warning when more than one grid is active in a frame;
only the selected V1 grid is used. WebGL compiles grid sampling only when the
scene shader can fit the optional grid sampler. It binds the SH texture to unit
`15` and emits a warning when that sampler path is unavailable.

If a captured-scene grid requests mesh capture without a compatible WebGPU face
capture source, capture falls back to analytic lights and environment only.

## Compatibility / Breaking Changes

The grid affects diffuse irradiance only. It does not affect reflection probe
specular IBL. Auto placement, per-object probe override, visibility masks, and
occlusion masks are not part of V1.

Breaking change: `IrradianceProbeGridRuntimeCache.worldToGrid3x3` is now a
`Matrix3` instance. Code that read a bare `Matrix3Arr` must read
`worldToGrid3x3.elements`.
