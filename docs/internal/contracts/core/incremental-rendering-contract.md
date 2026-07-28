# Incremental Rendering Coverage Contract

## Scope

This document defines the public per-frame incremental coverage contract for
`Renderer.renderFrame()` and the internal backend preservation report.

## Background

Incremental rendering reuses prior frame output outside planner-selected dirty
tiles. A planner decision alone does not prove that post-processing or
presentation preserved those tiles, so the public result exposes both states.

## API/Contract

- Every `RenderFrameResult` must include `incremental`.
- `IncrementalFrameStatus.plan` must describe the planner inputs and selected
  dirty rectangles and tiles for the completed frame.
- `plannedCoverage` and `finalOutputCoverage` must use an
  `IncrementalTileCoverage` tile grid with sorted, non-overlapping half-open
  `IncrementalTileRange` values.
- The updated and reusable ranges of each coverage object must partition its
  complete tile grid.
- `finalOutputCoverage` must report `"full"` unless every backend path that
  affects final visible output proves it preserved non-dirty tiles.
- `"unchanged"` must be used for an on-demand clean frame. Its reusable range
  must cover the whole tile grid.
- `IRenderBackend.getCompletedFrameCoverage()` is internal. It must return
  `"dirty-tiles"` only after a successful frame that preserved non-dirty tiles;
  otherwise it must return `"full-frame"`.
- A post-process pass must return `preservesOutsideDirtyTiles: true` to take
  part in a local-preservation guarantee. Omitted declarations must be treated
  conservatively as full-frame output work.

## Usage

```ts
const result = await renderer.renderFrame(performance.now());
const coverage = result.incremental.finalOutputCoverage;

if (coverage.mode === "partial") {
	for (const range of coverage.reusableTileRanges) {
		console.info(range.startTile, range.endTileExclusive);
	}
}
```

## Errors & Diagnostics

- A rejected frame must not return an `IncrementalFrameStatus`.
- A failed or aborted frame must not replace the last successful incremental
  stats snapshot.
- Backends that cannot prove final-output preservation must return the
  conservative `"full-frame"` internal coverage report.

## Compatibility / Breaking Changes

- `RenderFrameResult` now requires `incremental` in both union branches.
- Consumers that constructed `RenderFrameResult` values must add the required
  `IncrementalFrameStatus` value.
