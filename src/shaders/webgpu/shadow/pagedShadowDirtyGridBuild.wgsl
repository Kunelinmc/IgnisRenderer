const DIRTY_PHYSICAL_PAGE_RECORD_UINTS: u32 = 8u;
const DIRTY_GRID_CELL_COUNT: u32 = 64u;
const PAGE_CLIP_XY_MARGIN: f32 = 4.0;

struct PagedShadowDrawParams {
	candidateCount: u32,
	dirtyCapacity: u32,
	physicalPageCount: u32,
	pageSize: u32,
	physicalGridSize: u32,
	drawInstanceCapacity: u32,
	frameId: u32,
	_pad0: u32,
}

@group(0) @binding(0) var<uniform> params: PagedShadowDrawParams;
@group(0) @binding(1) var<storage, read> dirtyPhysicalPages: array<u32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> dirtyGridCounts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> dirtyGridOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> dirtyGridIndices: array<u32>;
@group(0) @binding(6) var<storage, read_write> dirtyPageUvRanges: array<vec4<f32>>;

fn dirtyGridCellIndex(dirtyBase: u32) -> u32 {
	let matrixIndex = dirtyPhysicalPages[dirtyBase + 2u];
	let pageX = dirtyPhysicalPages[dirtyBase + 3u];
	let pageY = dirtyPhysicalPages[dirtyBase + 4u];
	let pageGridSize = max(dirtyPhysicalPages[dirtyBase + 7u], 1u);
	let coarseX = min((pageX * 4u) / pageGridSize, 3u);
	let coarseY = min((pageY * 4u) / pageGridSize, 3u);
	return matrixIndex * 16u + coarseY * 4u + coarseX;
}

fn dirtyPageUvRange(dirtyBase: u32) -> vec4<f32> {
	let pageX = dirtyPhysicalPages[dirtyBase + 3u];
	let pageY = dirtyPhysicalPages[dirtyBase + 4u];
	let pageGridSize = max(dirtyPhysicalPages[dirtyBase + 7u], 1u);
	let gridSize = f32(pageGridSize);
	let px = f32(pageX);
	let py = f32(pageY);
	let atlasSize = max(gridSize * f32(max(params.pageSize, 1u)), 1.0);
	let marginUv = PAGE_CLIP_XY_MARGIN / atlasSize;
	return vec4<f32>(
		vec2<f32>(px / gridSize, py / gridSize) - vec2<f32>(marginUv),
		vec2<f32>((px + 1.0) / gridSize, (py + 1.0) / gridSize) + vec2<f32>(marginUv)
	);
}

@compute @workgroup_size(64)
fn csMain(@builtin(local_invocation_index) localIndex: u32) {
	let dirtyCount = min(
		min(atomicLoad(&counters[1]), params.dirtyCapacity),
		arrayLength(&dirtyGridIndices)
	);

	if (localIndex < DIRTY_GRID_CELL_COUNT) {
		atomicStore(&dirtyGridCounts[localIndex], 0u);
	}
	workgroupBarrier();
	storageBarrier();

	for (var i = localIndex; i < dirtyCount; i += 64u) {
		let dirtyBase = i * DIRTY_PHYSICAL_PAGE_RECORD_UINTS;
		if (dirtyBase + 7u < arrayLength(&dirtyPhysicalPages)) {
			let matrixIndex = dirtyPhysicalPages[dirtyBase + 2u];
			if (matrixIndex < 4u) {
				if (i < arrayLength(&dirtyPageUvRanges)) {
					dirtyPageUvRanges[i] = dirtyPageUvRange(dirtyBase);
				}
				atomicAdd(&dirtyGridCounts[dirtyGridCellIndex(dirtyBase)], 1u);
			}
		}
	}
	workgroupBarrier();
	storageBarrier();

	if (localIndex == 0u) {
		var sum = 0u;
		for (var cell = 0u; cell < DIRTY_GRID_CELL_COUNT; cell = cell + 1u) {
			dirtyGridOffsets[cell] = sum;
			sum = sum + atomicLoad(&dirtyGridCounts[cell]);
			atomicStore(&dirtyGridCounts[cell], 0u);
		}
		dirtyGridOffsets[DIRTY_GRID_CELL_COUNT] = sum;
	}
	workgroupBarrier();
	storageBarrier();

	for (var i = localIndex; i < dirtyCount; i += 64u) {
		let dirtyBase = i * DIRTY_PHYSICAL_PAGE_RECORD_UINTS;
		if (dirtyBase + 7u < arrayLength(&dirtyPhysicalPages)) {
			let matrixIndex = dirtyPhysicalPages[dirtyBase + 2u];
			if (matrixIndex < 4u) {
				let cellIndex = dirtyGridCellIndex(dirtyBase);
				let localOffset = atomicAdd(&dirtyGridCounts[cellIndex], 1u);
				let insertIndex = dirtyGridOffsets[cellIndex] + localOffset;
				if (insertIndex < arrayLength(&dirtyGridIndices)) {
					dirtyGridIndices[insertIndex] = i;
				}
			}
		}
	}
}
