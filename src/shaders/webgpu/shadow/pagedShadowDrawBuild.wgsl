const DIRTY_PHYSICAL_PAGE_RECORD_UINTS: u32 = 8u;
const DRAW_INDIRECT_UINTS: u32 = 5u;
const SHADOW_INSTANCE_DATA_UINTS: u32 = 12u;
const DIRTY_GRID_CELL_COUNT: u32 = 64u;
const DIRTY_GRID_CASCADE_CELLS: u32 = 16u;
const DIRTY_GRID_COARSE_SIZE: u32 = 4u;
const DIRTY_CELL_CACHE_SIZE: u32 = 64u;
const INVALID_DIRTY_INDEX: u32 = 0xffffffffu;
const PAGE_CLIP_Z_MIN: f32 = -30.0;
const PAGE_CLIP_Z_MAX: f32 = 10.0;

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

struct CasterBounds {
	centerRadius: vec4<f32>,
}

struct ShadowInstanceData {
	instanceBaseOffset: u32,
	vertexBaseOffset: u32,
	jointBaseOffset: u32,
	morphWeightBaseOffset: u32,
	morphDeltaBaseOffset: u32,
	atlasOffsetX: u32,
	atlasOffsetY: u32,
	atlasPageSize: u32,
	atlasSize: u32,
	flags: u32,
	_pad0: u32,
	_pad1: u32,
}

@group(0) @binding(0) var<uniform> params: PagedShadowDrawParams;
@group(0) @binding(1) var<storage, read> dirtyPhysicalPages: array<u32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> casterBounds: array<CasterBounds>;
@group(0) @binding(4) var<storage, read> drawWorldMatrices: array<mat4x4<f32>>;
@group(0) @binding(5) var<storage, read> cascadeViewProjections: array<mat4x4<f32>>;
@group(0) @binding(6) var<storage, read_write> drawMvps: array<mat4x4<f32>>;
@group(0) @binding(7) var<storage, read_write> drawInstanceMeta: array<ShadowInstanceData>;
@group(0) @binding(8) var<storage, read_write> drawTransmittance: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> drawIndirectArgs: array<u32>;
@group(0) @binding(10) var<storage, read> dirtyGridCounts: array<u32>;
@group(0) @binding(11) var<storage, read> dirtyGridOffsets: array<u32>;
@group(0) @binding(12) var<storage, read> dirtyGridIndices: array<u32>;
@group(0) @binding(13) var<storage, read> dirtyPageUvRanges: array<vec4<f32>>;

fn dirtyPageViewProjection(dirtyBase: u32) -> mat4x4<f32> {
	let matrixIndex = dirtyPhysicalPages[dirtyBase + 2u];
	let pageX = dirtyPhysicalPages[dirtyBase + 3u];
	let pageY = dirtyPhysicalPages[dirtyBase + 4u];
	let pageGridSize = max(dirtyPhysicalPages[dirtyBase + 7u], 1u);
	return cropMatrix(pageGridSize, pageX, pageY) *
		cascadeViewProjections[matrixIndex];
}

fn cropMatrix(grid: u32, pageX: u32, pageY: u32) -> mat4x4<f32> {
	let safeGrid = max(grid, 1u);
	let scale = f32(safeGrid);
	let offsetX = f32(safeGrid) - f32(pageX * 2u) - 1.0;
	let offsetY = f32(pageY * 2u) + 1.0 - f32(safeGrid);
	return mat4x4<f32>(
		vec4<f32>(scale, 0.0, 0.0, 0.0),
		vec4<f32>(0.0, scale, 0.0, 0.0),
		vec4<f32>(0.0, 0.0, 1.0, 0.0),
		vec4<f32>(offsetX, offsetY, 0.0, 1.0)
	);
}

fn depthRemapMatrix() -> mat4x4<f32> {
	return mat4x4<f32>(
		vec4<f32>(1.0, 0.0, 0.0, 0.0),
		vec4<f32>(0.0, 1.0, 0.0, 0.0),
		vec4<f32>(0.0, 0.0, 0.5, 0.0),
		vec4<f32>(0.0, 0.0, 0.5, 1.0)
	);
}

fn projectPoint(matrix: mat4x4<f32>, point: vec3<f32>) -> vec3<f32> {
	let clip = matrix * vec4<f32>(point, 1.0);
	if (abs(clip.w) <= 0.000001) {
		return vec3<f32>(2.0, 2.0, 2.0);
	}
	return clip.xyz / vec3<f32>(clip.w);
}

struct CascadeUVRange {
	minUv: vec2<f32>,
	maxUv: vec2<f32>,
	minCoarse: vec2<u32>,
	maxCoarse: vec2<u32>,
	hasProjected: bool,
}

struct CachedDirtyPage {
	dirtyIndex: u32,
	uvRange: vec4<f32>,
}

var<workgroup> g_cachedDirtyPages: array<CachedDirtyPage, DIRTY_CELL_CACHE_SIZE>;
var<workgroup> g_cachedCellStart: u32;
var<workgroup> g_cachedCellCount: u32;

fn invalidCascadeUVRange() -> CascadeUVRange {
	return CascadeUVRange(
		vec2<f32>(1.0),
		vec2<f32>(0.0),
		vec2<u32>(1u),
		vec2<u32>(0u),
		false
	);
}

fn coarseCellRange(minUv: vec2<f32>, maxUv: vec2<f32>) -> vec4<u32> {
	let coarseScale = f32(DIRTY_GRID_COARSE_SIZE);
	let coarseMax = f32(DIRTY_GRID_COARSE_SIZE - 1u);
	let minCoarse = vec2<u32>(clamp(
		floor(minUv * vec2<f32>(coarseScale)),
		vec2<f32>(0.0),
		vec2<f32>(coarseMax)
	));
	let maxCoarse = vec2<u32>(clamp(
		floor(maxUv * vec2<f32>(coarseScale)),
		vec2<f32>(0.0),
		vec2<f32>(coarseMax)
	));
	return vec4<u32>(minCoarse, maxCoarse);
}

fn projectedCascadeUVRange(minUv: vec2<f32>, maxUv: vec2<f32>) -> CascadeUVRange {
	let coarseRange = coarseCellRange(minUv, maxUv);
	return CascadeUVRange(
		minUv,
		maxUv,
		coarseRange.xy,
		coarseRange.zw,
		true
	);
}

fn projectSphereBoundsToUvRange(
	viewProjection: mat4x4<f32>,
	center: vec3<f32>,
	radius: f32
) -> CascadeUVRange {
	let safeRadius = max(radius, 0.0);
	let corners = array<vec3<f32>, 8>(
		center + vec3<f32>(-safeRadius, -safeRadius, -safeRadius),
		center + vec3<f32>( safeRadius, -safeRadius, -safeRadius),
		center + vec3<f32>(-safeRadius,  safeRadius, -safeRadius),
		center + vec3<f32>( safeRadius,  safeRadius, -safeRadius),
		center + vec3<f32>(-safeRadius, -safeRadius,  safeRadius),
		center + vec3<f32>( safeRadius, -safeRadius,  safeRadius),
		center + vec3<f32>(-safeRadius,  safeRadius,  safeRadius),
		center + vec3<f32>( safeRadius,  safeRadius,  safeRadius)
	);

	var outsideNear = true;
	var outsideFar = true;
	var minUv = vec2<f32>(1.0, 1.0);
	var maxUv = vec2<f32>(0.0, 0.0);
	for (var cornerIndex = 0u; cornerIndex < 8u; cornerIndex = cornerIndex + 1u) {
		let ndc = projectPoint(viewProjection, corners[cornerIndex]);
		if (ndc.z >= PAGE_CLIP_Z_MIN) {
			outsideNear = false;
		}
		if (ndc.z <= PAGE_CLIP_Z_MAX) {
			outsideFar = false;
		}
		let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
		minUv = min(minUv, uv);
		maxUv = max(maxUv, uv);
	}
	if (outsideNear || outsideFar) {
		return invalidCascadeUVRange();
	}
	return projectedCascadeUVRange(minUv, maxUv);
}

fn rangesIntersect(left: CascadeUVRange, right: CascadeUVRange) -> bool {
	if (!left.hasProjected || !right.hasProjected) {
		return false;
	}
	return !(
		left.minUv.x > right.maxUv.x || left.maxUv.x < right.minUv.x ||
		left.minUv.y > right.maxUv.y || left.maxUv.y < right.minUv.y
	);
}

fn rangeIntersectsCoarseCell(
	range: CascadeUVRange,
	cellX: u32,
	cellY: u32
) -> bool {
	if (!range.hasProjected) {
		return false;
	}
	return (
		cellX >= range.minCoarse.x && cellX <= range.maxCoarse.x &&
		cellY >= range.minCoarse.y && cellY <= range.maxCoarse.y
	);
}

// MAX_LOCAL_HITS is per-thread private memory (function-local array), so it
// does NOT count against workgroup storage. 64 entries allows large casters
// spanning up to 64 dirty pages without truncation.
const MAX_LOCAL_HITS: u32 = 64u;

@compute @workgroup_size(64)
fn csMain(
	@builtin(global_invocation_id) globalId: vec3<u32>,
	@builtin(local_invocation_index) localIndex: u32
) {
	let candidateIndex = globalId.x;
	let dirtyCount = min(
		min(atomicLoad(&counters[1]), params.dirtyCapacity),
		arrayLength(&dirtyGridIndices)
	);

	let argsBase = candidateIndex * DRAW_INDIRECT_UINTS;
	let hasArgs = (
		candidateIndex < params.candidateCount &&
		argsBase + 4u < arrayLength(&drawIndirectArgs)
	);
	if (hasArgs) {
		drawIndirectArgs[argsBase + 1u] = 0u;
		drawIndirectArgs[argsBase + 4u] = 0u;
	}

	let hasCandidateInput = (
		hasArgs &&
		dirtyCount > 0u &&
		candidateIndex < arrayLength(&casterBounds) &&
		candidateIndex < arrayLength(&drawWorldMatrices)
	);

	// 2. Precompute candidate UV ranges in each cascade (Optimization 2)
	var candidateRanges: array<CascadeUVRange, 4>;
	let cascadeCount = arrayLength(&cascadeViewProjections);
	for (var c = 0u; c < 4u; c = c + 1u) {
		if (hasCandidateInput && c < cascadeCount) {
			let bounds = casterBounds[candidateIndex].centerRadius;
			let center = bounds.xyz;
			let radius = max(bounds.w, 0.0);
			candidateRanges[c] = projectSphereBoundsToUvRange(cascadeViewProjections[c], center, radius);
		} else {
			candidateRanges[c] = invalidCascadeUVRange();
		}
	}

	// 3. Loop 1: Cache one coarse cell's dirty pages in workgroup memory, then
	// let all candidate lanes in this workgroup reuse the cached page records.
	var hitDirtyIndices: array<u32, MAX_LOCAL_HITS>;
	var intersectingPageCount = 0u;

	for (var cellIndex = 0u; cellIndex < DIRTY_GRID_CELL_COUNT; cellIndex = cellIndex + 1u) {
		if (localIndex == 0u) {
			let startIdx = min(dirtyGridOffsets[cellIndex], arrayLength(&dirtyGridIndices));
			let endIdx = min(
				min(dirtyGridOffsets[cellIndex + 1u], dirtyCount),
				arrayLength(&dirtyGridIndices)
			);
			let safeEndIdx = max(startIdx, endIdx);
			g_cachedCellStart = startIdx;
			g_cachedCellCount = min(dirtyGridCounts[cellIndex], safeEndIdx - startIdx);
		}
		workgroupBarrier();

		let cellStart = workgroupUniformLoad(&g_cachedCellStart);
		let cellEntryCount = workgroupUniformLoad(&g_cachedCellCount);
		let cascadeIndex = cellIndex / DIRTY_GRID_CASCADE_CELLS;
		let cellWithinCascade = cellIndex - cascadeIndex * DIRTY_GRID_CASCADE_CELLS;
		let cellY = cellWithinCascade / DIRTY_GRID_COARSE_SIZE;
		let cellX = cellWithinCascade - cellY * DIRTY_GRID_COARSE_SIZE;
		let range = candidateRanges[cascadeIndex];
		let candidateUsesCell = (
			hasCandidateInput &&
			rangeIntersectsCoarseCell(range, cellX, cellY)
		);

		for (
			var chunkStart = 0u;
			chunkStart < cellEntryCount;
			chunkStart = chunkStart + DIRTY_CELL_CACHE_SIZE
		) {
			let cacheOffset = chunkStart + localIndex;
			if (cacheOffset < cellEntryCount) {
				let dirtyGridIndex = cellStart + cacheOffset;
				let dirtyIndex = dirtyGridIndices[dirtyGridIndex];
				let dirtyBase = dirtyIndex * DIRTY_PHYSICAL_PAGE_RECORD_UINTS;
				if (
					dirtyIndex < dirtyCount &&
					dirtyIndex < arrayLength(&dirtyPageUvRanges) &&
					dirtyBase + 7u < arrayLength(&dirtyPhysicalPages)
				) {
					g_cachedDirtyPages[localIndex] = CachedDirtyPage(
						dirtyIndex,
						dirtyPageUvRanges[dirtyIndex]
					);
				} else {
					g_cachedDirtyPages[localIndex] = CachedDirtyPage(
						INVALID_DIRTY_INDEX,
						vec4<f32>(0.0)
					);
				}
			} else {
				g_cachedDirtyPages[localIndex] = CachedDirtyPage(
					INVALID_DIRTY_INDEX,
					vec4<f32>(0.0)
				);
			}
			workgroupBarrier();

			let chunkCount = min(DIRTY_CELL_CACHE_SIZE, cellEntryCount - chunkStart);
			if (candidateUsesCell) {
				for (var h = 0u; h < chunkCount; h = h + 1u) {
					let cachedPage = g_cachedDirtyPages[h];
					if (cachedPage.dirtyIndex != INVALID_DIRTY_INDEX) {
						let pageRange = CascadeUVRange(
							cachedPage.uvRange.xy,
							cachedPage.uvRange.zw,
							vec2<u32>(0u),
							vec2<u32>(0u),
							true
						);
						if (rangesIntersect(range, pageRange)) {
							if (intersectingPageCount < MAX_LOCAL_HITS) {
								hitDirtyIndices[intersectingPageCount] = cachedPage.dirtyIndex;
							}
							intersectingPageCount = intersectingPageCount + 1u;
						}
					}
				}
			}
			workgroupBarrier();
		}
	}

	if (!hasCandidateInput || intersectingPageCount == 0u) {
		return;
	}

	if (intersectingPageCount > MAX_LOCAL_HITS) {
		intersectingPageCount = MAX_LOCAL_HITS;
	}
	let firstInstance = atomicAdd(&counters[3], intersectingPageCount);
	drawIndirectArgs[argsBase + 4u] = firstInstance;

	let capacity = min(
		params.drawInstanceCapacity,
		min(arrayLength(&drawMvps), min(arrayLength(&drawInstanceMeta), arrayLength(&drawTransmittance)))
	);
	if (firstInstance >= capacity) {
		atomicAdd(&counters[4], intersectingPageCount);
		return;
	}

	let writableInstanceCount = min(intersectingPageCount, capacity - firstInstance);
	var localInstanceCount = 0u;
	let worldMatrix = drawWorldMatrices[candidateIndex];

	// 4. Loop 2: Directly write instances using Cached Hits (Optimization 3)
	for (var h = 0u; h < writableInstanceCount; h = h + 1u) {
		let dirtyIndex = hitDirtyIndices[h];
		let dirtyBase = dirtyIndex * DIRTY_PHYSICAL_PAGE_RECORD_UINTS;

		let pageViewProjection = dirtyPageViewProjection(dirtyBase);
		let viewportX = dirtyPhysicalPages[dirtyBase + 5u];
		let viewportY = dirtyPhysicalPages[dirtyBase + 6u];
		let instanceIndex = firstInstance + localInstanceCount;

		drawMvps[instanceIndex] = depthRemapMatrix() * pageViewProjection * worldMatrix;
		drawInstanceMeta[instanceIndex] = ShadowInstanceData(
			firstInstance,
			0u,
			0u,
			0u,
			0u,
			viewportX,
			viewportY,
			params.pageSize,
			params.physicalGridSize * params.pageSize,
			1u,
			0u,
			0u
		);
		drawTransmittance[instanceIndex] = vec4<f32>(1.0);
		localInstanceCount = localInstanceCount + 1u;
	}

	drawIndirectArgs[argsBase + 1u] = localInstanceCount;
	if (localInstanceCount < intersectingPageCount) {
		atomicAdd(&counters[4], intersectingPageCount - localInstanceCount);
	}
}
