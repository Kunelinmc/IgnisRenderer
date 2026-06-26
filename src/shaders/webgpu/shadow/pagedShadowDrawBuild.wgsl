const DIRTY_PHYSICAL_PAGE_RECORD_UINTS: u32 = 8u;
const DRAW_INDIRECT_UINTS: u32 = 5u;
const SHADOW_INSTANCE_DATA_UINTS: u32 = 12u;
const PAGE_CLIP_XY_MARGIN: f32 = 4.0;
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
	hasProjected: bool,
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
	for (var cornerIndex = 0u; cornerIndex < 8u; cornerIndex = cornerIndex + 1u) {
		let ndc = projectPoint(viewProjection, corners[cornerIndex]);
		if (ndc.z >= PAGE_CLIP_Z_MIN) {
			outsideNear = false;
		}
		if (ndc.z <= PAGE_CLIP_Z_MAX) {
			outsideFar = false;
		}
	}
	if (outsideNear || outsideFar) {
		return CascadeUVRange(vec2<f32>(1.0), vec2<f32>(0.0), false);
	}

	var minUv = vec2<f32>(1.0, 1.0);
	var maxUv = vec2<f32>(0.0, 0.0);
	for (var cornerIndex = 0u; cornerIndex < 8u; cornerIndex = cornerIndex + 1u) {
		let ndc = projectPoint(viewProjection, corners[cornerIndex]);
		let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
		minUv = min(minUv, uv);
		maxUv = max(maxUv, uv);
	}
	return CascadeUVRange(minUv, maxUv, true);
}

fn dirtyPageUvRange(pageX: u32, pageY: u32, pageGridSize: u32) -> CascadeUVRange {
	let gridSize = f32(max(pageGridSize, 1u));
	let px = f32(pageX);
	let py = f32(pageY);
	let atlasSize = max(gridSize * f32(max(params.pageSize, 1u)), 1.0);
	let marginUv = PAGE_CLIP_XY_MARGIN / atlasSize;
	return CascadeUVRange(
		vec2<f32>(px / gridSize, py / gridSize) - vec2<f32>(marginUv),
		vec2<f32>((px + 1.0) / gridSize, (py + 1.0) / gridSize) + vec2<f32>(marginUv),
		true
	);
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

var<workgroup> g_cellCounts: array<atomic<u32>, 64>;
var<workgroup> g_cellOffsets: array<u32, 65>;
var<workgroup> g_groupedDirtyIndices: array<u32, 2048>;

const MAX_LOCAL_HITS: u32 = 32u;

@compute @workgroup_size(64)
fn csMain(
	@builtin(global_invocation_id) globalId: vec3<u32>,
	@builtin(local_invocation_index) localIndex: u32
) {
	let candidateIndex = globalId.x;
	let dirtyCount = min(atomicLoad(&counters[1]), params.dirtyCapacity);

	// Cooperative Spatial Hash/Tile Grid building (all threads must participate in barriers)

	// 1. Cooperative building of Page Spatial Hash / Tile Grid in workgroup shared memory
	// Initialize cell counts
	if (localIndex < 64u) {
		atomicStore(&g_cellCounts[localIndex], 0u);
	}
	workgroupBarrier();

	// Count pages in each cell
	for (var i = localIndex; i < dirtyCount; i += 64u) {
		let dirtyBase = i * DIRTY_PHYSICAL_PAGE_RECORD_UINTS;
		if (dirtyBase + 7u < arrayLength(&dirtyPhysicalPages)) {
			let matrixIndex = dirtyPhysicalPages[dirtyBase + 2u];
			if (matrixIndex < 4u) {
				let pageX = dirtyPhysicalPages[dirtyBase + 3u];
				let pageY = dirtyPhysicalPages[dirtyBase + 4u];
				let pageGridSize = max(dirtyPhysicalPages[dirtyBase + 7u], 1u);
				let coarseX = min((pageX * 4u) / pageGridSize, 3u);
				let coarseY = min((pageY * 4u) / pageGridSize, 3u);
				let cellIndex = matrixIndex * 16u + coarseY * 4u + coarseX;
				atomicAdd(&g_cellCounts[cellIndex], 1u);
			}
		}
	}
	workgroupBarrier();

	// Compute prefix sums (offsets) of cell counts sequentially on thread 0
	if (localIndex == 0u) {
		var sum = 0u;
		for (var c = 0u; c < 64u; c = c + 1u) {
			g_cellOffsets[c] = sum;
			sum = sum + atomicLoad(&g_cellCounts[c]);
			atomicStore(&g_cellCounts[c], 0u); // Reset counts for tracking insertions
		}
		g_cellOffsets[64] = sum;
	}
	workgroupBarrier();

	// Insert dirty pages into grouped list
	for (var i = localIndex; i < dirtyCount; i += 64u) {
		let dirtyBase = i * DIRTY_PHYSICAL_PAGE_RECORD_UINTS;
		if (dirtyBase + 7u < arrayLength(&dirtyPhysicalPages)) {
			let matrixIndex = dirtyPhysicalPages[dirtyBase + 2u];
			if (matrixIndex < 4u) {
				let pageX = dirtyPhysicalPages[dirtyBase + 3u];
				let pageY = dirtyPhysicalPages[dirtyBase + 4u];
				let pageGridSize = max(dirtyPhysicalPages[dirtyBase + 7u], 1u);
				let coarseX = min((pageX * 4u) / pageGridSize, 3u);
				let coarseY = min((pageY * 4u) / pageGridSize, 3u);
				let cellIndex = matrixIndex * 16u + coarseY * 4u + coarseX;
				let localOffset = atomicAdd(&g_cellCounts[cellIndex], 1u);
				let insertIndex = g_cellOffsets[cellIndex] + localOffset;
				if (insertIndex < 2048u) {
					g_groupedDirtyIndices[insertIndex] = i;
				}
			}
		}
	}
	workgroupBarrier();

	// Now, after all workgroup barriers, threads that are out-of-bounds for candidates can safely exit
	if (candidateIndex >= params.candidateCount) {
		return;
	}

	let argsBase = candidateIndex * DRAW_INDIRECT_UINTS;
	if (argsBase + 4u >= arrayLength(&drawIndirectArgs)) {
		return;
	}
	drawIndirectArgs[argsBase + 1u] = 0u;
	drawIndirectArgs[argsBase + 4u] = 0u;

	if (
		dirtyCount == 0u ||
		candidateIndex >= arrayLength(&casterBounds) ||
		candidateIndex >= arrayLength(&drawWorldMatrices)
	) {
		return;
	}

	let bounds = casterBounds[candidateIndex].centerRadius;
	let worldMatrix = drawWorldMatrices[candidateIndex];

	let center = bounds.xyz;
	let radius = max(bounds.w, 0.0);

	// 2. Precompute candidate UV ranges in each cascade (Optimization 2)
	var candidateRanges: array<CascadeUVRange, 4>;
	let cascadeCount = arrayLength(&cascadeViewProjections);
	for (var c = 0u; c < 4u; c = c + 1u) {
		if (c < cascadeCount) {
			candidateRanges[c] = projectSphereBoundsToUvRange(cascadeViewProjections[c], center, radius);
		} else {
			candidateRanges[c] = CascadeUVRange(vec2<f32>(1.0), vec2<f32>(0.0), false);
		}
	}

	// 3. Loop 1: Find intersecting pages using Page Spatial Hash/Tile Grid (Optimizations 1, 4)
	var hitDirtyIndices: array<u32, MAX_LOCAL_HITS>;
	var intersectingPageCount = 0u;

	for (var c = 0u; c < 4u; c = c + 1u) {
		let range = candidateRanges[c];
		if (!range.hasProjected) {
			continue;
		}

		// Map UV range to overlapping coarse cells
		let minCoarseX = u32(clamp(floor(range.minUv.x * 4.0), 0.0, 3.0));
		let maxCoarseX = u32(clamp(floor(range.maxUv.x * 4.0), 0.0, 3.0));
		let minCoarseY = u32(clamp(floor(range.minUv.y * 4.0), 0.0, 3.0));
		let maxCoarseY = u32(clamp(floor(range.maxUv.y * 4.0), 0.0, 3.0));

		for (var cy = minCoarseY; cy <= maxCoarseY; cy = cy + 1u) {
			for (var cx = minCoarseX; cx <= maxCoarseX; cx = cx + 1u) {
				let cellIndex = c * 16u + cy * 4u + cx;
				let startIdx = g_cellOffsets[cellIndex];
				let endIdx = g_cellOffsets[cellIndex + 1u];

				for (var idx = startIdx; idx < endIdx; idx = idx + 1u) {
					let dirtyIndex = g_groupedDirtyIndices[idx];
					let dirtyBase = dirtyIndex * DIRTY_PHYSICAL_PAGE_RECORD_UINTS;

					let pageX = dirtyPhysicalPages[dirtyBase + 3u];
					let pageY = dirtyPhysicalPages[dirtyBase + 4u];
					let pageGridSize = max(dirtyPhysicalPages[dirtyBase + 7u], 1u);
					let pageRange = dirtyPageUvRange(pageX, pageY, pageGridSize);

					if (rangesIntersect(range, pageRange)) {
						if (intersectingPageCount < MAX_LOCAL_HITS) {
							hitDirtyIndices[intersectingPageCount] = dirtyIndex;
						}
						intersectingPageCount = intersectingPageCount + 1u;
					}
				}
			}
		}
	}

	if (intersectingPageCount == 0u) {
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
