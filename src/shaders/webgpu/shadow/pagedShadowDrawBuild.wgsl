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
	let cascadeIndex = dirtyPhysicalPages[dirtyBase + 2u];
	let pageX = dirtyPhysicalPages[dirtyBase + 3u];
	let pageY = dirtyPhysicalPages[dirtyBase + 4u];
	let pageGridSize = max(dirtyPhysicalPages[dirtyBase + 7u], 1u);
	return cropMatrix(pageGridSize, pageX, pageY) *
		cascadeViewProjections[cascadeIndex];
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

@compute @workgroup_size(64)
fn csMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
	let candidateIndex = globalId.x;
	if (candidateIndex >= params.candidateCount) {
		return;
	}
	let argsBase = candidateIndex * DRAW_INDIRECT_UINTS;
	if (argsBase + 4u >= arrayLength(&drawIndirectArgs)) {
		return;
	}
	drawIndirectArgs[argsBase + 1u] = 0u;
	drawIndirectArgs[argsBase + 4u] = 0u;

	let dirtyCount = min(atomicLoad(&counters[1]), params.dirtyCapacity);
	if (dirtyCount == 0u || candidateIndex >= arrayLength(&casterBounds)) {
		return;
	}
	let bounds = casterBounds[candidateIndex].centerRadius;
	let worldMatrix = drawWorldMatrices[candidateIndex];

	let center = bounds.xyz;
	let radius = max(bounds.w, 0.0);
	let corners = array<vec3<f32>, 8>(
		center + vec3<f32>(-radius, -radius, -radius),
		center + vec3<f32>( radius, -radius, -radius),
		center + vec3<f32>(-radius,  radius, -radius),
		center + vec3<f32>( radius,  radius, -radius),
		center + vec3<f32>(-radius, -radius,  radius),
		center + vec3<f32>( radius, -radius,  radius),
		center + vec3<f32>(-radius,  radius,  radius),
		center + vec3<f32>( radius,  radius,  radius)
	);

	var localRanges: array<CascadeUVRange, 4>;
	let maxCascades = min(4u, arrayLength(&cascadeViewProjections));
	for (var c = 0u; c < maxCascades; c = c + 1u) {
		let viewProjection = cascadeViewProjections[c];
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
			localRanges[c] = CascadeUVRange(vec2<f32>(1.0), vec2<f32>(0.0), false);
			continue;
		}

		var minUv = vec2<f32>(1.0, 1.0);
		var maxUv = vec2<f32>(0.0, 0.0);
		for (var cornerIndex = 0u; cornerIndex < 8u; cornerIndex = cornerIndex + 1u) {
			let ndc = projectPoint(viewProjection, corners[cornerIndex]);
			let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
			minUv = min(minUv, uv);
			maxUv = max(maxUv, uv);
		}
		localRanges[c] = CascadeUVRange(minUv, maxUv, true);
	}

	var intersectingPageCount = 0u;
	for (var dirtyIndex = 0u; dirtyIndex < dirtyCount; dirtyIndex = dirtyIndex + 1u) {
		let dirtyBase = dirtyIndex * DIRTY_PHYSICAL_PAGE_RECORD_UINTS;
		if (dirtyBase + 7u >= arrayLength(&dirtyPhysicalPages)) {
			break;
		}
		let cascadeIndex = dirtyPhysicalPages[dirtyBase + 2u];
		if (cascadeIndex >= maxCascades) {
			continue;
		}
		let range = localRanges[cascadeIndex];
		if (!range.hasProjected) {
			continue;
		}
		let pageX = dirtyPhysicalPages[dirtyBase + 3u];
		let pageY = dirtyPhysicalPages[dirtyBase + 4u];
		let pageGridSize = max(dirtyPhysicalPages[dirtyBase + 7u], 1u);

		let gridSize = f32(pageGridSize);
		let px = f32(pageX);
		let py = f32(pageY);

		let pageMinUv = vec2<f32>((px - 1.5) / gridSize, (py - 1.5) / gridSize);
		let pageMaxUv = vec2<f32>((px + 2.5) / gridSize, (py + 2.5) / gridSize);

		if (range.minUv.x > pageMaxUv.x || range.maxUv.x < pageMinUv.x ||
			range.minUv.y > pageMaxUv.y || range.maxUv.y < pageMinUv.y) {
			continue;
		}
		intersectingPageCount = intersectingPageCount + 1u;
	}
	if (intersectingPageCount == 0u) {
		return;
	}
	let firstInstance = atomicAdd(&counters[3], intersectingPageCount);
	drawIndirectArgs[argsBase + 4u] = firstInstance;
	let capacity = min(params.drawInstanceCapacity, arrayLength(&drawMvps));
	if (firstInstance >= capacity) {
		atomicAdd(&counters[4], intersectingPageCount);
		return;
	}
	let writableInstanceCount = min(intersectingPageCount, capacity - firstInstance);
	var localInstanceCount = 0u;
	for (var dirtyIndex = 0u; dirtyIndex < dirtyCount; dirtyIndex = dirtyIndex + 1u) {
		if (localInstanceCount >= writableInstanceCount) {
			break;
		}
		let dirtyBase = dirtyIndex * DIRTY_PHYSICAL_PAGE_RECORD_UINTS;
		if (dirtyBase + 7u >= arrayLength(&dirtyPhysicalPages)) {
			break;
		}
		let cascadeIndex = dirtyPhysicalPages[dirtyBase + 2u];
		if (cascadeIndex >= maxCascades) {
			continue;
		}
		let range = localRanges[cascadeIndex];
		if (!range.hasProjected) {
			continue;
		}
		let pageX = dirtyPhysicalPages[dirtyBase + 3u];
		let pageY = dirtyPhysicalPages[dirtyBase + 4u];
		let pageGridSize = max(dirtyPhysicalPages[dirtyBase + 7u], 1u);

		let gridSize = f32(pageGridSize);
		let px = f32(pageX);
		let py = f32(pageY);

		let pageMinUv = vec2<f32>((px - 1.5) / gridSize, (py - 1.5) / gridSize);
		let pageMaxUv = vec2<f32>((px + 2.5) / gridSize, (py + 2.5) / gridSize);

		if (range.minUv.x > pageMaxUv.x || range.maxUv.x < pageMinUv.x ||
			range.minUv.y > pageMaxUv.y || range.maxUv.y < pageMinUv.y) {
			continue;
		}
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
