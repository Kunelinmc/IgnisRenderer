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

fn intersectsPage(bounds: vec4<f32>, viewProjection: mat4x4<f32>) -> bool {
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
	// Reject only when every AABB corner is outside the same loose clip plane.
	var outsideLeft = true;
	var outsideRight = true;
	var outsideBottom = true;
	var outsideTop = true;
	var outsideNear = true;
	var outsideFar = true;
	for (var index = 0u; index < 8u; index = index + 1u) {
		let clip = viewProjection * vec4<f32>(corners[index], 1.0);
		outsideLeft = outsideLeft && clip.x < -PAGE_CLIP_XY_MARGIN * clip.w;
		outsideRight = outsideRight && clip.x > PAGE_CLIP_XY_MARGIN * clip.w;
		outsideBottom = outsideBottom && clip.y < -PAGE_CLIP_XY_MARGIN * clip.w;
		outsideTop = outsideTop && clip.y > PAGE_CLIP_XY_MARGIN * clip.w;
		outsideNear = outsideNear && clip.z < PAGE_CLIP_Z_MIN * clip.w;
		outsideFar = outsideFar && clip.z > PAGE_CLIP_Z_MAX * clip.w;
	}
	return !(
		outsideLeft ||
		outsideRight ||
		outsideBottom ||
		outsideTop ||
		outsideNear ||
		outsideFar
	);
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
	let firstInstance = candidateIndex * params.physicalPageCount;
	drawIndirectArgs[argsBase + 1u] = 0u;
	drawIndirectArgs[argsBase + 4u] = firstInstance;

	let dirtyCount = min(atomicLoad(&counters[1]), params.dirtyCapacity);
	if (dirtyCount == 0u || candidateIndex >= arrayLength(&casterBounds)) {
		return;
	}
	let bounds = casterBounds[candidateIndex].centerRadius;
	let worldMatrix = drawWorldMatrices[candidateIndex];
	var localInstanceCount = 0u;
	for (var dirtyIndex = 0u; dirtyIndex < dirtyCount; dirtyIndex = dirtyIndex + 1u) {
		let dirtyBase = dirtyIndex * DIRTY_PHYSICAL_PAGE_RECORD_UINTS;
		if (dirtyBase + 7u >= arrayLength(&dirtyPhysicalPages)) {
			break;
		}
		let cascadeIndex = dirtyPhysicalPages[dirtyBase + 2u];
		if (cascadeIndex >= arrayLength(&cascadeViewProjections)) {
			continue;
		}
		let pageX = dirtyPhysicalPages[dirtyBase + 3u];
		let pageY = dirtyPhysicalPages[dirtyBase + 4u];
		let viewportX = dirtyPhysicalPages[dirtyBase + 5u];
		let viewportY = dirtyPhysicalPages[dirtyBase + 6u];
		let pageGridSize = max(dirtyPhysicalPages[dirtyBase + 7u], 1u);
		let pageViewProjection =
			cropMatrix(pageGridSize, pageX, pageY) *
			cascadeViewProjections[cascadeIndex];
		if (!intersectsPage(bounds, pageViewProjection)) {
			continue;
		}
		let instanceIndex = firstInstance + localInstanceCount;
		if (instanceIndex >= params.drawInstanceCapacity || instanceIndex >= arrayLength(&drawMvps)) {
			break;
		}
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
}
