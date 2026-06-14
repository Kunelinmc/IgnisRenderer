#import <ignis/webgpu/constants>
struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
	environmentBasisRight: vec4<f32>,
	environmentBasisUp: vec4<f32>,
	environmentBasisBackward: vec4<f32>,
	ambientColor: vec4<f32>,
	lightCounts: vec4<f32>,
	options: vec4<f32>,
	environmentOptionsA: vec4<f32>,
	environmentOptionsB: vec4<f32>,
	taaJitterCurrentPrev: vec4<f32>,
	directionalLights: array<DirectionalLightData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	pointLights: array<PointLightData, __WEBGPU_MAX_POINT_LIGHTS__>,
	spotLights: array<SpotLightData, __WEBGPU_MAX_SPOT_LIGHTS__>,
	directionalShadows: array<ShadowData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	spotShadows: array<ShadowData, __WEBGPU_MAX_SPOT_LIGHTS__>,
	shAmbientCoeffs: array<vec4<f32>, __WEBGPU_SH_COEFFICIENT_COUNT__>,
}

struct ClusterGridParams {
	screenWidth: u32,
	screenHeight: u32,
	tilesX: u32,
	tilesY: u32,
	zSlices: u32,
	clusterCount: u32,
	near: f32,
	far: f32,
	logScale: f32,
	logBias: f32,
	lightCount: u32,
	maxLightsPerCluster: u32,
}

struct ClusterCullData {
	positionCullRadius: vec4<f32>,
	scoreParams: vec4<f32>,
}

struct ClusterAreaPayload {
	rightWidth: vec4<f32>,
	upHeight: vec4<f32>,
	normalAreaScale: vec4<f32>,
}

struct ClusterMetadata {
	packedFlags: u32,
	shadowIndex: u32,
}

struct ClusterHeader {
	offset: u32,
	count: atomic<u32>,
	flags: atomic<u32>,
	reserved: u32,
}

struct ClusterCullBuffer {
	values: array<ClusterCullData>,
}

struct ClusterVec4Buffer {
	values: array<vec4<f32>>,
}

struct ClusterAreaPayloadBuffer {
	values: array<ClusterAreaPayload>,
}

struct ClusterMetadataBuffer {
	values: array<ClusterMetadata>,
}

struct ClusterHeaderBuffer {
	headers: array<ClusterHeader>,
}

struct ClusterLightIndexList {
	indices: array<u32>,
}

struct ClusterSliceDepthBuffer {
	depths: array<f32>,
}

struct ClusterAABB {
	minX: f32,
	maxX: f32,
	minY: f32,
	maxY: f32,
	zNear: f32,
	zFar: f32,
}

struct LightClusterRange {
	tileXMin: u32,
	tileXMax: u32,
	tileYMin: u32,
	tileYMax: u32,
	sliceMin: u32,
	sliceMax: u32,
	valid: bool,
}

const CLUSTER_LIGHT_TYPE_POINT: u32 = 0u;
const CLUSTER_LIGHT_TYPE_SPOT: u32 = 1u;
const CLUSTER_LIGHT_TYPE_AREA: u32 = 2u;
const CLUSTER_LIGHT_TYPE_MASK: u32 = 0x3u;
const CLUSTER_LIGHT_FLAG_CASTS_SHADOW: u32 = 1u << 2u;
const CLUSTER_LIGHT_FLAG_AFFECTS_VOLUMETRIC: u32 = 1u << 3u;
const CLUSTER_LIGHT_INDEX_MASK: u32 = 0x00ffffffu;
const CLUSTER_LIGHT_INDEX_TYPE_SHIFT: u32 = 24u;
const CLUSTER_LIGHT_INDEX_SHADOW_BIT: u32 = 1u << 26u;
const CLUSTER_LIGHT_INDEX_VOLUMETRIC_BIT: u32 = 1u << 27u;
const CLUSTER_HEADER_FLAG_OVERFLOW: u32 = 1u << 0u;
const CLUSTER_HEADER_FLAG_HAS_SHADOWED: u32 = 1u << 1u;
const CLUSTER_HEADER_FLAG_HAS_VOLUMETRIC: u32 = 1u << 2u;
const CLUSTERED_LIGHT_WORKGROUP_SIZE: u32 = 128u;
const CLUSTERED_LIGHT_MAX_LIGHTS: u32 = 1024u;

@group(0) @binding(0) var<uniform> clusterParams: ClusterGridParams;
@group(0) @binding(1) var<storage, read> clusterCullData: ClusterCullBuffer;
@group(0) @binding(2) var<storage, read> clusterDirectionOuters: ClusterVec4Buffer;
@group(0) @binding(3) var<storage, read> clusterAreaPayloads: ClusterAreaPayloadBuffer;
@group(0) @binding(4) var<storage, read> clusterMetadata: ClusterMetadataBuffer;
@group(0) @binding(5) var<storage, read_write> clusterHeaders: ClusterHeaderBuffer;
@group(0) @binding(6) var<storage, read_write> clusterIndices: ClusterLightIndexList;
@group(0) @binding(7) var<storage, read> clusterSliceDepths: ClusterSliceDepthBuffer;

@group(1) @binding(0) var<uniform> frame: FrameUniforms;

var<workgroup> sharedAABB: ClusterAABB;
var<workgroup> gatherScanA: array<u32, 128>;
var<workgroup> gatherScanB: array<u32, 128>;
var<workgroup> gatherBaseCount: u32;
var<workgroup> gatherSelectedFlags: atomic<u32>;
var<workgroup> overflowScores: array<f32, 1024>;
var<workgroup> overflowRefs: array<u32, 1024>;
var<workgroup> overflowMatchCount: atomic<u32>;
var<workgroup> overflowSelectedFlags: atomic<u32>;

fn clusteredEnabled() -> bool {
	return frame.environmentOptionsB.w > 0.5 &&
		clusterParams.logScale > 0.0 &&
		clusterParams.clusterCount > 0u &&
		clusterParams.zSlices > 0u;
}

fn activeClusterLightCount() -> u32 {
	return min(
		min(clusterParams.lightCount, arrayLength(&clusterCullData.values)),
		CLUSTERED_LIGHT_MAX_LIGHTS
	);
}

fn clusterMaxLightsPerCluster() -> u32 {
	let clusterTotal = max(clusterParams.clusterCount, 1u);
	let bufferSpan = max(arrayLength(&clusterIndices.indices) / clusterTotal, 1u);
	return min(min(max(clusterParams.maxLightsPerCluster, 1u), bufferSpan), 128u);
}

fn worldToView(worldPos: vec3<f32>) -> vec3<f32> {
	let rel = worldPos - frame.cameraPosition.xyz;
	let viewX = dot(rel, frame.environmentBasisRight.xyz);
	let viewY = dot(rel, frame.environmentBasisUp.xyz);
	let depth = dot(frame.cameraPosition.xyz - worldPos, frame.environmentBasisBackward.xyz);
	return vec3<f32>(viewX, viewY, depth);
}

fn dirToView(dir: vec3<f32>) -> vec3<f32> {
	return vec3<f32>(
		dot(dir, frame.environmentBasisRight.xyz),
		dot(dir, frame.environmentBasisUp.xyz),
		-dot(dir, frame.environmentBasisBackward.xyz)
	);
}

fn sliceDepthBoundary(slice: u32) -> f32 {
	let lastIndex = max(arrayLength(&clusterSliceDepths.depths), 1u) - 1u;
	return clusterSliceDepths.depths[min(slice, lastIndex)];
}

fn depthToSlice(depth: f32) -> u32 {
	let z = clamp(depth, clusterParams.near, clusterParams.far);
	let slice = i32(floor(log(z) * clusterParams.logScale + clusterParams.logBias));
	return u32(clamp(slice, 0, i32(max(clusterParams.zSlices, 1u)) - 1));
}

fn clampToGridIndex(value: f32, maxExclusive: u32) -> u32 {
	let safeMax = max(maxExclusive, 1u);
	return u32(clamp(i32(floor(value)), 0, i32(safeMax) - 1));
}

fn invalidLightClusterRange() -> LightClusterRange {
	return LightClusterRange(0u, 0u, 0u, 0u, 0u, 0u, false);
}

fn resolveLightClusterRange(viewPos: vec3<f32>, range: f32) -> LightClusterRange {
	let viewDepth = viewPos.z;
	if (viewDepth + range < clusterParams.near ||
		viewDepth - range > clusterParams.far) {
		return invalidLightClusterRange();
	}

	let tilesX = max(clusterParams.tilesX, 1u);
	let tilesY = max(clusterParams.tilesY, 1u);
	let projectedDepth = max(viewDepth, clusterParams.near);
	let tanHalfFov = max(frame.environmentBasisRight.w, 1e-6);
	let aspect = max(frame.environmentBasisUp.w, 1e-6);
	let ndcX = clamp(
		viewPos.x / max(projectedDepth * tanHalfFov * aspect, 1e-6),
		-1.5,
		1.5
	);
	let ndcY = clamp(
		viewPos.y / max(projectedDepth * tanHalfFov, 1e-6),
		-1.5,
		1.5
	);
	let radiusNdcY = clamp(range / max(projectedDepth * tanHalfFov, 1e-6), 0.0, 2.0);
	let radiusNdcX = clamp(radiusNdcY / aspect, 0.0, 2.0);
	let tileXMin = clampToGridIndex(
		(clamp(ndcX - radiusNdcX, -1.0, 1.0) * 0.5 + 0.5) * f32(tilesX),
		tilesX
	);
	let tileXMax = clampToGridIndex(
		(clamp(ndcX + radiusNdcX, -1.0, 1.0) * 0.5 + 0.5) * f32(tilesX),
		tilesX
	);
	let tileYMin = clampToGridIndex(
		(0.5 - clamp(ndcY + radiusNdcY, -1.0, 1.0) * 0.5) * f32(tilesY),
		tilesY
	);
	let tileYMax = clampToGridIndex(
		(0.5 - clamp(ndcY - radiusNdcY, -1.0, 1.0) * 0.5) * f32(tilesY),
		tilesY
	);
	let zMin = clamp(max(clusterParams.near, viewDepth - range), clusterParams.near, clusterParams.far);
	let zMax = clamp(min(clusterParams.far, viewDepth + range), clusterParams.near, clusterParams.far);
	if (zMax < zMin) {
		return invalidLightClusterRange();
	}
	return LightClusterRange(
		min(tileXMin, tileXMax),
		max(tileXMin, tileXMax),
		min(tileYMin, tileYMax),
		max(tileYMin, tileYMax),
		depthToSlice(zMin),
		depthToSlice(zMax),
		true
	);
}

fn buildClusterAABB(clusterX: u32, clusterY: u32, clusterZ: u32) -> ClusterAABB {
	let safeTilesX = max(clusterParams.tilesX, 1u);
	let safeTilesY = max(clusterParams.tilesY, 1u);
	let invTilesX = 1.0 / f32(safeTilesX);
	let invTilesY = 1.0 / f32(safeTilesY);
	let ndcLeftX = f32(clusterX) * invTilesX * 2.0 - 1.0;
	let ndcRightX = f32(clusterX + 1u) * invTilesX * 2.0 - 1.0;
	let ndcTopY = 1.0 - f32(clusterY) * invTilesY * 2.0;
	let ndcBottomY = 1.0 - f32(clusterY + 1u) * invTilesY * 2.0;
	let zNear = sliceDepthBoundary(clusterZ);
	let zFar = sliceDepthBoundary(clusterZ + 1u);
	let tanHalfFov = max(frame.environmentBasisRight.w, 1e-6);
	let aspect = max(frame.environmentBasisUp.w, 1e-6);
	let nearScaleX = zNear * tanHalfFov * aspect;
	let nearScaleY = zNear * tanHalfFov;
	let farScaleX = zFar * tanHalfFov * aspect;
	let farScaleY = zFar * tanHalfFov;
	let x0n = min(ndcLeftX, ndcRightX) * nearScaleX;
	let x1n = max(ndcLeftX, ndcRightX) * nearScaleX;
	let x0f = min(ndcLeftX, ndcRightX) * farScaleX;
	let x1f = max(ndcLeftX, ndcRightX) * farScaleX;
	let y0n = min(ndcBottomY, ndcTopY) * nearScaleY;
	let y1n = max(ndcBottomY, ndcTopY) * nearScaleY;
	let y0f = min(ndcBottomY, ndcTopY) * farScaleY;
	let y1f = max(ndcBottomY, ndcTopY) * farScaleY;
	return ClusterAABB(
		min(x0n, x0f),
		max(x1n, x1f),
		min(y0n, y0f),
		max(y1n, y1f),
		zNear,
		zFar
	);
}

fn clusterAABBCenter(aabb: ClusterAABB) -> vec3<f32> {
	return vec3<f32>(
		(aabb.minX + aabb.maxX) * 0.5,
		(aabb.minY + aabb.maxY) * 0.5,
		(aabb.zNear + aabb.zFar) * 0.5
	);
}

fn sphereIntersectsAABB(center: vec3<f32>, radius: f32, aabb: ClusterAABB) -> bool {
	var distSq = 0.0;
	if (center.x < aabb.minX) {
		let d = aabb.minX - center.x;
		distSq += d * d;
	} else if (center.x > aabb.maxX) {
		let d = center.x - aabb.maxX;
		distSq += d * d;
	}
	if (center.y < aabb.minY) {
		let d = aabb.minY - center.y;
		distSq += d * d;
	} else if (center.y > aabb.maxY) {
		let d = center.y - aabb.maxY;
		distSq += d * d;
	}
	if (center.z < aabb.zNear) {
		let d = aabb.zNear - center.z;
		distSq += d * d;
	} else if (center.z > aabb.zFar) {
		let d = center.z - aabb.zFar;
		distSq += d * d;
	}
	return distSq <= radius * radius;
}

fn spotConeIntersectsAABB(
	lightView: vec3<f32>,
	lightDirView: vec3<f32>,
	outerCos: f32,
	aabb: ClusterAABB
) -> bool {
	let nearest = vec3<f32>(
		clamp(lightView.x, aabb.minX, aabb.maxX),
		clamp(lightView.y, aabb.minY, aabb.maxY),
		clamp(lightView.z, aabb.zNear, aabb.zFar)
	);
	let toNearest = nearest - lightView;
	let distNearest = length(toNearest);
	if (distNearest < 1e-6) {
		return true;
	}
	let halfExtent = vec3<f32>(
		(aabb.maxX - aabb.minX) * 0.5,
		(aabb.maxY - aabb.minY) * 0.5,
		(aabb.zFar - aabb.zNear) * 0.5
	);
	let angularSlack = length(halfExtent) / max(distNearest, 1e-6);
	return dot(toNearest / distNearest, lightDirView) >= outerCos - angularSlack;
}

fn lightIntersectsCluster(lightIndex: u32, aabb: ClusterAABB) -> bool {
	if (lightIndex >= activeClusterLightCount()) {
		return false;
	}
	let cullData = clusterCullData.values[lightIndex];
	let viewPos = worldToView(cullData.positionCullRadius.xyz);
	let radius = max(cullData.positionCullRadius.w, 0.001);
	if (!sphereIntersectsAABB(viewPos, radius, aabb)) {
		return false;
	}
	let metadata = clusterMetadata.values[lightIndex];
	let lightType = metadata.packedFlags & CLUSTER_LIGHT_TYPE_MASK;
	if (lightType == CLUSTER_LIGHT_TYPE_SPOT) {
		let directionOuter = clusterDirectionOuters.values[lightIndex];
		return spotConeIntersectsAABB(
			viewPos,
			dirToView(normalize(directionOuter.xyz)),
			directionOuter.w,
			aabb
		);
	}
	return lightType <= CLUSTER_LIGHT_TYPE_AREA;
}

fn packClusteredLightRef(lightIndex: u32, packedFlags: u32) -> u32 {
	let lightType = packedFlags & CLUSTER_LIGHT_TYPE_MASK;
	var packedRef = (lightIndex & CLUSTER_LIGHT_INDEX_MASK) |
		((lightType & CLUSTER_LIGHT_TYPE_MASK) << CLUSTER_LIGHT_INDEX_TYPE_SHIFT);
	if ((packedFlags & CLUSTER_LIGHT_FLAG_CASTS_SHADOW) != 0u) {
		packedRef |= CLUSTER_LIGHT_INDEX_SHADOW_BIT;
	}
	if ((packedFlags & CLUSTER_LIGHT_FLAG_AFFECTS_VOLUMETRIC) != 0u) {
		packedRef |= CLUSTER_LIGHT_INDEX_VOLUMETRIC_BIT;
	}
	return packedRef;
}

fn headerFlagsForLight(packedFlags: u32) -> u32 {
	var flags = 0u;
	if ((packedFlags & CLUSTER_LIGHT_FLAG_CASTS_SHADOW) != 0u) {
		flags |= CLUSTER_HEADER_FLAG_HAS_SHADOWED;
	}
	if ((packedFlags & CLUSTER_LIGHT_FLAG_AFFECTS_VOLUMETRIC) != 0u) {
		flags |= CLUSTER_HEADER_FLAG_HAS_VOLUMETRIC;
	}
	return flags;
}

fn appendLightToCluster(clusterIndex: u32, lightIndex: u32, packedFlags: u32) {
	let maxLightsPerCluster = clusterMaxLightsPerCluster();
	let previousCount = atomicAdd(&clusterHeaders.headers[clusterIndex].count, 1u);
	if (previousCount >= maxLightsPerCluster) {
		atomicOr(&clusterHeaders.headers[clusterIndex].flags, CLUSTER_HEADER_FLAG_OVERFLOW);
		return;
	}
	let offset = clusterHeaders.headers[clusterIndex].offset + previousCount;
	if (offset >= arrayLength(&clusterIndices.indices)) {
		atomicOr(&clusterHeaders.headers[clusterIndex].flags, CLUSTER_HEADER_FLAG_OVERFLOW);
		return;
	}
	clusterIndices.indices[offset] = packClusteredLightRef(lightIndex, packedFlags);
	let flags = headerFlagsForLight(packedFlags);
	if (flags != 0u) {
		atomicOr(&clusterHeaders.headers[clusterIndex].flags, flags);
	}
}

fn pointRangeAttenuation(distance: f32, range: f32) -> f32 {
	let normalized = clamp(1.0 - distance / max(range, 0.001), 0.0, 1.0);
	return normalized * normalized / max(distance * distance, 0.01);
}

fn estimateLightContribution(lightIndex: u32, aabb: ClusterAABB) -> f32 {
	let cullData = clusterCullData.values[lightIndex];
	let metadata = clusterMetadata.values[lightIndex];
	let lightType = metadata.packedFlags & CLUSTER_LIGHT_TYPE_MASK;
	let lightView = worldToView(cullData.positionCullRadius.xyz);
	let clusterCenter = clusterAABBCenter(aabb);
	let range = max(cullData.scoreParams.x, 0.001);
	let luminance = max(cullData.scoreParams.y, 0.0);
	if (lightType == CLUSTER_LIGHT_TYPE_AREA) {
		let payload = clusterAreaPayloads.values[lightIndex];
		let right = normalize(dirToView(payload.rightWidth.xyz));
		let up = normalize(dirToView(payload.upHeight.xyz));
		let normal = normalize(dirToView(payload.normalAreaScale.xyz));
		let toCenter = clusterCenter - lightView;
		let closest = lightView +
			right * clamp(dot(toCenter, right), -payload.rightWidth.w * 0.5, payload.rightWidth.w * 0.5) +
			up * clamp(dot(toCenter, up), -payload.upHeight.w * 0.5, payload.upHeight.w * 0.5);
		let toCluster = clusterCenter - closest;
		let distance = length(toCluster);
		let facing = max(dot(normal, toCluster / max(distance, 1e-6)), 0.0);
		return luminance * max(payload.normalAreaScale.w, 0.0) * facing *
			pointRangeAttenuation(distance, range);
	}

	let toCluster = clusterCenter - lightView;
	let distance = length(toCluster);
	var attenuation = pointRangeAttenuation(distance, range);
	if (lightType == CLUSTER_LIGHT_TYPE_SPOT) {
		let directionOuter = clusterDirectionOuters.values[lightIndex];
		let cosTheta = dot(normalize(toCluster), dirToView(normalize(directionOuter.xyz)));
		let innerCos = cullData.scoreParams.w;
		attenuation *= smoothstep(directionOuter.w, max(innerCos, directionOuter.w + 1e-5), cosTheta);
	}
	return luminance * attenuation;
}

fn scoreIsBetter(scoreA: f32, refA: u32, scoreB: f32, refB: u32) -> bool {
	let lightIndexA = refA & CLUSTER_LIGHT_INDEX_MASK;
	let lightIndexB = refB & CLUSTER_LIGHT_INDEX_MASK;
	return scoreA > scoreB ||
		(scoreA == scoreB && lightIndexA < lightIndexB);
}

@compute @workgroup_size(128, 1, 1)
fn csClear(@builtin(global_invocation_id) globalId: vec3<u32>) {
	let clusterIndex = globalId.x;
	if (clusterIndex >= clusterParams.clusterCount) {
		return;
	}
	let maxLightsPerCluster = clusterMaxLightsPerCluster();
	clusterHeaders.headers[clusterIndex].offset = clusterIndex * maxLightsPerCluster;
	atomicStore(&clusterHeaders.headers[clusterIndex].count, 0u);
	atomicStore(&clusterHeaders.headers[clusterIndex].flags, 0u);
	clusterHeaders.headers[clusterIndex].reserved = 0u;
}

@compute @workgroup_size(128, 1, 1)
fn csScatter(@builtin(global_invocation_id) globalId: vec3<u32>) {
	if (!clusteredEnabled()) {
		return;
	}
	let lightIndex = globalId.x;
	if (lightIndex >= activeClusterLightCount()) {
		return;
	}
	let cullData = clusterCullData.values[lightIndex];
	let metadata = clusterMetadata.values[lightIndex];
	let packedFlags = metadata.packedFlags;
	let lightType = packedFlags & CLUSTER_LIGHT_TYPE_MASK;
	if (lightType > CLUSTER_LIGHT_TYPE_AREA) {
		return;
	}
	let viewPos = worldToView(cullData.positionCullRadius.xyz);
	let cullRange = max(cullData.positionCullRadius.w, 0.001);
	let clusterRange = resolveLightClusterRange(viewPos, cullRange);
	if (!clusterRange.valid) {
		return;
	}
	let tilesX = max(clusterParams.tilesX, 1u);
	let tilesY = max(clusterParams.tilesY, 1u);
	let tilesPerLayer = max(tilesX * tilesY, 1u);
	var z = clusterRange.sliceMin;
	loop {
		var y = clusterRange.tileYMin;
		loop {
			var x = clusterRange.tileXMin;
			loop {
				let clusterIndex = x + y * tilesX + z * tilesPerLayer;
				if (clusterIndex < clusterParams.clusterCount) {
					let aabb = buildClusterAABB(x, y, z);
					if (lightIntersectsCluster(lightIndex, aabb)) {
						appendLightToCluster(clusterIndex, lightIndex, packedFlags);
					}
				}
				if (x >= clusterRange.tileXMax) { break; }
				x++;
			}
			if (y >= clusterRange.tileYMax) { break; }
			y++;
		}
		if (z >= clusterRange.sliceMax) { break; }
		z++;
	}
}

@compute @workgroup_size(128, 1, 1)
fn csFinalize(@builtin(global_invocation_id) globalId: vec3<u32>) {
	let clusterIndex = globalId.x;
	if (clusterIndex >= clusterParams.clusterCount) {
		return;
	}
	let maxLightsPerCluster = clusterMaxLightsPerCluster();
	let rawCount = atomicLoad(&clusterHeaders.headers[clusterIndex].count);
	var flags = atomicLoad(&clusterHeaders.headers[clusterIndex].flags);
	if (rawCount > maxLightsPerCluster) {
		flags |= CLUSTER_HEADER_FLAG_OVERFLOW;
	}
	atomicStore(&clusterHeaders.headers[clusterIndex].count, min(rawCount, maxLightsPerCluster));
	atomicStore(&clusterHeaders.headers[clusterIndex].flags, flags);
	clusterHeaders.headers[clusterIndex].reserved = rawCount;
}

@compute @workgroup_size(128, 1, 1)
fn csGather(
	@builtin(workgroup_id) workgroupId: vec3<u32>,
	@builtin(local_invocation_index) lane: u32
) {
	let clusterIndex = workgroupId.x;
	if (!clusteredEnabled() || clusterIndex >= clusterParams.clusterCount) {
		return;
	}
	let tilesX = max(clusterParams.tilesX, 1u);
	let tilesY = max(clusterParams.tilesY, 1u);
	let tilesPerLayer = tilesX * tilesY;
	let clusterZ = clusterIndex / tilesPerLayer;
	let layerIndex = clusterIndex - clusterZ * tilesPerLayer;
	let clusterY = layerIndex / tilesX;
	let clusterX = layerIndex - clusterY * tilesX;
	if (lane == 0u) {
		sharedAABB = buildClusterAABB(clusterX, clusterY, clusterZ);
		gatherBaseCount = 0u;
		atomicStore(&gatherSelectedFlags, 0u);
	}
	workgroupBarrier();

	let activeCount = activeClusterLightCount();
	let capacity = clusterMaxLightsPerCluster();
	var batchStart = 0u;
	loop {
		if (batchStart >= activeCount) { break; }
		let lightIndex = batchStart + lane;
		let hit = select(0u, 1u, lightIntersectsCluster(lightIndex, sharedAABB));
		gatherScanA[lane] = hit;
		workgroupBarrier();
		var scanOffset = 1u;
		loop {
			if (scanOffset >= CLUSTERED_LIGHT_WORKGROUP_SIZE) { break; }
			var scanned = gatherScanA[lane];
			if (lane >= scanOffset) {
				scanned += gatherScanA[lane - scanOffset];
			}
			gatherScanB[lane] = scanned;
			workgroupBarrier();
			gatherScanA[lane] = gatherScanB[lane];
			workgroupBarrier();
			scanOffset *= 2u;
		}
		let batchCount = gatherScanA[CLUSTERED_LIGHT_WORKGROUP_SIZE - 1u];
		let baseBefore = gatherBaseCount;
		if (hit != 0u) {
			let slot = baseBefore + gatherScanA[lane] - 1u;
			if (slot < capacity) {
				let packedFlags = clusterMetadata.values[lightIndex].packedFlags;
				clusterIndices.indices[clusterIndex * capacity + slot] =
					packClusteredLightRef(lightIndex, packedFlags);
				atomicOr(&gatherSelectedFlags, headerFlagsForLight(packedFlags));
			}
		}
		workgroupBarrier();
		if (lane == 0u) {
			gatherBaseCount = baseBefore + batchCount;
		}
		workgroupBarrier();
		batchStart += CLUSTERED_LIGHT_WORKGROUP_SIZE;
	}

	if (lane == 0u) {
		let rawCount = gatherBaseCount;
		var flags = atomicLoad(&gatherSelectedFlags);
		if (rawCount > capacity) {
			flags |= CLUSTER_HEADER_FLAG_OVERFLOW;
		}
		clusterHeaders.headers[clusterIndex].offset = clusterIndex * capacity;
		atomicStore(&clusterHeaders.headers[clusterIndex].count, min(rawCount, capacity));
		atomicStore(&clusterHeaders.headers[clusterIndex].flags, flags);
		clusterHeaders.headers[clusterIndex].reserved = rawCount;
	}
}

@compute @workgroup_size(128, 1, 1)
fn csResolveOverflow(
	@builtin(workgroup_id) workgroupId: vec3<u32>,
	@builtin(local_invocation_index) lane: u32
) {
	let clusterIndex = workgroupId.x;
	if (clusterIndex >= clusterParams.clusterCount ||
		(atomicLoad(&clusterHeaders.headers[clusterIndex].flags) &
			CLUSTER_HEADER_FLAG_OVERFLOW) == 0u) {
		return;
	}
	let tilesX = max(clusterParams.tilesX, 1u);
	let tilesY = max(clusterParams.tilesY, 1u);
	let tilesPerLayer = tilesX * tilesY;
	let clusterZ = clusterIndex / tilesPerLayer;
	let layerIndex = clusterIndex - clusterZ * tilesPerLayer;
	let clusterY = layerIndex / tilesX;
	let clusterX = layerIndex - clusterY * tilesX;
	if (lane == 0u) {
		sharedAABB = buildClusterAABB(clusterX, clusterY, clusterZ);
		atomicStore(&overflowMatchCount, 0u);
		atomicStore(&overflowSelectedFlags, 0u);
	}
	workgroupBarrier();

	let activeCount = activeClusterLightCount();
	var candidate = lane;
	loop {
		if (candidate >= CLUSTERED_LIGHT_MAX_LIGHTS) { break; }
		if (candidate < activeCount && lightIntersectsCluster(candidate, sharedAABB)) {
			let packedFlags = clusterMetadata.values[candidate].packedFlags;
			overflowScores[candidate] = estimateLightContribution(candidate, sharedAABB);
			overflowRefs[candidate] = packClusteredLightRef(candidate, packedFlags);
			atomicAdd(&overflowMatchCount, 1u);
		} else {
			overflowScores[candidate] = -1.0;
			overflowRefs[candidate] = 0xffffffffu;
		}
		candidate += CLUSTERED_LIGHT_WORKGROUP_SIZE;
	}
	workgroupBarrier();

	var sortSize = 2u;
	loop {
		if (sortSize > CLUSTERED_LIGHT_MAX_LIGHTS) { break; }
		var compareStride = sortSize / 2u;
		loop {
			if (compareStride == 0u) { break; }
			var index = lane;
			loop {
				if (index >= CLUSTERED_LIGHT_MAX_LIGHTS) { break; }
				let partner = index ^ compareStride;
				if (partner > index) {
					let descending = (index & sortSize) == 0u;
					let leftScore = overflowScores[index];
					let leftRef = overflowRefs[index];
					let rightScore = overflowScores[partner];
					let rightRef = overflowRefs[partner];
					let swap = select(
						scoreIsBetter(leftScore, leftRef, rightScore, rightRef),
						scoreIsBetter(rightScore, rightRef, leftScore, leftRef),
						descending
					);
					if (swap) {
						overflowScores[index] = rightScore;
						overflowRefs[index] = rightRef;
						overflowScores[partner] = leftScore;
						overflowRefs[partner] = leftRef;
					}
				}
				index += CLUSTERED_LIGHT_WORKGROUP_SIZE;
			}
			workgroupBarrier();
			compareStride /= 2u;
		}
		sortSize *= 2u;
	}

	let capacity = clusterMaxLightsPerCluster();
	let selectedCount = min(atomicLoad(&overflowMatchCount), capacity);
	if (lane < selectedCount) {
		let packedRef = overflowRefs[lane];
		clusterIndices.indices[clusterIndex * capacity + lane] = packedRef;
		let lightIndex = packedRef & CLUSTER_LIGHT_INDEX_MASK;
		atomicOr(
			&overflowSelectedFlags,
			headerFlagsForLight(clusterMetadata.values[lightIndex].packedFlags)
		);
	}
	workgroupBarrier();
	if (lane == 0u) {
		clusterHeaders.headers[clusterIndex].offset = clusterIndex * capacity;
		atomicStore(&clusterHeaders.headers[clusterIndex].count, selectedCount);
		atomicStore(
			&clusterHeaders.headers[clusterIndex].flags,
			CLUSTER_HEADER_FLAG_OVERFLOW | atomicLoad(&overflowSelectedFlags)
		);
		clusterHeaders.headers[clusterIndex].reserved = atomicLoad(&overflowMatchCount);
	}
}
