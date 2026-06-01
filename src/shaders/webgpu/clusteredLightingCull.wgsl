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

struct ClusterLightRecord {
	positionRange: vec4<f32>,
	directionOuter: vec4<f32>,
	colorInner: vec4<f32>,
	packedFlags: u32,
	shadowIndex: u32,
	reserved0: u32,
	reserved1: u32,
}

struct ClusterHeader {
	offset: u32,
	count: atomic<u32>,
	flags: atomic<u32>,
	reserved: u32,
}

struct ClusterLightBuffer {
	lights: array<ClusterLightRecord>,
}

struct ClusterHeaderBuffer {
	headers: array<ClusterHeader>,
}

struct ClusterLightIndexList {
	indices: array<u32>,
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

@group(0) @binding(0) var<uniform> clusterParams: ClusterGridParams;
@group(0) @binding(1) var<storage, read> clusterLights: ClusterLightBuffer;
@group(0) @binding(2) var<storage, read_write> clusterHeaders: ClusterHeaderBuffer;
@group(0) @binding(3) var<storage, read_write> clusterIndices: ClusterLightIndexList;

@group(1) @binding(0) var<uniform> frame: FrameUniforms;

fn clusteredEnabled() -> bool {
	return frame.environmentOptionsB.w > 0.5 &&
		clusterParams.logScale > 0.0 &&
		clusterParams.clusterCount > 0u &&
		clusterParams.zSlices > 0u;
}

fn activeClusterLightCount() -> u32 {
	return min(clusterParams.lightCount, arrayLength(&clusterLights.lights));
}

fn clusterMaxLightsPerCluster() -> u32 {
	let clusterTotal = max(clusterParams.clusterCount, 1u);
	let bufferSpan = max(arrayLength(&clusterIndices.indices) / clusterTotal, 1u);
	return min(max(clusterParams.maxLightsPerCluster, 1u), bufferSpan);
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
	let safeScale = max(clusterParams.logScale, 1e-6);
	let logDepth = (f32(slice) - clusterParams.logBias) / safeScale;
	return clamp(exp(logDepth), clusterParams.near, clusterParams.far);
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

fn resolveLightClusterRange(
	viewPos: vec3<f32>,
	range: f32
) -> LightClusterRange {
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

	let xMinNdc = clamp(ndcX - radiusNdcX, -1.0, 1.0);
	let xMaxNdc = clamp(ndcX + radiusNdcX, -1.0, 1.0);
	let yMinNdc = clamp(ndcY - radiusNdcY, -1.0, 1.0);
	let yMaxNdc = clamp(ndcY + radiusNdcY, -1.0, 1.0);

	let tileXMin = clampToGridIndex((xMinNdc * 0.5 + 0.5) * f32(tilesX), tilesX);
	let tileXMax = clampToGridIndex((xMaxNdc * 0.5 + 0.5) * f32(tilesX), tilesX);
	let tileYMin = clampToGridIndex((0.5 - yMaxNdc * 0.5) * f32(tilesY), tilesY);
	let tileYMax = clampToGridIndex((0.5 - yMinNdc * 0.5) * f32(tilesY), tilesY);

	let zMin = clamp(max(clusterParams.near, viewDepth - range), clusterParams.near, clusterParams.far);
	let zMax = clamp(min(clusterParams.far, viewDepth + range), clusterParams.near, clusterParams.far);
	if (zMax < zMin) {
		return invalidLightClusterRange();
	}

	let sliceMin = depthToSlice(zMin);
	let sliceMax = depthToSlice(zMax);
	return LightClusterRange(
		min(tileXMin, tileXMax),
		max(tileXMin, tileXMax),
		min(tileYMin, tileYMax),
		max(tileYMin, tileYMax),
		min(sliceMin, sliceMax),
		max(sliceMin, sliceMax),
		true
	);
}

fn buildClusterAABB(
	clusterX: u32,
	clusterY: u32,
	clusterZ: u32,
	tanHalfFov: f32,
	aspect: f32
) -> ClusterAABB {
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

fn sphereIntersectsAABB(
	center: vec3<f32>,
	radius: f32,
	aabb: ClusterAABB
) -> bool {
	var distSq: f32 = 0.0;

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
	range: f32,
	aabb: ClusterAABB
) -> bool {
	let cx = clamp(lightView.x, aabb.minX, aabb.maxX);
	let cy = clamp(lightView.y, aabb.minY, aabb.maxY);
	let cz = clamp(lightView.z, aabb.zNear, aabb.zFar);
	let nearest = vec3<f32>(cx, cy, cz);
	let toNearest = nearest - lightView;
	let distNearest = length(toNearest);
	if (distNearest < 1e-6) {
		return true;
	}

	let dirToNearest = toNearest / distNearest;
	let cosAngle = dot(dirToNearest, lightDirView);
	let halfExtentX = (aabb.maxX - aabb.minX) * 0.5;
	let halfExtentY = (aabb.maxY - aabb.minY) * 0.5;
	let halfExtentZ = (aabb.zFar - aabb.zNear) * 0.5;
	let halfDiag = sqrt(
		halfExtentX * halfExtentX +
		halfExtentY * halfExtentY +
		halfExtentZ * halfExtentZ
	);
	let angularSlack = halfDiag / max(distNearest, 1e-6);
	let effectiveCos = max(outerCos - angularSlack, -1.0);

	return cosAngle >= effectiveCos;
}

fn packClusteredLightRef(
	lightIndex: u32,
	lightType: u32,
	packedFlags: u32
) -> u32 {
	var packedRef = (lightIndex & CLUSTER_LIGHT_INDEX_MASK) |
		((lightType & CLUSTER_LIGHT_TYPE_MASK) << CLUSTER_LIGHT_INDEX_TYPE_SHIFT);
	if ((packedFlags & CLUSTER_LIGHT_FLAG_CASTS_SHADOW) != 0u) {
		packedRef = packedRef | CLUSTER_LIGHT_INDEX_SHADOW_BIT;
	}
	if ((packedFlags & CLUSTER_LIGHT_FLAG_AFFECTS_VOLUMETRIC) != 0u) {
		packedRef = packedRef | CLUSTER_LIGHT_INDEX_VOLUMETRIC_BIT;
	}
	return packedRef;
}

fn headerFlagsForLight(packedFlags: u32) -> u32 {
	var flags = 0u;
	if ((packedFlags & CLUSTER_LIGHT_FLAG_CASTS_SHADOW) != 0u) {
		flags = flags | CLUSTER_HEADER_FLAG_HAS_SHADOWED;
	}
	if ((packedFlags & CLUSTER_LIGHT_FLAG_AFFECTS_VOLUMETRIC) != 0u) {
		flags = flags | CLUSTER_HEADER_FLAG_HAS_VOLUMETRIC;
	}
	return flags;
}

fn appendLightToCluster(
	clusterIndex: u32,
	lightIndex: u32,
	lightType: u32,
	packedFlags: u32
) {
	let maxLightsPerCluster = clusterMaxLightsPerCluster();
	let previousCount = atomicAdd(&clusterHeaders.headers[clusterIndex].count, 1u);
	if (previousCount >= maxLightsPerCluster) {
		atomicOr(
			&clusterHeaders.headers[clusterIndex].flags,
			CLUSTER_HEADER_FLAG_OVERFLOW
		);
		return;
	}

	let offset = clusterHeaders.headers[clusterIndex].offset + previousCount;
	if (offset >= arrayLength(&clusterIndices.indices)) {
		atomicOr(
			&clusterHeaders.headers[clusterIndex].flags,
			CLUSTER_HEADER_FLAG_OVERFLOW
		);
		return;
	}

	clusterIndices.indices[offset] =
		packClusteredLightRef(lightIndex, lightType, packedFlags);
	let flags = headerFlagsForLight(packedFlags);
	if (flags != 0u) {
		atomicOr(&clusterHeaders.headers[clusterIndex].flags, flags);
	}
}

@compute @workgroup_size(128, 1, 1)
fn csClear(@builtin(global_invocation_id) globalId: vec3<u32>) {
	let clusterIndex = globalId.x;
	if (clusterIndex >= clusterParams.clusterCount) {
		return;
	}

	let maxLightsPerCluster = clusterMaxLightsPerCluster();
	clusterHeaders.headers[clusterIndex].offset =
		clusterIndex * maxLightsPerCluster;
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
	let activeLightCount = activeClusterLightCount();
	if (lightIndex >= activeLightCount) {
		return;
	}

	let light = clusterLights.lights[lightIndex];
	let packedFlags = light.packedFlags;
	let lightType = packedFlags & CLUSTER_LIGHT_TYPE_MASK;
	if (lightType > CLUSTER_LIGHT_TYPE_SPOT) {
		return;
	}

	let range = max(light.positionRange.w, 0.001);
	let viewPos = worldToView(light.positionRange.xyz);
	let clusterRange = resolveLightClusterRange(viewPos, range);
	if (!clusterRange.valid) {
		return;
	}

	var directionOuterView = vec4<f32>(
		0.0,
		0.0,
		0.0,
		light.directionOuter.w
	);
	if (lightType == CLUSTER_LIGHT_TYPE_SPOT && light.directionOuter.w > -0.999) {
		let lightDirView = dirToView(normalize(light.directionOuter.xyz));
		directionOuterView = vec4<f32>(lightDirView, light.directionOuter.w);
	}

	let tanHalfFov = max(frame.environmentBasisRight.w, 1e-6);
	let aspect = max(frame.environmentBasisUp.w, 1e-6);
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
					let aabb = buildClusterAABB(x, y, z, tanHalfFov, aspect);
					if (!(viewPos.z + range < aabb.zNear ||
						viewPos.z - range > aabb.zFar) &&
						sphereIntersectsAABB(viewPos, range, aabb)) {
						var intersects = true;
						if (lightType == CLUSTER_LIGHT_TYPE_SPOT &&
							directionOuterView.w > -0.999) {
							intersects = spotConeIntersectsAABB(
								viewPos,
								directionOuterView.xyz,
								directionOuterView.w,
								range,
								aabb
							);
						}
						if (intersects) {
							appendLightToCluster(
								clusterIndex,
								lightIndex,
								lightType,
								packedFlags
							);
						}
					}
				}

				if (x >= clusterRange.tileXMax) {
					break;
				}
				x = x + 1u;
			}

			if (y >= clusterRange.tileYMax) {
				break;
			}
			y = y + 1u;
		}

		if (z >= clusterRange.sliceMax) {
			break;
		}
		z = z + 1u;
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
		flags = flags | CLUSTER_HEADER_FLAG_OVERFLOW;
	}

	atomicStore(
		&clusterHeaders.headers[clusterIndex].count,
		min(rawCount, maxLightsPerCluster)
	);
	atomicStore(&clusterHeaders.headers[clusterIndex].flags, flags);
}
