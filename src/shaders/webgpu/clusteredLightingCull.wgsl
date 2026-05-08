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
	count: u32,
	flags: u32,
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

// View-space AABB for a cluster tile
struct ClusterAABB {
	minX: f32,
	maxX: f32,
	minY: f32,
	maxY: f32,
	zNear: f32,
	zFar: f32,
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

@group(0) @binding(0) var<uniform> clusterParams: ClusterGridParams;
@group(0) @binding(1) var<storage, read> clusterLights: ClusterLightBuffer;
@group(0) @binding(2) var<storage, read_write> clusterHeaders: ClusterHeaderBuffer;
@group(0) @binding(3) var<storage, read_write> clusterIndices: ClusterLightIndexList;

@group(1) @binding(0) var<uniform> frame: FrameUniforms;

fn clusteredEnabled() -> bool {
	return frame.environmentOptionsB.w > 0.5 &&
		clusterParams.logScale > 0.0 &&
		clusterParams.zSlices > 0u;
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

// Build the view-space AABB for a given cluster tile.
// All cluster-constant projection math is done here ONCE.
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

	// NDC bounds for this tile
	let ndcLeftX = f32(clusterX) * invTilesX * 2.0 - 1.0;
	let ndcRightX = f32(clusterX + 1u) * invTilesX * 2.0 - 1.0;
	let ndcTopY = 1.0 - f32(clusterY) * invTilesY * 2.0;
	let ndcBottomY = 1.0 - f32(clusterY + 1u) * invTilesY * 2.0;

	let zNear = sliceDepthBoundary(clusterZ);
	let zFar = sliceDepthBoundary(clusterZ + 1u);

	// Project NDC bounds to view-space at both near and far depth,
	// then take the envelope to form the full frustum-aligned AABB.
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

// Proper sphere-AABB squared-distance test.
// Returns true if the light sphere overlaps the cluster AABB.
fn sphereIntersectsAABB(
	center: vec3<f32>,
	radius: f32,
	aabb: ClusterAABB
) -> bool {
	var distSq: f32 = 0.0;

	// X axis
	if (center.x < aabb.minX) {
		let d = aabb.minX - center.x;
		distSq += d * d;
	} else if (center.x > aabb.maxX) {
		let d = center.x - aabb.maxX;
		distSq += d * d;
	}

	// Y axis
	if (center.y < aabb.minY) {
		let d = aabb.minY - center.y;
		distSq += d * d;
	} else if (center.y > aabb.maxY) {
		let d = center.y - aabb.maxY;
		distSq += d * d;
	}

	// Z axis (depth)
	if (center.z < aabb.zNear) {
		let d = aabb.zNear - center.z;
		distSq += d * d;
	} else if (center.z > aabb.zFar) {
		let d = center.z - aabb.zFar;
		distSq += d * d;
	}

	return distSq <= radius * radius;
}

// Test whether AABB center is within the spot cone's half-angle,
// using a conservative separating-axis check.
fn spotConeIntersectsAABB(
	lightView: vec3<f32>,
	lightDirView: vec3<f32>,
	outerCos: f32,
	range: f32,
	aabb: ClusterAABB
) -> bool {
	// Clamp AABB center to the nearest point to the light
	let cx = clamp(lightView.x, aabb.minX, aabb.maxX);
	let cy = clamp(lightView.y, aabb.minY, aabb.maxY);
	let cz = clamp(lightView.z, aabb.zNear, aabb.zFar);
	let nearest = vec3<f32>(cx, cy, cz);
	let toNearest = nearest - lightView;
	let distNearest = length(toNearest);
	if (distNearest < 1e-6) {
		// Light is inside or touching the AABB - always intersects
		return true;
	}
	let dirToNearest = toNearest / distNearest;
	let cosAngle = dot(dirToNearest, lightDirView);

	// Use AABB half-diagonal as angular slack
	let halfExtentX = (aabb.maxX - aabb.minX) * 0.5;
	let halfExtentY = (aabb.maxY - aabb.minY) * 0.5;
	let halfExtentZ = (aabb.zFar - aabb.zNear) * 0.5;
	let halfDiag = sqrt(halfExtentX * halfExtentX + halfExtentY * halfExtentY + halfExtentZ * halfExtentZ);
	let angularSlack = halfDiag / max(distNearest, 1e-6);
	let effectiveCos = max(outerCos - angularSlack, -1.0);

	return cosAngle >= effectiveCos;
}

@compute @workgroup_size(128, 1, 1)
fn csMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
	let clusterIndex = globalId.x;
	if (clusterIndex >= clusterParams.clusterCount) {
		return;
	}

	let clusterTotal = max(clusterParams.clusterCount, 1u);
	let maxLightsPerCluster = max(arrayLength(&clusterIndices.indices) / clusterTotal, 1u);
	let baseOffset = clusterIndex * maxLightsPerCluster;
	let maxLights = arrayLength(&clusterLights.lights);
	if (!clusteredEnabled() || maxLights == 0u) {
		clusterHeaders.headers[clusterIndex] = ClusterHeader(baseOffset, 0u, 0u, 0u);
		return;
	}

	// Decompose flat cluster index into 3D tile coordinates
	let tilesPerLayer = max(clusterParams.tilesX * clusterParams.tilesY, 1u);
	let zSlice = clusterIndex / tilesPerLayer;
	let layerOffset = clusterIndex - zSlice * tilesPerLayer;
	let clusterY = layerOffset / max(clusterParams.tilesX, 1u);
	let clusterX = layerOffset - clusterY * max(clusterParams.tilesX, 1u);

	// Pre-compute the cluster AABB once - moves ALL projection math
	// out of the per-light loop.
	let tanHalfFov = max(frame.environmentBasisRight.w, 1e-6);
	let aspect = max(frame.environmentBasisUp.w, 1e-6);
	let aabb = buildClusterAABB(clusterX, clusterY, zSlice, tanHalfFov, aspect);

	var count: u32 = 0u;
	var flags: u32 = 0u;
	for (var lightIndex: u32 = 0u; lightIndex < maxLights; lightIndex = lightIndex + 1u) {
		let light = clusterLights.lights[lightIndex];
		let lightType = light.packedFlags & CLUSTER_LIGHT_TYPE_MASK;
		if (lightType > 1u) {
			continue;
		}

		let range = max(light.positionRange.w, 0.001);
		let viewPos = worldToView(light.positionRange.xyz);
		let lightDepth = max(viewPos.z, 0.0);

		// Fast depth-range rejection (cheapest test first)
		if (lightDepth + range < aabb.zNear || lightDepth - range > aabb.zFar) {
			continue;
		}

		// Sphere-AABB squared-distance test (tighter than axis-separated)
		if (!sphereIntersectsAABB(viewPos, range, aabb)) {
			continue;
		}

		// Spot light cone culling - reject clusters outside the cone
		if (lightType == CLUSTER_LIGHT_TYPE_SPOT) {
			let outerCos = light.directionOuter.w;
			if (outerCos > -0.999) {
				let lightDirView = dirToView(
					normalize(light.directionOuter.xyz)
				);
				if (!spotConeIntersectsAABB(viewPos, lightDirView, outerCos, range, aabb)) {
					continue;
				}
			}
		}

		// Check per-cluster budget - break on overflow instead of continuing
		if (count >= maxLightsPerCluster) {
			flags = flags | CLUSTER_HEADER_FLAG_OVERFLOW;
			break;
		}

		let castsShadow = (light.packedFlags & CLUSTER_LIGHT_FLAG_CASTS_SHADOW) != 0u;
		let affectsVolumetric = (light.packedFlags & CLUSTER_LIGHT_FLAG_AFFECTS_VOLUMETRIC) != 0u;

		var packedRef = (lightIndex & CLUSTER_LIGHT_INDEX_MASK) |
			((lightType & CLUSTER_LIGHT_TYPE_MASK) << CLUSTER_LIGHT_INDEX_TYPE_SHIFT);
		if (castsShadow) {
			packedRef = packedRef | CLUSTER_LIGHT_INDEX_SHADOW_BIT;
			flags = flags | CLUSTER_HEADER_FLAG_HAS_SHADOWED;
		}
		if (affectsVolumetric) {
			packedRef = packedRef | CLUSTER_LIGHT_INDEX_VOLUMETRIC_BIT;
			flags = flags | CLUSTER_HEADER_FLAG_HAS_VOLUMETRIC;
		}
		clusterIndices.indices[baseOffset + count] = packedRef;
		count = count + 1u;
	}

	clusterHeaders.headers[clusterIndex] = ClusterHeader(baseOffset, count, flags, 0u);
}
