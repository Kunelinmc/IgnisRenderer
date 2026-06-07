struct OcclusionCandidate {
	rect: vec4<f32>,
	depth: vec4<f32>,
};

struct OcclusionParams {
	depthBias: f32,
	_reserved0: f32,
	_reserved1: f32,
	_reserved2: f32,
};

@group(0) @binding(0) var hiZTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> candidates: array<OcclusionCandidate>;
@group(0) @binding(2) var<uniform> params: OcclusionParams;
@group(0) @binding(3) var<storage, read_write> results: array<u32>;

fn chooseMip(rectSize: vec2<f32>) -> u32 {
	let maxDimension = max(max(rectSize.x, rectSize.y), 1.0);
	let requested = u32(max(floor(log2(maxDimension * 0.25)), 0.0));
	let maxMip = textureNumLevels(hiZTex) - 1u;
	return min(requested, maxMip);
}

fn loadMaxDepth(pointPx: vec2<f32>, mip: u32) -> f32 {
	let mipSize = vec2<i32>(textureDimensions(hiZTex, mip));
	let scale = exp2(f32(mip));
	let coord = clamp(
		vec2<i32>(floor(pointPx / vec2<f32>(scale))),
		vec2<i32>(0, 0),
		mipSize - vec2<i32>(1, 1)
	);
	return textureLoad(hiZTex, coord, mip).y;
}

fn sampleOccludes(pointPx: vec2<f32>, mip: u32, candidateNear: f32) -> bool {
	let maxDepth = loadMaxDepth(pointPx, mip);
	return maxDepth > 0.0 && maxDepth < candidateNear - params.depthBias;
}

@compute @workgroup_size(64, 1, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let index = gid.x;
	if (index >= arrayLength(&candidates)) {
		return;
	}
	let candidate = candidates[index];
	let rect = candidate.rect;
	let candidateNear = candidate.depth.x;
	if (rect.z <= 0.0 || rect.w <= 0.0 || candidateNear <= 0.0) {
		results[index] = 1u;
		return;
	}

	let mip = chooseMip(rect.zw);
	let minPoint = rect.xy;
	let maxPoint = rect.xy + max(rect.zw - vec2<f32>(1.0), vec2<f32>(0.0));
	let center = (minPoint + maxPoint) * 0.5;
	let occluded =
		sampleOccludes(minPoint, mip, candidateNear) &&
		sampleOccludes(vec2<f32>(maxPoint.x, minPoint.y), mip, candidateNear) &&
		sampleOccludes(vec2<f32>(minPoint.x, maxPoint.y), mip, candidateNear) &&
		sampleOccludes(maxPoint, mip, candidateNear) &&
		sampleOccludes(center, mip, candidateNear);
	results[index] = select(1u, 0u, occluded);
}
