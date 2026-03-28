struct Params {
	invSizeRadiusIntensity: vec4<f32>,
	tuning: vec4<f32>,
}

@group(0) @binding(0) var sceneColor: texture_2d<f32>;
@group(0) @binding(1) var gAlbedoAlpha: texture_2d<f32>;
@group(0) @binding(2) var gNormalRoughMetal: texture_2d<f32>;
@group(0) @binding(3) var gMotionDepth: texture_2d<f32>;
@group(0) @binding(4) var linearSampler: sampler;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var outTex: texture_storage_2d<rgba16float, write>;

const SAMPLE_OFFSETS = array<vec2<f32>, 8>(
	vec2<f32>(1.0, 0.0),
	vec2<f32>(-1.0, 0.0),
	vec2<f32>(0.0, 1.0),
	vec2<f32>(0.0, -1.0),
	vec2<f32>(0.70710677, 0.70710677),
	vec2<f32>(-0.70710677, 0.70710677),
	vec2<f32>(0.70710677, -0.70710677),
	vec2<f32>(-0.70710677, -0.70710677),
);

fn decodeNormal(encoded: vec2<f32>) -> vec3<f32> {
	let xy = encoded * 2.0 - vec2<f32>(1.0, 1.0);
	let z2 = max(1.0 - dot(xy, xy), 0.0);
	return normalize(vec3<f32>(xy, sqrt(z2)));
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}

	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.invSizeRadiusIntensity.xy;
	let src = textureLoad(sceneColor, coord, 0);
	let centerDepth = textureLoad(gMotionDepth, coord, 0).z;
	if (centerDepth <= 0.0) {
		textureStore(outTex, coord, src);
		return;
	}

	let centerNormal = decodeNormal(textureLoad(gNormalRoughMetal, coord, 0).xy);
	let radiusPixels = max(params.invSizeRadiusIntensity.z, 1.0);
	let intensity = max(params.invSizeRadiusIntensity.w, 0.0);
	let falloff = max(params.tuning.x, 0.1);
	let depthPhi = max(params.tuning.y, 0.01);
	let normalPhi = max(params.tuning.z, 0.1);
	let albedoBoost = max(params.tuning.w, 0.0);

	var indirectAccum = vec3<f32>(0.0);
	var weightSum = 0.0;

	for (var i = 0u; i < 8u; i = i + 1u) {
		let offset = SAMPLE_OFFSETS[i];
		let sampleUv = clamp(
			uv + offset * radiusPixels * params.invSizeRadiusIntensity.xy,
			vec2<f32>(0.0),
			vec2<f32>(1.0)
		);
		let sampleDepth = textureSampleLevel(gMotionDepth, linearSampler, sampleUv, 0.0).z;
		if (sampleDepth <= 0.0) {
			continue;
		}

		let sampleNormal = decodeNormal(
			textureSampleLevel(gNormalRoughMetal, linearSampler, sampleUv, 0.0).xy
		);
		let sampleColor = textureSampleLevel(sceneColor, linearSampler, sampleUv, 0.0).rgb;
		let sampleAlbedo = textureSampleLevel(
			gAlbedoAlpha,
			linearSampler,
			sampleUv,
			0.0
		).rgb;
		let depthWeight = exp(-abs(sampleDepth - centerDepth) * depthPhi);
		let normalWeight = pow(max(dot(centerNormal, sampleNormal), 0.0), normalPhi);
		let distanceWeight = 1.0 / (1.0 + length(offset) * falloff);
		let weight = depthWeight * normalWeight * distanceWeight;
		let bounceTint = mix(vec3<f32>(1.0), sampleAlbedo * albedoBoost, 0.75);
		indirectAccum += sampleColor * bounceTint * weight;
		weightSum += weight;
	}

	let indirect = select(
		vec3<f32>(0.0),
		indirectAccum / max(weightSum, 1e-4),
		weightSum > 0.0
	);
	let finalColor = max(src.rgb + indirect * intensity, vec3<f32>(0.0));
	textureStore(outTex, coord, vec4<f32>(finalColor, src.a));
}
