const PI: f32 = 3.141592653589793;
const TWO_PI: f32 = 6.283185307179586;

struct PrefilterParams {
	outputWidth: u32,
	outputHeight: u32,
	sourceWidth: u32,
	sourceHeight: u32,
	roughness: f32,
	sampleCount: u32,
	sourceIsLinear: u32,
	_pad: u32,
};

@group(0) @binding(0) var envSampler: sampler;
@group(0) @binding(1) var envTexture: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: PrefilterParams;

fn radicalInverseVdC(bits: u32) -> f32 {
	var value = bits;
	value = (value << 16u) | (value >> 16u);
	value = ((value & 0x55555555u) << 1u) | ((value & 0xAAAAAAAAu) >> 1u);
	value = ((value & 0x33333333u) << 2u) | ((value & 0xCCCCCCCCu) >> 2u);
	value = ((value & 0x0F0F0F0Fu) << 4u) | ((value & 0xF0F0F0F0u) >> 4u);
	value = ((value & 0x00FF00FFu) << 8u) | ((value & 0xFF00FF00u) >> 8u);
	return f32(value) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, count: u32) -> vec2<f32> {
	return vec2<f32>(
		f32(i) / max(f32(count), 1.0),
		radicalInverseVdC(i)
	);
}

fn importanceSampleGGX(xi: vec2<f32>, normal: vec3<f32>, roughness: f32) -> vec3<f32> {
	let a = max(roughness * roughness, 1e-4);
	let a2 = a * a;
	let phi = TWO_PI * xi.x;
	let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
	let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

	let tangentHalf = vec3<f32>(
		cos(phi) * sinTheta,
		sin(phi) * sinTheta,
		cosTheta
	);

	let up = select(
		vec3<f32>(0.0, 1.0, 0.0),
		vec3<f32>(1.0, 0.0, 0.0),
		abs(normal.y) > 0.999
	);
	let tangent = normalize(cross(up, normal));
	let bitangent = cross(normal, tangent);
	return normalize(
		tangent * tangentHalf.x +
		bitangent * tangentHalf.y +
		normal * tangentHalf.z
	);
}

fn directionToEquirectUV(direction: vec3<f32>) -> vec2<f32> {
	let normalized = normalize(direction);
	let phi = atan2(normalized.x, normalized.z);
	let theta = acos(clamp(normalized.y, -1.0, 1.0));
	let u = (phi + PI) / TWO_PI;
	let v = theta / PI;
	return vec2<f32>(u, v);
}

fn sRGBToLinear(color: vec3<f32>) -> vec3<f32> {
	let low = color / 12.92;
	let high = pow((color + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
	return select(high, low, color <= vec3<f32>(0.04045));
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
	if (globalId.x >= params.outputWidth || globalId.y >= params.outputHeight) {
		return;
	}

	let outputSize = vec2<f32>(f32(params.outputWidth), f32(params.outputHeight));
	let uv = (vec2<f32>(vec2<u32>(globalId.xy)) + vec2<f32>(0.5)) / outputSize;
	let theta = uv.y * PI;
	let phi = uv.x * TWO_PI;
	let normal = vec3<f32>(
		sin(theta) * sin(phi),
		cos(theta),
		sin(theta) * cos(phi)
	);

	let sampleCount = max(params.sampleCount, 1u);
	var totalWeight = 0.0;
	var accumulated = vec3<f32>(0.0);

	for (var i: u32 = 0u; i < sampleCount; i = i + 1u) {
		let xi = hammersley(i, sampleCount);
		let halfVector = importanceSampleGGX(xi, normal, params.roughness);
		let nDotH = max(dot(normal, halfVector), 0.0);
		let lightDir = normalize(2.0 * nDotH * halfVector - normal);
		let nDotL = max(dot(normal, lightDir), 0.0);
		if (nDotL <= 0.0) {
			continue;
		}

		let sampleUv = directionToEquirectUV(lightDir);
		var sampleColor = textureSampleLevel(envTexture, envSampler, sampleUv, 0.0).rgb;
		if (params.sourceIsLinear == 0u) {
			sampleColor = sRGBToLinear(sampleColor);
		}

		accumulated += sampleColor * nDotL;
		totalWeight += nDotL;
	}

	var outputColor = vec3<f32>(0.0);
	if (totalWeight > 1e-6) {
		outputColor = accumulated / totalWeight;
	}

	textureStore(
		outputTexture,
		vec2<i32>(vec2<u32>(globalId.xy)),
		vec4<f32>(outputColor, 1.0)
	);
}
