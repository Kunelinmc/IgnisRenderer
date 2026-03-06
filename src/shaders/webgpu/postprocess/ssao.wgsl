struct Params {
	fullInvSize: vec2<f32>,
	aoInvSize: vec2<f32>,
	radius: f32,
	bias: f32,
	intensity: f32,
	blurRadius: f32,
	blurSharpness: f32,
	_pad0: f32,
}

@group(0) @binding(0) var texA: texture_2d<f32>;
@group(0) @binding(1) var texB: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var outTex: texture_storage_2d<rgba16float, write>;

fn decodeNormal(encoded: vec2<f32>) -> vec3<f32> {
	let xy = encoded * 2.0 - vec2<f32>(1.0, 1.0);
	let z2 = max(1.0 - dot(xy, xy), 0.0);
	return normalize(vec3<f32>(xy, sqrt(z2)));
}

@compute @workgroup_size(8, 8, 1)
fn csRaw(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.aoInvSize;
	let center = textureSampleLevel(texB, linearSampler, uv, 0.0);
	let depth = center.z;
	if (depth <= 0.0) {
		textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(1.0, 1.0, 1.0, 1.0));
		return;
	}
	let normal = decodeNormal(textureSampleLevel(texA, linearSampler, uv, 0.0).xy);
	let kernel = array<vec2<f32>, 8>(
		vec2<f32>(1.0, 0.0),
		vec2<f32>(-1.0, 0.0),
		vec2<f32>(0.0, 1.0),
		vec2<f32>(0.0, -1.0),
		vec2<f32>(0.7071, 0.7071),
		vec2<f32>(-0.7071, 0.7071),
		vec2<f32>(0.7071, -0.7071),
		vec2<f32>(-0.7071, -0.7071)
	);
	var occ = 0.0;
	for (var i: i32 = 0; i < 8; i = i + 1) {
		let offset = normalize(vec3<f32>(kernel[i], 0.25) + normal * 0.2).xy;
		let sampleUv = uv + offset * params.radius * params.fullInvSize;
		let sampleDepth = textureSampleLevel(texB, linearSampler, sampleUv, 0.0).z;
		occ += select(0.0, 1.0, sampleDepth > 0.0 && sampleDepth < depth - params.bias);
	}
	let ao = clamp(1.0 - (occ / 8.0) * params.intensity, 0.0, 1.0);
	textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(ao, ao, ao, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn csBlur(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.aoInvSize;
	let centerDepth = textureSampleLevel(texB, linearSampler, uv, 0.0).z;
	let radius = clamp(i32(params.blurRadius + 0.5), i32(1), i32(4));
	var sum = 0.0;
	var weightSum = 0.0;
	for (var y: i32 = -radius; y <= radius; y = y + 1) {
		for (var x: i32 = -radius; x <= radius; x = x + 1) {
			let sampleCoord = clamp(
				coord + vec2<i32>(x, y),
				vec2<i32>(0, 0),
				vec2<i32>(i32(size.x) - 1, i32(size.y) - 1)
			);
			let sampleUv = (vec2<f32>(sampleCoord) + vec2<f32>(0.5)) * params.aoInvSize;
			let sampleDepth = textureSampleLevel(texB, linearSampler, sampleUv, 0.0).z;
			let depthDelta = abs(sampleDepth - centerDepth);
			let bilateral = exp(-depthDelta * max(params.blurSharpness, 1e-3));
			let sampleAo = textureLoad(texA, sampleCoord, 0).x;
			sum += sampleAo * bilateral;
			weightSum += bilateral;
		}
	}
	let ao = select(textureLoad(texA, coord, 0).x, sum / max(weightSum, 1e-4), weightSum > 0.0);
	textureStore(outTex, coord, vec4<f32>(ao, ao, ao, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn csCombine(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.fullInvSize;
	let color = textureLoad(texA, coord, 0);
	let ao = textureSampleLevel(texB, linearSampler, uv, 0.0).x;
	textureStore(
		outTex,
		coord,
		vec4<f32>(max(color.rgb * clamp(ao, 0.0, 1.0), vec3<f32>(0.0)), color.a)
	);
}
