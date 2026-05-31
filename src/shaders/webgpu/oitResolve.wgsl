struct ResolveVSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var sceneColorTexture: texture_2d<f32>;
@group(0) @binding(1) var oitAccumTexture: texture_2d<f32>;
@group(0) @binding(2) var oitRevealTexture: texture_2d<f32>;
@group(0) @binding(3) var linearSampler: sampler;

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> ResolveVSOut {
	var positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0)
	);
	let pos = positions[vertexIndex];
	var output: ResolveVSOut;
	output.position = vec4<f32>(pos, 0.0, 1.0);
	output.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
	return output;
}

@fragment
fn fsMain(input: ResolveVSOut) -> @location(0) vec4<f32> {
	let base = textureSample(sceneColorTexture, linearSampler, input.uv);
	let accum = textureSample(oitAccumTexture, linearSampler, input.uv);
	let reveal = clamp(
		textureSample(oitRevealTexture, linearSampler, input.uv).r,
		0.0,
		1.0
	);
	let weightedColor = accum.rgb / max(accum.a, 1e-5);
	let alpha = clamp(1.0 - reveal, 0.0, 1.0);
	let color = weightedColor * alpha + base.rgb * reveal;
	return vec4<f32>(max(color, vec3<f32>(0.0)), base.a);
}
