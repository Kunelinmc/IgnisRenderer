fn ignisSelectShaderMaterialUv(
	uv0: vec2<f32>,
	uv1: vec2<f32>,
	uv2: vec2<f32>,
	uv3: vec2<f32>,
	uvSet: u32
) -> vec2<f32> {
	if (uvSet == 1u) { return uv1; }
	if (uvSet == 2u) { return uv2; }
	if (uvSet >= 3u) { return uv3; }
	return uv0;
}

fn ignisDecodeShaderMaterialSample(
	sampled: vec4<f32>,
	linear: bool
) -> vec4<f32> {
	if (linear) { return sampled; }
	let rgb = select(
		pow((sampled.rgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4)),
		sampled.rgb / vec3<f32>(12.92),
		sampled.rgb <= vec3<f32>(0.04045)
	);
	return vec4<f32>(rgb, sampled.a);
}
