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
