struct MRTClearOutput {
	@location(0) sceneColor: vec4<f32>,
	@location(1) albedoAlpha: vec4<f32>,
	@location(2) normalRoughMetal: vec4<f32>,
	@location(3) emissiveOcclusion: vec4<f32>,
	@location(4) motionDepth: vec4<f32>,
}

struct ExtendedClearOutput {
	@location(0) sceneColor: vec4<f32>,
	@location(1) albedoAlpha: vec4<f32>,
	@location(2) normalRoughMetal: vec4<f32>,
	@location(3) emissiveOcclusion: vec4<f32>,
	@location(4) motionDepth: vec4<f32>,
	@location(5) specular: vec4<f32>,
	@location(6) coatSheen: vec4<f32>,
	@location(7) sheenReflectance: vec4<f32>,
}

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
	var positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0)
	);
	return vec4<f32>(positions[vertexIndex], 0.0, 1.0);
}

@fragment
fn fsColor() -> @location(0) vec4<f32> {
	return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fsMRT() -> MRTClearOutput {
	return MRTClearOutput(
		vec4<f32>(0.0, 0.0, 0.0, 1.0),
		vec4<f32>(0.0, 0.0, 0.0, 1.0),
		vec4<f32>(0.5, 0.5, 1.0, 0.0),
		vec4<f32>(0.0, 0.0, 0.0, 1.0),
		vec4<f32>(0.0)
	);
}

@fragment
fn fsDeferred() -> MRTClearOutput {
	return MRTClearOutput(
		vec4<f32>(0.0, 0.0, 0.0, 1.0),
		vec4<f32>(0.0),
		vec4<f32>(0.5, 0.5, 1.0, 0.0),
		vec4<f32>(0.0, 0.0, 0.0, 1.0),
		vec4<f32>(0.0)
	);
}

@fragment
fn fsExtended() -> ExtendedClearOutput {
	return ExtendedClearOutput(
		vec4<f32>(0.0, 0.0, 0.0, 1.0),
		vec4<f32>(0.0),
		vec4<f32>(0.5, 0.5, 1.0, 0.0),
		vec4<f32>(0.0, 0.0, 0.0, 1.0),
		vec4<f32>(0.0),
		vec4<f32>(0.0),
		vec4<f32>(0.0),
		vec4<f32>(0.0)
	);
}
