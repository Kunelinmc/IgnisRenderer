@fragment
fn fsMain(input: VertexOutput) -> SceneFragmentOutput {
	return shadeScene(input);
}

@fragment
fn fsMainSingle(input: VertexOutput) -> @location(0) vec4<f32> {
	return shadeScene(input).sceneColor;
}

@fragment
fn fsMainOIT(input: VertexOutput) -> SceneFragmentOITOutput {
	let shaded = shadeScene(input);
	return buildSceneOITOutput(shaded.sceneColor, shaded.gMotionDepth.z);
}
