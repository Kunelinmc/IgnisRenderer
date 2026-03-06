export const WEBGPU_SCENE_FRAGMENT_SINGLE_TARGET = /* wgsl */ `
@fragment
fn fsMain(input: VertexOutput) -> SceneFragmentOutput {
	return shadeScene(input);
}

@fragment
fn fsMainSingle(input: VertexOutput) -> @location(0) vec4<f32> {
	return shadeScene(input).sceneColor;
}
`;
