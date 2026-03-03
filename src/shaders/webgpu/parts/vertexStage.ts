export const WEBGPU_SCENE_VERTEX_STAGE = /* wgsl */ `
@vertex
fn vsMain(input: VertexInput) -> VertexOutput {
	var output: VertexOutput;
	let worldPosition = model.modelMatrix * vec4<f32>(input.position, 1.0);
	let worldNormal = safeNormalize(
		(model.normalMatrix * vec4<f32>(input.normal, 0.0)).xyz,
		vec3<f32>(0.0, 0.0, 1.0)
	);
	let worldTangent = (model.normalMatrix * vec4<f32>(input.tangent.xyz, 0.0)).xyz;
	var clipPosition = frame.viewProjection * worldPosition;
	clipPosition.z = clipPosition.z * 0.5 + clipPosition.w * 0.5;

	output.position = clipPosition;
	output.worldPosition = worldPosition.xyz;
	output.worldNormal = worldNormal;
	output.uv = input.uv;
	output.worldTangent = vec4<f32>(
		safeNormalize(worldTangent, vec3<f32>(1.0, 0.0, 0.0)),
		input.tangent.w
	);
	output.uv2 = input.uv2;
	return output;
}
`
