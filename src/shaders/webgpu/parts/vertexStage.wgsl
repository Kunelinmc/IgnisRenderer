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
	let currJitter = frame.taaJitterCurrentPrev.xy * clipPosition.w;
	clipPosition = vec4<f32>(
		clipPosition.x + currJitter.x,
		clipPosition.y + currJitter.y,
		clipPosition.z,
		clipPosition.w
	);
	clipPosition.z = clipPosition.z * 0.5 + clipPosition.w * 0.5;
	let prevWorldPosition = model.prevModelMatrix * vec4<f32>(input.position, 1.0);
	var prevClipPosition = frame.prevViewProjection * prevWorldPosition;
	let prevJitter = frame.taaJitterCurrentPrev.zw * prevClipPosition.w;
	prevClipPosition = vec4<f32>(
		prevClipPosition.x + prevJitter.x,
		prevClipPosition.y + prevJitter.y,
		prevClipPosition.z,
		prevClipPosition.w
	);
	prevClipPosition.z = prevClipPosition.z * 0.5 + prevClipPosition.w * 0.5;

	output.position = clipPosition;
	output.worldPosition = worldPosition.xyz;
	output.worldNormal = worldNormal;
	output.uv = input.uv;
	output.worldTangent = vec4<f32>(
		safeNormalize(worldTangent, vec3<f32>(1.0, 0.0, 0.0)),
		input.tangent.w
	);
	output.uv2 = input.uv2;
	output.currentClip = clipPosition;
	output.prevClip = prevClipPosition;
	return output;
}
