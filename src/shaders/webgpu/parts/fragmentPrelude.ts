export const WEBGPU_SCENE_FRAGMENT_PRELUDE = /* wgsl */ `
@fragment
fn fsMain(input: VertexOutput) -> @location(0) vec4<f32> {
	let shadingMode = u32(model.materialFlags.x + 0.5);
	let alphaModeMask = model.materialFlags.y > 0.5;
	let doubleSided = model.materialFlags.z > 0.5;
	let enableLighting = frame.options.x > 0.5;

	let baseSample = sampleColorTexture(
		baseColorTexture,
		baseColorSampler,
		TEX_BASE_COLOR,
		input.uv,
		input.uv2
	);
	let baseColor = model.baseColorFactor.rgb * baseSample.rgb;
	let alpha = clamp(model.baseColorFactor.a * baseSample.a, 0.0, 1.0);

	if (alphaModeMask && alpha < model.surfaceParams0.w) {
		discard;
	}

	let viewDir = safeNormalize(
		frame.cameraPosition.xyz - input.worldPosition,
		vec3<f32>(0.0, 0.0, 1.0)
	);

	var normal = safeNormalize(input.worldNormal, vec3<f32>(0.0, 0.0, 1.0));
	if (shadingMode == SHADING_FLAT) {
		let faceNormal = cross(dpdx(input.worldPosition), dpdy(input.worldPosition));
		normal = safeNormalize(faceNormal, normal);
	}

	if (doubleSided && dot(normal, viewDir) < 0.0) {
		normal = -normal;
	}

	let emissiveSample = sampleColorTexture(
		emissiveTexture,
		emissiveSampler,
		TEX_EMISSIVE,
		input.uv,
		input.uv2
	);
	let emissive = model.emissiveFactor.rgb * emissiveSample.rgb * model.emissiveFactor.a;

	if (shadingMode == SHADING_UNLIT || !enableLighting) {
		let outputColor = encodeOutput(baseColor);
		return vec4<f32>(clamp(outputColor, vec3<f32>(0.0), vec3<f32>(1.0)), alpha);
	}

	if (shadingMode == SHADING_PHONG || shadingMode == SHADING_FLAT) {
`
