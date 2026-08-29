fn shadeSceneWithOptions(
	input: VertexOutput,
	frontFacing: bool,
	includeTransmissionBackground: bool,
	material: ResolvedLightingMaterial
) -> SceneFragmentOutput {
	let shadingMode = material.shadingMode;
	let alphaModeMask = materialCommon.materialParams.z > 0.5;
	let doubleSided = materialCommon.materialParams.w > 0.5;
	let enableLighting = frame.options.x > 0.5;

	let materialRenderFlags = u32(materialCommon.renderParams.x + 0.5);
	let isWireframe = (materialRenderFlags & 1u) != 0u;
	var baseSample = vec4<f32>(1.0);
	if (
		!isWireframe &&
		(shadingMode != SHADING_PBR ||
			hasPBRTexture(material, PBR_TEXTURE_BASE_COLOR_MAP))
	) {
		baseSample = sampleColorTexture(
			baseColorTexture,
			baseColorSampler,
			TEX_BASE_COLOR,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}

	let baseColor = materialCommon.baseColorFactor.rgb * baseSample.rgb;
	let alpha = clamp(materialCommon.baseColorFactor.a * baseSample.a, 0.0, 1.0);

	if (alphaModeMask && alpha < materialCommon.materialParams.x) {
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

	if (doubleSided && !frontFacing) {
		normal = -normal;
	}
	let geometryNormal = normal;
	let shadowNormal = normal;

	var emissiveSample = vec4<f32>(1.0);
	if (
		shadingMode != SHADING_PBR ||
		hasPBRTexture(material, PBR_TEXTURE_EMISSIVE_MAP)
	) {
		emissiveSample = sampleColorTexture(
			emissiveTexture,
			emissiveSampler,
			TEX_EMISSIVE,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	let emissive = materialCommon.emissiveFactor.rgb * emissiveSample.rgb * materialCommon.emissiveFactor.a;
	let linearDepth = dot(frame.cameraPosition.xyz - input.worldPosition, frame.environmentBasisBackward.xyz);
	let invCurrentW = 1.0 / max(abs(input.currentClip.w), EPSILON);
	let invPrevW = 1.0 / max(abs(input.prevClip.w), EPSILON);
	let currentNdc = input.currentClip.xy * invCurrentW;
	let prevNdc = input.prevClip.xy * invPrevW;
	let motion = currentNdc - prevNdc;

	if (shadingMode == SHADING_UNLIT || !enableLighting) {
		return buildSceneOutput(
			baseColor,
			alpha,
			baseColor,
			normal,
			1.0,
			0.0,
			emissive,
			1.0,
			motion,
			linearDepth
		);
	}

	if (shadingMode == SHADING_PHONG || shadingMode == SHADING_FLAT) {
