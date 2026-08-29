fn resolveOITWeight(alpha: f32, linearDepth: f32) -> f32 {
	let clampedAlpha = clamp(alpha, 0.0, 1.0);
	let normalizedDepth = clamp(linearDepth / 400.0, 0.0, 1.0);
	let depthWeight = clamp(1.0 - normalizedDepth, 0.05, 1.0);
	let alphaWeight = max(clampedAlpha * 8.0 + 0.01, 0.01);
	let weight = alphaWeight * alphaWeight * alphaWeight * depthWeight;
	return clamp(weight, 1e-2, 3e3);
}

fn buildSceneOITOutput(sceneColor: vec4<f32>, linearDepth: f32) -> SceneFragmentOITOutput {
	let alpha = clamp(sceneColor.a, 0.0, 1.0);
	let weight = resolveOITWeight(alpha, linearDepth);
	var output: SceneFragmentOITOutput;
	output.accum = vec4<f32>(sceneColor.rgb * alpha, alpha) * weight;
	output.reveal = vec4<f32>(alpha, alpha, alpha, alpha);
	return output;
}

fn shadeScene(
	input: VertexOutput,
	frontFacing: bool,
	material: ResolvedLightingMaterial
) -> SceneFragmentOutput {
	return shadeSceneWithOptions(input, frontFacing, true, material);
}

fn shadeTransmissionCapture(
	input: VertexOutput,
	frontFacing: bool,
	material: ResolvedLightingMaterial
) -> TransmissionFragmentOutput {
	let shaded = shadeSceneWithOptions(input, frontFacing, false, material);
	let shadingMode = material.shadingMode;
	let alphaModeMask = materialCommon.materialParams.z > 0.5;
	let doubleSided = materialCommon.materialParams.w > 0.5;
	let materialRenderFlags = u32(materialCommon.renderParams.x + 0.5);
	let isWireframe = (materialRenderFlags & 1u) != 0u;

	var baseSample = vec4<f32>(1.0);
	if (!isWireframe && hasPBRTexture(material, PBR_TEXTURE_BASE_COLOR_MAP)) {
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

	var normalSample = vec3<f32>(0.5, 0.5, 1.0);
	if (hasPBRTexture(material, PBR_TEXTURE_NORMAL_MAP)) {
		normalSample = sampleLinearTexture(
			normalTexture,
			normalSampler,
			TEX_NORMAL,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		).rgb;
	}
	var pbrNormal = applyNormalMap(
		normal,
		input.worldTangent,
		normalSample,
		material.pbr.surfaceParams1.y
	);
	if (dot(pbrNormal, geometryNormal) < 0.0) {
		pbrNormal = -pbrNormal;
	}

	var mrSample = vec4<f32>(1.0);
	if (hasPBRTexture(material, PBR_TEXTURE_METALLIC_ROUGHNESS_MAP)) {
		mrSample = sampleLinearTexture(
			metallicRoughnessTexture,
			metallicRoughnessSampler,
			TEX_METALLIC_ROUGHNESS,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var transmissionSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_TRANSMISSION) &&
		hasPBRTexture(material, PBR_TEXTURE_TRANSMISSION_MAP)
	) {
		transmissionSample = sampleLinearTexture(
			transmissionTexture,
			transmissionSampler,
			TEX_TRANSMISSION,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var thicknessSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_TRANSMISSION) &&
		hasPBRTexture(material, PBR_TEXTURE_THICKNESS_MAP)
	) {
		thicknessSample = sampleLinearTexture(
			thicknessTexture,
			transmissionSampler,
			TEX_THICKNESS,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}

	let roughness = clamp(material.pbr.surfaceParams0.x * mrSample.g, 0.04, 1.0);
	let metalness = clamp(material.pbr.surfaceParams0.y * mrSample.b, 0.0, 1.0);
	let transmission =
		clamp(material.pbr.surfaceParams2.y * transmissionSample.r, 0.0, 1.0) *
		(1.0 - metalness);
	if (transmission <= EPSILON) {
		discard;
	}
	let ior = max(material.pbr.surfaceParams2.z, 1.0);
	let thickness = max(material.pbr.surfaceParams2.w * thicknessSample.g, 0.0);
	let attenuationDistance = material.pbr.surfaceParams3.x;
	let attenuationColor = clamp(
		material.pbr.attenuationColor.rgb,
		vec3<f32>(0.0001),
		vec3<f32>(1.0)
	);
	var transmissionPathLength = thickness;
	let interfaceNdotV = max(dot(pbrNormal, viewDir), PBR_MIN_NDOTV);
	if (thickness > 0.0) {
		let eta = 1.0 / ior;
		let sin2ThetaT = eta * eta * (1.0 - interfaceNdotV * interfaceNdotV);
		let cosThetaT = sqrt(max(1.0 - sin2ThetaT, 0.0));
		transmissionPathLength = thickness / max(cosThetaT, PBR_MIN_NDOTV);
	}
	var volumeAttenuation = vec3<f32>(1.0);
	if (thickness > 0.0 && attenuationDistance > 0.0) {
		let absorb = -log(attenuationColor) / attenuationDistance;
		volumeAttenuation = exp(-absorb * transmissionPathLength);
	}

	let f0Scalar = pow((ior - 1.0) / max(ior + 1.0, EPSILON), 2.0);
	let nDotV = max(interfaceNdotV, PBR_MIN_NDOTV);
	let fresnelAverage = clamp(
		f0Scalar + (1.0 - f0Scalar) * pow(max(1.0 - nDotV, 0.0), 5.0),
		0.0,
		1.0
	);
	let coverage = alpha;
	let tint = clamp(baseColor, vec3<f32>(0.0), vec3<f32>(1.0)) * volumeAttenuation;

	var output: TransmissionFragmentOutput;
	output.lighting = vec4<f32>(shaded.sceneColor.rgb * coverage, coverage);
	output.surface0 = vec4<f32>(
		encodeNormalForGBuffer(pbrNormal),
		max(shaded.gMotionDepth.z, 0.0),
		transmission
	);
	output.surface1 = vec4<f32>(ior, thickness, roughness, fresnelAverage);
	output.surface2 = vec4<f32>(tint, coverage);
	return output;
}

fn buildFamilyOITOutput(
	input: VertexOutput,
	frontFacing: bool,
	material: ResolvedLightingMaterial
) -> SceneFragmentOITOutput {
	let shaded = shadeScene(input, frontFacing, material);
	return buildSceneOITOutput(shaded.sceneColor, shaded.gMotionDepth.z);
}

@fragment fn fsMainPBR(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> SceneFragmentOutput {
	return shadeScene(input, frontFacing, resolvePBRLightingMaterial());
}
@fragment fn fsMainPhong(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> SceneFragmentOutput {
	return shadeScene(input, frontFacing, resolvePhongLightingMaterial());
}
@fragment fn fsMainFlat(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> SceneFragmentOutput {
	return shadeScene(input, frontFacing, resolveFlatLightingMaterial());
}
@fragment fn fsMainUnlit(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> SceneFragmentOutput {
	return shadeScene(input, frontFacing, resolveUnlitLightingMaterial());
}

@fragment fn fsMainSinglePBR(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> @location(0) vec4<f32> {
	return shadeScene(input, frontFacing, resolvePBRLightingMaterial()).sceneColor;
}
@fragment fn fsMainSinglePhong(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> @location(0) vec4<f32> {
	return shadeScene(input, frontFacing, resolvePhongLightingMaterial()).sceneColor;
}
@fragment fn fsMainSingleFlat(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> @location(0) vec4<f32> {
	return shadeScene(input, frontFacing, resolveFlatLightingMaterial()).sceneColor;
}
@fragment fn fsMainSingleUnlit(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> @location(0) vec4<f32> {
	return shadeScene(input, frontFacing, resolveUnlitLightingMaterial()).sceneColor;
}

@fragment fn fsMainOITPBR(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> SceneFragmentOITOutput {
	return buildFamilyOITOutput(input, frontFacing, resolvePBRLightingMaterial());
}
@fragment fn fsMainOITPhong(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> SceneFragmentOITOutput {
	return buildFamilyOITOutput(input, frontFacing, resolvePhongLightingMaterial());
}
@fragment fn fsMainOITFlat(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> SceneFragmentOITOutput {
	return buildFamilyOITOutput(input, frontFacing, resolveFlatLightingMaterial());
}
@fragment fn fsMainOITUnlit(input: VertexOutput, @builtin(front_facing) frontFacing: bool)
	-> SceneFragmentOITOutput {
	return buildFamilyOITOutput(input, frontFacing, resolveUnlitLightingMaterial());
}

@fragment
fn fsMainTransmissionCapturePBR(
	input: VertexOutput,
	@builtin(front_facing) frontFacing: bool
) -> TransmissionFragmentOutput {
	return shadeTransmissionCapture(
		input,
		frontFacing,
		resolvePBRLightingMaterial()
	);
}

@fragment
fn fsMainDepthMask(input: VertexOutput) {
	let alphaModeMask = materialCommon.materialParams.z > 0.5;
	if (!alphaModeMask) {
		return;
	}
	let materialRenderFlags = u32(materialCommon.renderParams.x + 0.5);
	let isWireframe = (materialRenderFlags & 1u) != 0u;
	var baseSample = vec4<f32>(1.0);
	if (
		!isWireframe
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
	let alpha = clamp(materialCommon.baseColorFactor.a * baseSample.a, 0.0, 1.0);
	if (alpha < materialCommon.materialParams.x) {
		discard;
	}
}
