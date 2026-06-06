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

fn shadeScene(input: VertexOutput) -> SceneFragmentOutput {
	return shadeSceneWithOptions(input, true);
}

fn shadeTransmissionCapture(input: VertexOutput) -> TransmissionFragmentOutput {
	let shaded = shadeSceneWithOptions(input, false);
	let shadingMode = u32(model.materialFlags.x + 0.5);
	let alphaModeMask = model.materialFlags.y > 0.5;
	let doubleSided = model.materialFlags.z > 0.5;
	let isWireframe = model.materialFlags.w > 0.5;

	var baseSample = vec4<f32>(1.0);
	if (!isWireframe) {
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

	let normalSample = sampleLinearTexture(
		normalTexture,
		normalSampler,
		TEX_NORMAL,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	).rgb;
	var pbrNormal = applyNormalMap(
		normal,
		input.worldTangent,
		normalSample,
		model.surfaceParams1.y
	);
	if (doubleSided && dot(pbrNormal, viewDir) < 0.0) {
		pbrNormal = -pbrNormal;
	}

	let mrSample = sampleLinearTexture(
		metallicRoughnessTexture,
		metallicRoughnessSampler,
		TEX_METALLIC_ROUGHNESS,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
	let transmissionSample = sampleLinearTexture(
		transmissionTexture,
		transmissionSampler,
		TEX_TRANSMISSION,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
	let thicknessSample = sampleLinearTexture(
		thicknessTexture,
		thicknessSampler,
		TEX_THICKNESS,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);

	let roughness = clamp(model.surfaceParams0.x * mrSample.g, 0.04, 1.0);
	let transmission = clamp(model.surfaceParams2.y * transmissionSample.r, 0.0, 1.0);
	if (transmission <= EPSILON) {
		discard;
	}
	let ior = max(model.surfaceParams2.z, 1.0);
	let thickness = max(model.surfaceParams2.w * thicknessSample.g, 0.0);
	let attenuationDistance = model.surfaceParams3.x;
	let attenuationColor = clamp(
		model.attenuationColor.rgb,
		vec3<f32>(0.0001),
		vec3<f32>(1.0)
	);
	var volumeAttenuation = vec3<f32>(1.0);
	if (thickness > 0.0 && attenuationDistance > 0.0) {
		let absorb = -log(attenuationColor) / attenuationDistance;
		volumeAttenuation = exp(-absorb * thickness);
	}

	let f0Scalar = pow((ior - 1.0) / max(ior + 1.0, EPSILON), 2.0);
	let nDotV = max(dot(pbrNormal, viewDir), PBR_MIN_NDOTV);
	let fresnelAverage = clamp(
		f0Scalar + (1.0 - f0Scalar) * pow(max(1.0 - nDotV, 0.0), 5.0),
		0.0,
		1.0
	);
	let coverage = resolveTransmissionAlpha(
		alpha,
		transmission,
		nDotV,
		vec3<f32>(f0Scalar),
		0.0,
		0.0,
		1.0
	);
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

@fragment
fn fsMainTransmissionCapture(input: VertexOutput) -> TransmissionFragmentOutput {
	return shadeTransmissionCapture(input);
}

@fragment
fn fsMainDepthMask(input: VertexOutput) {
	let alphaModeMask = model.materialFlags.y > 0.5;
	if (!alphaModeMask) {
		return;
	}
	let isWireframe = model.materialFlags.w > 0.5;
	var baseSample = vec4<f32>(1.0);
	if (!isWireframe) {
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
	let alpha = clamp(model.baseColorFactor.a * baseSample.a, 0.0, 1.0);
	if (alpha < model.surfaceParams0.w) {
		discard;
	}
}
