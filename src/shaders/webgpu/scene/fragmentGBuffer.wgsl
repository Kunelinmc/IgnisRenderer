@group(3) @binding(0) var gMaterialExt0Out: texture_storage_2d<rgba16float, write>;
@group(3) @binding(1) var gMaterialExt3Out: texture_storage_2d<rgba16uint, write>;

struct GBufferBaseFragmentOutput {
	@location(0) gAlbedoAlpha: vec4<f32>,
	@location(1) gNormalRoughMetal: vec4<f32>,
	@location(2) gEmissiveOcclusion: vec4<f32>,
	@location(3) gMotionDepth: vec4<f32>,
}

fn buildGBufferOutputExtended(
	fragCoord: vec2<f32>,
	alpha: f32,
	albedo: vec3<f32>,
	worldNormal: vec3<f32>,
	roughness: f32,
	metalness: f32,
	emissive: vec3<f32>,
	occlusion: f32,
	motion: vec2<f32>,
	linearDepth: f32,
	materialWord: f32,
	specularData: vec4<f32>,
	coatSheenData: vec4<f32>,
	sheenReflectanceData: vec4<f32>,
	materialExt0: vec4<f32>,
	materialExt3: vec4<u32>,
	nodeRenderLayers: vec2<f32>
) -> GBufferFragmentOutput {
	let coord = vec2<i32>(fragCoord);
	textureStore(gMaterialExt0Out, coord, materialExt0);
	textureStore(gMaterialExt3Out, coord, materialExt3);
	var resolvedMaterialWord = decodeDeferredMaterialWord(materialWord);
	if (nodeRenderLayers.y > 0.5) {
		resolvedMaterialWord =
			resolvedMaterialWord | DEFERRED_MATERIAL_RECEIVE_SHADOWS_BIT;
	} else {
		resolvedMaterialWord =
			resolvedMaterialWord & ~DEFERRED_MATERIAL_RECEIVE_SHADOWS_BIT;
	}

	var output: GBufferFragmentOutput;
	output.gAlbedoAlpha = vec4<f32>(
		clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0)),
		clamp(alpha, 0.0, 1.0)
	);
	output.gNormalRoughMetal = vec4<f32>(
		encodeNormalForGBuffer(worldNormal),
		clamp(roughness, 0.0, 1.0),
		clamp(metalness, 0.0, 1.0)
	);
	output.gEmissiveOcclusion = vec4<f32>(
		clamp(emissive, vec3<f32>(0.0), vec3<f32>(65504.0)),
		clamp(occlusion, 0.0, 1.0)
	);
	output.gMotionDepth = vec4<f32>(
		clamp(motion, vec2<f32>(-1.0), vec2<f32>(1.0)),
		max(linearDepth, 0.0),
		f32(resolvedMaterialWord)
	);
	output.gSpecular = specularData;
	output.gCoatSheen = coatSheenData;
	output.gSheenReflectance = sheenReflectanceData;
	return output;
}

fn buildGBufferOutput(
	fragCoord: vec2<f32>,
	alpha: f32,
	albedo: vec3<f32>,
	worldNormal: vec3<f32>,
	roughness: f32,
	metalness: f32,
	emissive: vec3<f32>,
	occlusion: f32,
	motion: vec2<f32>,
	linearDepth: f32,
	materialWord: f32,
	specularData: vec4<f32>,
	coatSheenData: vec4<f32>,
	sheenReflectanceData: vec4<f32>,
	materialExt0: vec4<f32>,
	nodeRenderLayers: vec2<f32>
) -> GBufferFragmentOutput {
	return buildGBufferOutputExtended(
		fragCoord,
		alpha,
		albedo,
		worldNormal,
		roughness,
		metalness,
		emissive,
		occlusion,
		motion,
		linearDepth,
		materialWord,
		specularData,
		coatSheenData,
		sheenReflectanceData,
		materialExt0,
		packDeferredExt3(
			fallbackTangentFromNormal(worldNormal),
			0.0,
			nodeRenderLayers.x
		),
		nodeRenderLayers
	);
}

fn evaluateGBuffer(
	input: VertexOutput,
	frontFacing: bool
) -> GBufferFragmentOutput {
	let shadingMode = u32(model.materialFlags.x + 0.5);
	let alphaModeMask = model.materialFlags.y > 0.5;
	let doubleSided = model.materialFlags.z > 0.5;
	let enableLighting = frame.options.x > 0.5;

	let isWireframe = model.materialFlags.w > 0.5;
	var baseSample = vec4<f32>(1.0);
	if (
		!isWireframe &&
		(shadingMode != SHADING_PBR ||
			hasPBRTexture(PBR_TEXTURE_BASE_COLOR_MAP))
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

	let baseColor = model.baseColorFactor.rgb * baseSample.rgb;
	let alpha = clamp(model.baseColorFactor.a * baseSample.a, 0.0, 1.0);

	if (alphaModeMask && alpha < model.surfaceParams0.w) {
		discard;
	}

	var normal = safeNormalize(input.worldNormal, vec3<f32>(0.0, 0.0, 1.0));
	if (shadingMode == SHADING_FLAT) {
		let faceNormal = cross(dpdx(input.worldPosition), dpdy(input.worldPosition));
		normal = safeNormalize(faceNormal, normal);
	}
	if (doubleSided && !frontFacing) {
		normal = -normal;
	}
	let geometryNormal = normal;

	var emissiveSample = vec4<f32>(1.0);
	if (
		shadingMode != SHADING_PBR ||
		hasPBRTexture(PBR_TEXTURE_EMISSIVE_MAP)
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
	let emissive =
		model.emissiveFactor.rgb * emissiveSample.rgb * model.emissiveFactor.a;
	let linearDepth = dot(
		frame.cameraPosition.xyz - input.worldPosition,
		frame.environmentBasisBackward.xyz
	);
	let invCurrentW = 1.0 / max(abs(input.currentClip.w), EPSILON);
	let invPrevW = 1.0 / max(abs(input.prevClip.w), EPSILON);
	let currentNdc = input.currentClip.xy * invCurrentW;
	let prevNdc = input.prevClip.xy * invPrevW;
	let motion = currentNdc - prevNdc;

	if (shadingMode == SHADING_UNLIT || !enableLighting) {
		return buildGBufferOutput(
			input.position.xy,
			alpha,
			baseColor,
			normal,
			1.0,
			0.0,
			emissive,
			1.0,
			motion,
			linearDepth,
			f32(SHADING_UNLIT),
			vec4<f32>(0.0),
			vec4<f32>(0.0),
			vec4<f32>(0.0),
			vec4<f32>(encodeNormalForGBuffer(normal), 0.0, 0.0),
			input.instanceMeta
		);
	}

	if (shadingMode == SHADING_PHONG || shadingMode == SHADING_FLAT) {
		let phongAmbient = model.phongAmbientShininess.rgb;
		let phongSpecular = model.phongSpecularShading.rgb;
		let shininess = max(model.phongAmbientShininess.a, 0.0);
		return buildGBufferOutput(
			input.position.xy,
			alpha,
			baseColor,
			normal,
			1.0,
			0.0,
			emissive,
			1.0,
			motion,
			linearDepth,
			f32(shadingMode),
			vec4<f32>(phongSpecular, shininess),
			vec4<f32>(0.0),
			vec4<f32>(phongAmbient, 0.0),
			vec4<f32>(encodeNormalForGBuffer(normal), 0.0, 0.0),
			input.instanceMeta
		);
	}

	var mrSample = vec4<f32>(1.0);
	if (hasPBRTexture(PBR_TEXTURE_METALLIC_ROUGHNESS_MAP)) {
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
	var occlusionSample = vec4<f32>(1.0);
	if (hasPBRTexture(PBR_TEXTURE_OCCLUSION_MAP)) {
		occlusionSample = sampleLinearTexture(
			occlusionTexture,
			occlusionSampler,
			TEX_OCCLUSION,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var specularSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(PBR_FEATURE_SPECULAR) &&
		model.specularColorFactor.a > EPSILON &&
		max(max(model.specularColorFactor.r, model.specularColorFactor.g),
			model.specularColorFactor.b) > EPSILON &&
		hasPBRTexture(PBR_TEXTURE_SPECULAR_MAP)
	) {
		specularSample = sampleLinearTexture(
			specularTexture,
			specularSampler,
			TEX_SPECULAR,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var specularColorSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(PBR_FEATURE_SPECULAR) &&
		model.specularColorFactor.a > EPSILON &&
		max(max(model.specularColorFactor.r, model.specularColorFactor.g),
			model.specularColorFactor.b) > EPSILON &&
		hasPBRTexture(PBR_TEXTURE_SPECULAR_COLOR_MAP)
	) {
		specularColorSample = sampleColorTexture(
			specularColorTexture,
			specularColorSampler,
			TEX_SPECULAR_COLOR,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var clearcoatSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(PBR_FEATURE_CLEARCOAT) &&
		hasPBRTexture(PBR_TEXTURE_CLEARCOAT_MAP)
	) {
		clearcoatSample = sampleLinearTexture(
			clearcoatTexture,
			clearcoatSampler,
			TEX_CLEARCOAT,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var clearcoatRoughnessSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(PBR_FEATURE_CLEARCOAT) &&
		hasPBRTexture(PBR_TEXTURE_CLEARCOAT_ROUGHNESS_MAP)
	) {
		clearcoatRoughnessSample = sampleLinearTexture(
			clearcoatRoughnessTexture,
			clearcoatRoughnessSampler,
			TEX_CLEARCOAT_ROUGHNESS,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var sheenColorSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(PBR_FEATURE_SHEEN) &&
		hasPBRTexture(PBR_TEXTURE_SHEEN_COLOR_MAP)
	) {
		sheenColorSample = sampleColorTexture(
			sheenColorTexture,
			sheenColorSampler,
			TEX_SHEEN_COLOR,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var sheenRoughnessSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(PBR_FEATURE_SHEEN) &&
		hasPBRTexture(PBR_TEXTURE_SHEEN_ROUGHNESS_MAP)
	) {
		sheenRoughnessSample = sampleLinearTexture(
			sheenRoughnessTexture,
			sheenRoughnessSampler,
			TEX_SHEEN_ROUGHNESS,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var iridescenceSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(PBR_FEATURE_IRIDESCENCE) &&
		hasPBRTexture(PBR_TEXTURE_IRIDESCENCE_MAP)
	) {
		iridescenceSample = sampleLinearTexture(
			iridescenceTexture,
			transmissionSampler,
			TEX_IRIDESCENCE,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var iridescenceThicknessSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(PBR_FEATURE_IRIDESCENCE) &&
		hasPBRTexture(PBR_TEXTURE_IRIDESCENCE_THICKNESS_MAP)
	) {
		iridescenceThicknessSample = sampleLinearTexture(
			iridescenceThicknessTexture,
			transmissionSampler,
			TEX_IRIDESCENCE_THICKNESS,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}

	var normalSample = vec3<f32>(0.5, 0.5, 1.0);
	if (hasPBRTexture(PBR_TEXTURE_NORMAL_MAP)) {
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
	var clearcoatNormalSample = vec3<f32>(0.5, 0.5, 1.0);
	if (
		hasPBRFeature(PBR_FEATURE_CLEARCOAT) &&
		hasPBRTexture(PBR_TEXTURE_CLEARCOAT_NORMAL_MAP)
	) {
		clearcoatNormalSample = sampleLinearTexture(
			clearcoatNormalTexture,
			clearcoatNormalSampler,
			TEX_CLEARCOAT_NORMAL,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		).rgb;
	}

	let roughness = clamp(model.surfaceParams0.x * mrSample.g, 0.04, 1.0);
	let metalness = clamp(model.surfaceParams0.y * mrSample.b, 0.0, 1.0);
	let reflectance = clamp(model.surfaceParams0.z, 0.0, 1.0);
	let occlusion = clamp(
		1.0 + model.surfaceParams1.x * (occlusionSample.r - 1.0),
		0.0,
		1.0
	);
	let clearcoat = clamp(model.surfaceParams1.z * clearcoatSample.r, 0.0, 1.0);
	let clearcoatRoughness = clamp(
		model.surfaceParams1.w * clearcoatRoughnessSample.g,
		0.04,
		1.0
	);
	let sheenColor =
		model.sheenColorClearcoatNormalScale.rgb * sheenColorSample.rgb;
	let sheenRoughness =
		clamp(model.surfaceParams2.x * sheenRoughnessSample.a, 0.0, 1.0);
	let iridescence =
		clamp(model.surfaceParams3.y * iridescenceSample.r, 0.0, 1.0);
	let iridescenceIor = max(model.surfaceParams3.z, 1.0);
	let iridescenceThickness = max(
		mix(
			model.surfaceParams3.w,
			model.attenuationColor.a,
			iridescenceThicknessSample.g
		),
		0.0
	);

	let specularFactor =
		clamp(model.specularColorFactor.a * specularSample.a, 0.0, 1.0);
	let specularColor = clamp(
		model.specularColorFactor.rgb * specularColorSample.rgb,
		vec3<f32>(0.0),
		vec3<f32>(1.0)
	);

	var pbrNormal = applyNormalMap(
		normal,
		input.worldTangent,
		normalSample,
		model.surfaceParams1.y
	);
	if (dot(pbrNormal, geometryNormal) < 0.0) {
		pbrNormal = -pbrNormal;
	}

	var clearcoatNormal = applyNormalMap(
		pbrNormal,
		input.worldTangent,
		clearcoatNormalSample,
		model.sheenColorClearcoatNormalScale.a
	);
	if (dot(clearcoatNormal, pbrNormal) < 0.0) {
		clearcoatNormal = -clearcoatNormal;
	}

	let anisotropyData = resolveAnisotropyDirection(
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
	let anisotropyStrength = anisotropyData.z;
	let anisotropyTangent = resolveAnisotropyTangent(
		pbrNormal,
		input.worldTangent,
		anisotropyData.xy
	);

	let albedo = clamp(baseColor, vec3<f32>(0.0), vec3<f32>(1.0));
	let materialWord = encodeDeferredMaterialWord(
		SHADING_PBR,
		clearcoat,
		sheenColor,
		iridescence,
		anisotropyStrength,
		specularColor,
		specularFactor,
		reflectance,
		input.instanceMeta.y
	);
	return buildGBufferOutputExtended(
		input.position.xy,
		alpha,
		albedo,
		pbrNormal,
		roughness,
		metalness,
		emissive,
		occlusion,
		motion,
		linearDepth,
		materialWord,
		vec4<f32>(specularColor, specularFactor),
		vec4<f32>(clearcoat, clearcoatRoughness, sheenRoughness, iridescenceIor),
		vec4<f32>(sheenColor, reflectance),
		vec4<f32>(
			encodeNormalForGBuffer(clearcoatNormal),
			iridescence,
			iridescenceThickness
		),
		packDeferredExt3(
			anisotropyTangent,
			anisotropyStrength,
			input.instanceMeta.x
		),
		input.instanceMeta
	);
}

@fragment
fn fsMainGBuffer(
	input: VertexOutput,
	@builtin(front_facing) frontFacing: bool
) -> GBufferFragmentOutput {
	return evaluateGBuffer(input, frontFacing);
}

@fragment
fn fsMainGBufferBase(
	input: VertexOutput,
	@builtin(front_facing) frontFacing: bool
) -> GBufferBaseFragmentOutput {
	let shadingMode = u32(model.materialFlags.x + 0.5);
	let alphaModeMask = model.materialFlags.y > 0.5;
	let doubleSided = model.materialFlags.z > 0.5;
	var baseSample = vec4<f32>(1.0);
	if (
		shadingMode != SHADING_PBR ||
		hasPBRTexture(PBR_TEXTURE_BASE_COLOR_MAP)
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
	let baseColor = model.baseColorFactor.rgb * baseSample.rgb;
	let alpha = clamp(model.baseColorFactor.a * baseSample.a, 0.0, 1.0);
	if (alphaModeMask && alpha < model.surfaceParams0.w) {
		discard;
	}
	var normal = safeNormalize(input.worldNormal, vec3<f32>(0.0, 0.0, 1.0));
	if (doubleSided && !frontFacing) {
		normal = -normal;
	}
	let geometryNormal = normal;
	var emissiveSample = vec4<f32>(1.0);
	if (
		shadingMode != SHADING_PBR ||
		hasPBRTexture(PBR_TEXTURE_EMISSIVE_MAP)
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
	let emissive =
		model.emissiveFactor.rgb * emissiveSample.rgb * model.emissiveFactor.a;
	let linearDepth = dot(
		frame.cameraPosition.xyz - input.worldPosition,
		frame.environmentBasisBackward.xyz
	);
	let currentNdc = input.currentClip.xy / max(abs(input.currentClip.w), EPSILON);
	let prevNdc = input.prevClip.xy / max(abs(input.prevClip.w), EPSILON);
	let motion = currentNdc - prevNdc;
	var roughness = 1.0;
	var metalness = 0.0;
	var occlusion = 1.0;
	var materialWord = f32(SHADING_UNLIT);
	if (shadingMode == SHADING_PBR && frame.options.x > 0.5) {
		var mrSample = vec4<f32>(1.0);
		if (hasPBRTexture(PBR_TEXTURE_METALLIC_ROUGHNESS_MAP)) {
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
		var occlusionSample = vec4<f32>(1.0);
		if (hasPBRTexture(PBR_TEXTURE_OCCLUSION_MAP)) {
			occlusionSample = sampleLinearTexture(
				occlusionTexture,
				occlusionSampler,
				TEX_OCCLUSION,
				input.uv0,
				input.uv1,
				input.uv2,
				input.uv3
			);
		}
		var normalSample = vec3<f32>(0.5, 0.5, 1.0);
		if (hasPBRTexture(PBR_TEXTURE_NORMAL_MAP)) {
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
		roughness = clamp(model.surfaceParams0.x * mrSample.g, 0.04, 1.0);
		metalness = clamp(model.surfaceParams0.y * mrSample.b, 0.0, 1.0);
		occlusion = clamp(
			1.0 + model.surfaceParams1.x * (occlusionSample.r - 1.0),
			0.0,
			1.0
		);
		normal = applyNormalMap(
			normal,
			input.worldTangent,
			normalSample,
			model.surfaceParams1.y
		);
		if (dot(normal, geometryNormal) < 0.0) {
			normal = -normal;
		}
		materialWord = f32(SHADING_PBR);
	}
	if (input.instanceMeta.y > 0.5) {
		materialWord = f32(
			u32(materialWord) | DEFERRED_MATERIAL_RECEIVE_SHADOWS_BIT
		);
	}
	var output: GBufferBaseFragmentOutput;
	output.gAlbedoAlpha = vec4<f32>(
		clamp(baseColor, vec3<f32>(0.0), vec3<f32>(1.0)),
		alpha
	);
	output.gNormalRoughMetal = vec4<f32>(
		encodeNormalForGBuffer(normal),
		roughness,
		metalness
	);
	output.gEmissiveOcclusion = vec4<f32>(
		clamp(emissive, vec3<f32>(0.0), vec3<f32>(65504.0)),
		occlusion
	);
	output.gMotionDepth = vec4<f32>(
		clamp(motion, vec2<f32>(-1.0), vec2<f32>(1.0)),
		max(linearDepth, 0.0),
		materialWord
	);
	return output;
}
