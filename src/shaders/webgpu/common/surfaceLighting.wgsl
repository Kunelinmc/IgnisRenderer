fn evaluateOpaquePhongLight(
	normal: vec3<f32>,
	viewDirection: vec3<f32>,
	albedo: vec3<f32>,
	specularColor: vec3<f32>,
	shininess: f32,
	lightDirection: vec3<f32>,
	radiance: vec3<f32>,
	visibility: vec3<f32>
) -> vec3<f32> {
	let nDotL = max(dot(normal, lightDirection), 0.0);
	if (nDotL <= 0.0) {
		return vec3<f32>(0.0);
	}
	let halfVector = safeNormalize(viewDirection + lightDirection, viewDirection);
	let vDotH = max(dot(viewDirection, halfVector), 0.0);
	let fresnel = specularColor +
		(vec3<f32>(1.0) - specularColor) * pow(1.0 - vDotH, 5.0);
	let diffuseBRDF = (vec3<f32>(1.0) - fresnel) * albedo / PI;
	let normalizedLobe =
		((max(shininess, 0.0) + 8.0) / (8.0 * PI)) *
		pow(max(dot(normal, halfVector), 0.0), max(shininess, 0.0));
	let specularBRDF = fresnel * normalizedLobe;
	return radiance * visibility * (diffuseBRDF + specularBRDF) * nDotL;
}

struct OpaquePBRSurfaceInput {
	normal: vec3<f32>,
	viewDirection: vec3<f32>,
	albedo: vec3<f32>,
	realF0: vec3<f32>,
	roughness: f32,
	metalness: f32,
	transmission: f32,
	energyCompensation: vec3<f32>,
	iridescence: f32,
	iridescenceIor: f32,
	iridescenceThickness: f32,
	anisotropyTangent: vec3<f32>,
	anisotropyBitangent: vec3<f32>,
	anisotropyStrength: f32,
	clearcoat: f32,
	clearcoatRoughness: f32,
	clearcoatNormal: vec3<f32>,
	sheenColor: vec3<f32>,
	sheenRoughness: f32,
}

fn evaluateOpaquePBRLight(
	surface: OpaquePBRSurfaceInput,
	lightDirection: vec3<f32>,
	radiance: vec3<f32>,
	visibility: vec3<f32>
) -> vec3<f32> {
	let nDotL = max(dot(surface.normal, lightDirection), 0.0);
	if (nDotL <= 0.0) {
		return vec3<f32>(0.0);
	}
	let nDotV = max(dot(surface.normal, surface.viewDirection), PBR_MIN_NDOTV);
	let halfVector = safeNormalize(
		surface.viewDirection + lightDirection,
		surface.viewDirection
	);
	let fresnel = resolveIridescenceFresnel(
		max(dot(halfVector, surface.viewDirection), 0.0),
		surface.realF0,
		surface.iridescence,
		surface.iridescenceThickness,
		surface.iridescenceIor
	);
	var specular = vec3<f32>(0.0);
	if (surface.anisotropyStrength > 1e-6) {
		specular = resolveAnisotropicSpecular(
			fresnel,
			surface.roughness,
			surface.anisotropyStrength,
			nDotL,
			nDotV,
			max(dot(surface.normal, halfVector), 0.0),
			dot(surface.anisotropyTangent, surface.viewDirection),
			dot(surface.anisotropyBitangent, surface.viewDirection),
			dot(surface.anisotropyTangent, lightDirection),
			dot(surface.anisotropyBitangent, lightDirection),
			dot(surface.anisotropyTangent, halfVector),
			dot(surface.anisotropyBitangent, halfVector)
		);
	} else {
		let ndf = distributionGGX(surface.normal, halfVector, surface.roughness);
		let geometry = geometrySmith(nDotV, nDotL, surface.roughness);
		let denominator = max(4.0 * nDotV * nDotL, 0.0001);
		specular = (ndf * geometry * fresnel) / denominator;
	}
	specular *= surface.energyCompensation;
	let kd =
		diffuseFresnelWeight(fresnel, surface.iridescence) *
		(1.0 - surface.metalness) *
		(1.0 - surface.transmission);
	let diffuse = (kd * surface.albedo) / PI;

	var clearcoatSpecular = vec3<f32>(0.0);
	var clearcoatFresnel = vec3<f32>(0.0);
	if (surface.clearcoat > 0.0) {
		let ncDotL = max(dot(surface.clearcoatNormal, lightDirection), 0.0);
		if (ncDotL > 0.0) {
			let ncDotV = max(
				dot(surface.clearcoatNormal, surface.viewDirection),
				PBR_MIN_NDOTV
			);
			let ccHalfVector = safeNormalize(
				surface.viewDirection + lightDirection,
				surface.viewDirection
			);
			let hccDotV = max(dot(ccHalfVector, surface.viewDirection), 0.0);
			let ccNdf = distributionGGX(
				surface.clearcoatNormal,
				ccHalfVector,
				surface.clearcoatRoughness
			);
			let ccGeometry = geometrySmithClearcoat(
				ncDotV,
				ncDotL,
				surface.clearcoatRoughness
			);
			let ccF = fresnelSchlickScalar(hccDotV, 0.04);
			let ccDenominator = max(4.0 * ncDotV * ncDotL, 0.0001);
			clearcoatSpecular = vec3<f32>((ccNdf * ccGeometry * ccF) / ccDenominator);
			clearcoatFresnel = vec3<f32>(ccF);
		}
	}

	let clearcoatAttenuation =
		vec3<f32>(1.0) - clearcoatFresnel * surface.clearcoat;
	var sheenSpecular = vec3<f32>(0.0);
	var albedoSheenScaling = vec3<f32>(1.0);
	if (max(max(surface.sheenColor.x, surface.sheenColor.y), surface.sheenColor.z) > 0.0) {
		let nDotH = max(dot(surface.normal, halfVector), 0.0);
		let sheenNdf = distributionCharlie(nDotH, max(surface.sheenRoughness, 0.04));
		let sheenVisibility = visibilityAshikhmin(nDotL, nDotV);
		sheenSpecular = surface.sheenColor * sheenNdf * sheenVisibility;
		let hDotV = max(dot(halfVector, surface.viewDirection), 0.0);
		let sheenFresnel = fresnelSchlick(hDotV, surface.sheenColor);
		albedoSheenScaling = max(vec3<f32>(0.0), vec3<f32>(1.0) - sheenFresnel);
	}
	let baseLayerAttenuation = clearcoatAttenuation * albedoSheenScaling;
	return (
		(diffuse + specular) * baseLayerAttenuation +
		clearcoatSpecular * surface.clearcoat +
		sheenSpecular * clearcoatAttenuation
	) * nDotL * radiance * visibility;
}
