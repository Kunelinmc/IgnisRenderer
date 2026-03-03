export const WEBGPU_SCENE_FRAGMENT_PBR_AMBIENT = /* wgsl */ `
	var ambientColor = frame.ambientColor.rgb;
	if (ambientColor.x + ambientColor.y + ambientColor.z == 0.0) {
		ambientColor = vec3<f32>(PBR_AMBIENT_FALLBACK_LINEAR);
	}

	let ambientRadiance = ambientColor / PI;
	let fAmbient = fresnelSchlick(nDotV, realF0);
	let refractionResult = refractViewDirection(viewDir, pbrNormal, ior);
	let isTIR = transmission > 0.0 && refractionResult.valid < 0.5;
	let effectiveFAmbient = select(fAmbient, vec3<f32>(1.0), isTIR);
	let kdAmbient =
		(vec3<f32>(1.0) - effectiveFAmbient) *
		(1.0 - metalness) *
		(1.0 - transmission);
	let ktAmbient =
		(vec3<f32>(1.0) - effectiveFAmbient) *
		(1.0 - metalness) *
		transmission;

	let ccAmbientFresnel = select(0.0, fresnelSchlickScalar(nDotV, 0.04), clearcoat > 0.0);
	let clearcoatAmbientAttenuation = 1.0 - ccAmbientFresnel * clearcoat;
	let baseAmbientAttenuation =
		vec3<f32>(clearcoatAmbientAttenuation) * (vec3<f32>(1.0) - sheenColor * 0.5);

	var ambientLight = ambientColor * albedo * kdAmbient * baseAmbientAttenuation;
	if (transmission > 0.0 && refractionResult.valid > 0.5) {
		ambientLight +=
			ambientColor *
			albedo *
			ktAmbient *
			volumeAttenuation *
			clearcoatAmbientAttenuation;
	}

	let specularAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - roughness) * 0.5);
	let clearcoatAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - clearcoatRoughness) * 0.5);
	ambientLight +=
		ambientRadiance *
		effectiveFAmbient *
		specularAmbientFactor *
		clearcoatAmbientAttenuation;
	ambientLight += ambientRadiance * ccAmbientFresnel * clearcoatAmbientFactor * clearcoat;

	if (maxSheenColor > 0.0) {
		let sheenAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - max(sheenRoughness, 0.04)) * 0.5);
		ambientLight +=
			ambientRadiance *
			sheenColor *
			sheenAmbientFactor *
			clearcoatAmbientAttenuation;
	}

	ambientLight *= occlusion;

	let finalLinear = max(directLight + ambientLight + emissive, vec3<f32>(0.0));
	let outputColor = encodeOutput(finalLinear);
	return vec4<f32>(clamp(outputColor, vec3<f32>(0.0), vec3<f32>(1.0)), alpha);
}
`
