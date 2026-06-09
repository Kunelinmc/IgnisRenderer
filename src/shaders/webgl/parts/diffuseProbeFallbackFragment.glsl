vec3 sampleDiffuseProbeIrradiance(vec3 worldPosition, vec3 normal) {
	ivec2 localProbeIndices;
	vec2 localProbeWeights;
	selectTopTwoLocalLightProbes(worldPosition, localProbeIndices, localProbeWeights);
	vec3 globalAmbientBase = calculateIrradianceFromSH(normal);
	vec4 localAmbientBase = sampleBlendedLocalLightProbeIrradiance(
		localProbeIndices,
		localProbeWeights,
		normal
	);
	return mix(
		globalAmbientBase,
		localAmbientBase.rgb,
		localAmbientBase.w
	);
}
