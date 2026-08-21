void selectTopTwoLocalLightProbes(
	vec3 worldPosition,
	out ivec2 indices,
	out vec2 rawWeights
) {
	indices = ivec2(-1);
	rawWeights = vec2(0.0);
}

vec4 sampleBlendedLocalLightProbeIrradiance(
	ivec2 indices,
	vec2 rawWeights,
	vec3 normal
) {
	return vec4(0.0);
}

vec4 sampleBlendedLocalLightProbeRadiance(
	ivec2 indices,
	vec2 rawWeights,
	vec3 direction
) {
	return vec4(0.0);
}

vec3 sampleDiffuseProbeIrradiance(vec3 worldPosition, vec3 normal) {
	return calculateIrradianceFromSH(normal);
}
