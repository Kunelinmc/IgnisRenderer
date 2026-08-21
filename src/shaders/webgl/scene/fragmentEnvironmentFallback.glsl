vec3 sampleEnvironmentSpecular(
	vec3 worldPosition,
	vec3 direction,
	float roughness
) {
	if (uHasEnvSpecularMap == 0) {
		return sampleFallbackEnvSpecular(direction, roughness);
	}
	return samplePrefilteredEnvSpecularLayer(direction, roughness, 0.0, 1.0);
}
