vec2 directionToEquirectUV(vec3 direction) {
	float phi = atan(direction.x, direction.z);
	float theta = acos(clamp(direction.y, -1.0, 1.0));
	return vec2((phi + PI) / (2.0 * PI), theta / PI);
}

vec3 decodeEnvSpecularSample(vec3 sampled) {
	return uEnvSpecularMapIsLinear == 1 ? sampled : srgbToLinear(sampled);
}

vec3 samplePrefilteredEnvSpecularLayer(
	vec3 direction,
	float roughness,
	float layer,
	float layerCount
) {
	vec2 uv = directionToEquirectUV(safeNormalize(direction, vec3(0.0, 1.0, 0.0)));
	if (layerCount > 1.0) {
		uv.x = (uv.x + layer) / layerCount;
	}
	float mipLevel = clamp(
		roughness * max(uEnvSpecularMaxMipLevel, 0.0),
		0.0,
		max(uEnvSpecularMaxMipLevel, 0.0)
	);
	vec3 sampled = textureLod(uEnvSpecularMap, uv, mipLevel).rgb;
	return decodeEnvSpecularSample(sampled);
}

vec3 sampleFallbackEnvSpecular(vec3 direction, float roughness) {
	if (uHasEnvSpecularFallbackMap == 0) {
		return vec3(0.0);
	}

	vec2 uv = directionToEquirectUV(safeNormalize(direction, vec3(0.0, 1.0, 0.0)));
	float mipLevel = clamp(
		roughness * max(uEnvSpecularFallbackMaxMipLevel, 0.0),
		0.0,
		max(uEnvSpecularFallbackMaxMipLevel, 0.0)
	);
	vec3 sampled = textureLod(uEnvSpecularFallbackMap, uv, mipLevel).rgb;
	return uEnvSpecularFallbackMapIsLinear == 1 ?
		sampled
	:	srgbToLinear(sampled);
}

vec3 worldToProbePoint(int probeIndex, vec3 worldPosition) {
