vec2 directionToEquirectUV(vec3 direction) {
	float phi = atan(direction.x, direction.z);
	float theta = acos(clamp(direction.y, -1.0, 1.0));
	return vec2((phi + PI) / (2.0 * PI), theta / PI);
}

vec3 decodeEnvSpecularSample(vec3 sampled) {
	return uEnvSpecularMapIsLinear == 1 ? sampled : srgbToLinear(sampled);
}

vec3 samplePrefilteredEnvSpecularAtlasMip(
	vec2 uv,
	float layer,
	float layerCount,
	float mipLevel
) {
	ivec2 atlasSize = textureSize(uEnvSpecularMap, int(mipLevel));
	float atlasWidth = float(atlasSize.x);
	float layerWidth = max(atlasWidth / layerCount, 1.0);
	float localX = fract(uv.x) * layerWidth - 0.5;
	float x0 = floor(localX);
	float blend = fract(localX);
	float wrappedX0 = x0 < 0.0 ? layerWidth - 1.0 : x0;
	float wrappedX1 = wrappedX0 + 1.0 >= layerWidth ? 0.0 : wrappedX0 + 1.0;
	float layerOffset = layer * layerWidth;
	vec2 uv0 = vec2((layerOffset + wrappedX0 + 0.5) / atlasWidth, uv.y);
	vec2 uv1 = vec2((layerOffset + wrappedX1 + 0.5) / atlasWidth, uv.y);
	vec3 sample0 = textureLod(uEnvSpecularMap, uv0, mipLevel).rgb;
	vec3 sample1 = textureLod(uEnvSpecularMap, uv1, mipLevel).rgb;
	return mix(sample0, sample1, blend);
}

vec3 samplePrefilteredEnvSpecularLayer(
	vec3 direction,
	float roughness,
	float layer,
	float layerCount
) {
	vec2 uv = directionToEquirectUV(safeNormalize(direction, vec3(0.0, 1.0, 0.0)));
	float mipLevel = clamp(
		roughness * max(uEnvSpecularMaxMipLevel, 0.0),
		0.0,
		max(uEnvSpecularMaxMipLevel, 0.0)
	);
	if (layerCount > 1.0) {
		float level0 = floor(mipLevel);
		float level1 = ceil(mipLevel);
		vec3 sample0 = samplePrefilteredEnvSpecularAtlasMip(
			uv,
			layer,
			layerCount,
			level0
		);
		vec3 sample1 = samplePrefilteredEnvSpecularAtlasMip(
			uv,
			layer,
			layerCount,
			level1
		);
		return mix(sample0, sample1, fract(mipLevel));
	}
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
