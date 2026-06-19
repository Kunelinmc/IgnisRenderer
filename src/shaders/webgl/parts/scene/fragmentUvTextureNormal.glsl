vec3 safeNormalize(vec3 value, vec3 fallback) {
	float len = length(value);
	return len > EPSILON ? value / len : fallback;
}

float resolveOITWeight(float alpha, float linearDepth) {
	float clampedAlpha = clamp(alpha, 0.0, 1.0);
	float normalizedDepth = clamp(linearDepth / 400.0, 0.0, 1.0);
	float depthWeight = clamp(1.0 - normalizedDepth, 0.05, 1.0);
	float alphaWeight = max(clampedAlpha * 8.0 + 0.01, 0.01);
	float weight = alphaWeight * alphaWeight * alphaWeight * depthWeight;
	return clamp(weight, 1e-2, 3e3);
}

vec2 resolveUV(int uvSet) {
	if (uvSet == 1) return vUv1;
	if (uvSet == 2) return vUv2;
	if (uvSet >= 3) return vUv3;
	return vUv;
}

vec2 applyUVTransform(vec2 uv, vec4 transformA, vec2 transformB) {
	vec2 scaledUv = vec2(uv.x * transformA.x, uv.y * transformA.y);
	vec2 rotatedUv = vec2(
		scaledUv.x * transformB.x - scaledUv.y * transformB.y,
		scaledUv.x * transformB.y + scaledUv.y * transformB.x
	);
	return rotatedUv + transformA.zw;
}

vec2 resolveMappedUV(int uvSet, vec4 transformA, vec2 transformB) {
	return applyUVTransform(resolveUV(uvSet), transformA, transformB);
}

vec3 fallbackTangentFromNormal(vec3 n) {
	vec3 axis = abs(n.y) > 0.999 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	return safeNormalize(cross(axis, n), vec3(1.0, 0.0, 0.0));
}

bool resolveTangentFrame(vec3 baseNormal, vec4 tangent, out vec3 t, out vec3 b) {
	vec3 n = safeNormalize(baseNormal, vec3(0.0, 0.0, 1.0));
	float tangentLenSq = dot(tangent.xyz, tangent.xyz);
	bool hasValidTangent = tangentLenSq > 1e-12 && abs(tangent.w) > EPSILON;
	if (!hasValidTangent) {
		t = fallbackTangentFromNormal(n);
		b = safeNormalize(cross(n, t), fallbackTangentFromNormal(n));
		return false;
	}

	t = tangent.xyz - n * dot(n, tangent.xyz);
	float tLen = length(t);
	if (tLen <= EPSILON) {
		t = fallbackTangentFromNormal(n);
		b = safeNormalize(cross(n, t), fallbackTangentFromNormal(n));
		return false;
	}

	t /= tLen;
	float handedness = tangent.w < 0.0 ? -1.0 : 1.0;
	b = cross(n, t) * handedness;
	return true;
}

vec3 applyNormalMap(
	vec3 baseNormal,
	vec4 tangent,
	vec3 normalSample,
	float scale
) {
	vec3 n = safeNormalize(baseNormal, vec3(0.0, 0.0, 1.0));
	vec3 t;
	vec3 b;
	if (!resolveTangentFrame(n, tangent, t, b)) {
		return n;
	}

	vec3 tangentNormal = vec3(
		(normalSample.x * 2.0 - 1.0) * scale,
		(normalSample.y * 2.0 - 1.0) * scale,
		normalSample.z * 2.0 - 1.0
	);
	return safeNormalize(
		t * tangentNormal.x + b * tangentNormal.y + n * tangentNormal.z,
		n
	);
}

vec2 rotateAnisotropyDirection(vec2 direction) {
	return vec2(
		direction.x * uAnisotropy.y - direction.y * uAnisotropy.z,
		direction.x * uAnisotropy.z + direction.y * uAnisotropy.y
	);
}

float distributionAnisotropicGGX(float nDotH, float tDotH, float bDotH, float at, float ab) {
	float a2 = max(at * ab, 1e-6);
	vec3 f = vec3(ab * tDotH, at * bDotH, a2 * nDotH);
	float w2 = a2 / max(dot(f, f), 0.0001);
	return a2 * w2 * w2 / PI;
}

float visibilityAnisotropicGGX(
	float nDotL,
	float nDotV,
	float bDotV,
	float tDotV,
	float tDotL,
	float bDotL,
	float at,
	float ab
) {
	float ggxV = nDotL * length(vec3(at * tDotV, ab * bDotV, nDotV));
	float ggxL = nDotV * length(vec3(at * tDotL, ab * bDotL, nDotL));
	return clamp(0.5 / max(ggxV + ggxL, 0.0001), 0.0, 1.0);
}

vec3 resolveAnisotropicSpecular(
	vec3 fresnel,
	float roughness,
	float anisotropy,
	float nDotL,
	float nDotV,
	float nDotH,
	float tDotV,
	float bDotV,
	float tDotL,
	float bDotL,
	float tDotH,
	float bDotH
) {
	float alphaRoughness = roughness * roughness;
	float at = mix(alphaRoughness, 1.0, anisotropy * anisotropy);
	float ab = alphaRoughness;
	float d = distributionAnisotropicGGX(nDotH, tDotH, bDotH, at, ab);
	float v = visibilityAnisotropicGGX(
		nDotL,
		nDotV,
		bDotV,
		tDotV,
		tDotL,
		bDotL,
		at,
		ab
	);
	return fresnel * d * v;
}

vec3 resolveAnisotropicReflectionDirection(
	vec3 n,
	vec3 v,
	vec3 anisotropicB,
	float roughness,
	float anisotropy
) {
	vec3 bentNormal = cross(anisotropicB, v);
	bentNormal = safeNormalize(cross(bentNormal, anisotropicB), n);
	float a = 1.0 - anisotropy * (1.0 - roughness);
	float blendToNormal = a * a * a * a;
	bentNormal = safeNormalize(mix(bentNormal, n, blendToNormal), n);
	vec3 reflectionDir = safeNormalize(reflect(-v, bentNormal), bentNormal);
	reflectionDir = safeNormalize(
		mix(reflectionDir, bentNormal, roughness * roughness),
		reflectionDir
	);
	return reflectionDir;
}

ivec2 linearIndexToTexel(int linearIndex, vec2 textureSizeValue) {
	int width = max(int(floor(textureSizeValue.x + 0.5)), 1);
	int y = linearIndex / width;
	int x = linearIndex - y * width;
	return ivec2(x, y);
}
