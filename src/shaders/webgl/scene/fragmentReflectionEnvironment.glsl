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
	vec4 row0 = uReflectionProbeWorldToProbeRow0[probeIndex];
	vec4 row1 = uReflectionProbeWorldToProbeRow1[probeIndex];
	vec4 row2 = uReflectionProbeWorldToProbeRow2[probeIndex];
	return vec3(
		dot(row0.xyz, worldPosition) + row0.w,
		dot(row1.xyz, worldPosition) + row1.w,
		dot(row2.xyz, worldPosition) + row2.w
	);
}

vec3 worldToProbeDirection(int probeIndex, vec3 worldDirection) {
	vec4 row0 = uReflectionProbeWorldToProbeRow0[probeIndex];
	vec4 row1 = uReflectionProbeWorldToProbeRow1[probeIndex];
	vec4 row2 = uReflectionProbeWorldToProbeRow2[probeIndex];
	return vec3(
		dot(row0.xyz, worldDirection),
		dot(row1.xyz, worldDirection),
		dot(row2.xyz, worldDirection)
	);
}

vec3 probeToWorldPoint(int probeIndex, vec3 probePosition) {
	vec4 row0 = uReflectionProbeProbeToWorldRow0[probeIndex];
	vec4 row1 = uReflectionProbeProbeToWorldRow1[probeIndex];
	vec4 row2 = uReflectionProbeProbeToWorldRow2[probeIndex];
	return vec3(
		dot(row0.xyz, probePosition) + row0.w,
		dot(row1.xyz, probePosition) + row1.w,
		dot(row2.xyz, probePosition) + row2.w
	);
}

float computeReflectionProbeMetric(int probeIndex, vec3 worldPosition) {
	vec3 localPosition = worldToProbePoint(probeIndex, worldPosition);
	vec4 dataA = uReflectionProbeDataA[probeIndex];
	float shape = uReflectionProbeDataB[probeIndex].w;
	if (shape > 0.5) {
		return max(
			max(abs(localPosition.x) * dataA.x, abs(localPosition.y) * dataA.y),
			abs(localPosition.z) * dataA.z
		);
	}
	return length(localPosition) * dataA.w;
}

float computeReflectionProbeWeight(int probeIndex, float metric) {
	float blendDistance = max(uReflectionProbeDataC[probeIndex].y, 1e-5);
	float x = clamp((metric - 1.0) / blendDistance, 0.0, 1.0);
	float weight = 1.0 - smoothstep(0.0, 1.0, x);
	float blendExponent = max(uReflectionProbeDataC[probeIndex].z, 0.01);
	if (abs(blendExponent - 1.0) > 1e-5) {
		weight = pow(max(weight, 0.0), blendExponent);
	}
	return weight;
}

float computeReflectionProbeDepthOcclusion(int probeIndex, float metric) {
	float blendDistance = max(uReflectionProbeDataC[probeIndex].y, 1e-5);
	float normalizedDepth = clamp((1.0 - metric) / blendDistance, 0.0, 1.0);
	return smoothstep(0.0, 1.0, normalizedDepth);
}

bool isBetterReflectionProbeCandidate(
	float candidateWeight,
	int candidateIndex,
	float currentWeight,
	int currentIndex
) {
	if (candidateWeight > currentWeight + 1e-6) {
		return true;
	}
	if (abs(candidateWeight - currentWeight) <= 1e-6 && candidateIndex < currentIndex) {
		return true;
	}
	return false;
}

void selectTopTwoReflectionProbes(
	vec3 worldPosition,
	int probeCount,
	out ivec2 indices,
	out vec2 weights
) {
	indices = ivec2(-1);
	weights = vec2(0.0);

	for (int i = 0; i < MAX_REFLECTION_PROBES; i++) {
		if (i >= probeCount) {
			break;
		}
		float metric = computeReflectionProbeMetric(i, worldPosition);
		float weight = computeReflectionProbeWeight(i, metric);
		if (weight <= 1e-6) {
			continue;
		}

		if (
			indices.x < 0 ||
			isBetterReflectionProbeCandidate(weight, i, weights.x, indices.x)
		) {
			indices.y = indices.x;
			weights.y = weights.x;
			indices.x = i;
			weights.x = weight;
			continue;
		}

		if (
			indices.y < 0 ||
			isBetterReflectionProbeCandidate(weight, i, weights.y, indices.y)
		) {
			indices.y = i;
			weights.y = weight;
		}
	}

	if (indices.x < 0) {
		weights = vec2(0.0);
		return;
	}

	float sumWeight = weights.x + max(weights.y, 0.0);
	if (sumWeight <= 1e-6) {
		indices.y = -1;
		weights = vec2(1.0, 0.0);
		return;
	}

	if (indices.y < 0) {
		weights = vec2(1.0, 0.0);
		return;
	}

	weights /= sumWeight;
}

bool intersectReflectionProbeBox(
	int probeIndex,
	vec3 localOrigin,
	vec3 localDirection,
	out vec3 localHit
) {
	vec4 dataA = uReflectionProbeDataA[probeIndex];
	vec3 halfExtents = vec3(
		1.0 / max(dataA.x, 1e-5),
		1.0 / max(dataA.y, 1e-5),
		1.0 / max(dataA.z, 1e-5)
	);
	float tMin = -1e20;
	float tMax = 1e20;

	if (abs(localDirection.x) <= EPSILON) {
		if (localOrigin.x < -halfExtents.x || localOrigin.x > halfExtents.x) {
			return false;
		}
	} else {
		float invDirection = 1.0 / localDirection.x;
		float t0 = (-halfExtents.x - localOrigin.x) * invDirection;
		float t1 = (halfExtents.x - localOrigin.x) * invDirection;
		if (t0 > t1) {
			float swap = t0;
			t0 = t1;
			t1 = swap;
		}
		tMin = max(tMin, t0);
		tMax = min(tMax, t1);
	}

	if (abs(localDirection.y) <= EPSILON) {
		if (localOrigin.y < -halfExtents.y || localOrigin.y > halfExtents.y) {
			return false;
		}
	} else {
		float invDirection = 1.0 / localDirection.y;
		float t0 = (-halfExtents.y - localOrigin.y) * invDirection;
		float t1 = (halfExtents.y - localOrigin.y) * invDirection;
		if (t0 > t1) {
			float swap = t0;
			t0 = t1;
			t1 = swap;
		}
		tMin = max(tMin, t0);
		tMax = min(tMax, t1);
	}

	if (abs(localDirection.z) <= EPSILON) {
		if (localOrigin.z < -halfExtents.z || localOrigin.z > halfExtents.z) {
			return false;
		}
	} else {
		float invDirection = 1.0 / localDirection.z;
		float t0 = (-halfExtents.z - localOrigin.z) * invDirection;
		float t1 = (halfExtents.z - localOrigin.z) * invDirection;
		if (t0 > t1) {
			float swap = t0;
			t0 = t1;
			t1 = swap;
		}
		tMin = max(tMin, t0);
		tMax = min(tMax, t1);
	}

	if (tMax < max(tMin, 0.0)) {
		return false;
	}
	float t = tMin > EPSILON ? tMin : tMax;
	if (t <= EPSILON) {
		return false;
	}
	localHit = localOrigin + localDirection * t;
	return true;
}

bool intersectReflectionProbeSphere(
	int probeIndex,
	vec3 localOrigin,
	vec3 localDirection,
	out vec3 localHit
) {
	float radius = 1.0 / max(uReflectionProbeDataA[probeIndex].w, 1e-5);
	float b = dot(localOrigin, localDirection);
	float c = dot(localOrigin, localOrigin) - radius * radius;
	float discriminant = b * b - c;
	if (discriminant < 0.0) {
		return false;
	}

	float sqrtDiscriminant = sqrt(discriminant);
	float t0 = -b - sqrtDiscriminant;
	float t1 = -b + sqrtDiscriminant;
	float t = 1e20;
	if (t0 > EPSILON) {
		t = min(t, t0);
	}
	if (t1 > EPSILON) {
		t = min(t, t1);
	}
	if (t >= 1e19) {
		return false;
	}

	localHit = localOrigin + localDirection * t;
	return true;
}

vec3 computeReflectionProbeParallaxDirection(
	int probeIndex,
	vec3 worldPosition,
	vec3 reflectionDirection
) {
	vec3 fallback = safeNormalize(reflectionDirection, vec3(0.0, 0.0, 1.0));
	int parallaxMode = int(floor(uReflectionProbeDataC[probeIndex].x + 0.5));
	if (parallaxMode <= 0) {
		return fallback;
	}

	vec3 localOrigin = worldToProbePoint(probeIndex, worldPosition);
	vec3 localDirection = safeNormalize(
		worldToProbeDirection(probeIndex, fallback),
		fallback
	);
	vec3 localHit = vec3(0.0);
	bool hasHit = false;
	if (parallaxMode == 1) {
		hasHit = intersectReflectionProbeBox(
			probeIndex,
			localOrigin,
			localDirection,
			localHit
		);
	} else if (parallaxMode == 2) {
		hasHit = intersectReflectionProbeSphere(
			probeIndex,
			localOrigin,
			localDirection,
			localHit
		);
	}
	if (!hasHit) {
		return fallback;
	}

	vec3 worldHit = probeToWorldPoint(probeIndex, localHit);
	vec3 corrected = worldHit - uReflectionProbeDataB[probeIndex].xyz;
	return dot(corrected, corrected) > EPSILON ?
		normalize(corrected)
	:	fallback;
}

vec3 sampleEnvironmentSpecular(vec3 worldPosition, vec3 direction, float roughness) {
	if (uHasEnvSpecularMap == 0) {
		return vec3(0.0);
	}

	int probeCount = clamp(uReflectionProbeCount, 0, MAX_REFLECTION_PROBES);
	vec3 normalizedDirection = safeNormalize(direction, vec3(0.0, 1.0, 0.0));
	if (probeCount <= 0) {
		return samplePrefilteredEnvSpecularLayer(normalizedDirection, roughness, 0.0, 1.0);
	}

	ivec2 indices;
	vec2 weights;
	selectTopTwoReflectionProbes(worldPosition, probeCount, indices, weights);
	vec3 fallbackSample = sampleFallbackEnvSpecular(normalizedDirection, roughness);
	if (indices.x < 0) {
		return fallbackSample;
	}

	int firstIndex = indices.x;
	float firstLayer = uReflectionProbeDataC[firstIndex].w;
	vec3 firstDirection = computeReflectionProbeParallaxDirection(
		firstIndex,
		worldPosition,
		normalizedDirection
	);
	vec3 firstSample = samplePrefilteredEnvSpecularLayer(
		firstDirection,
		roughness,
		firstLayer,
		float(probeCount)
	);
	float firstMetric = computeReflectionProbeMetric(firstIndex, worldPosition);
	float firstDepthOcclusion = computeReflectionProbeDepthOcclusion(
		firstIndex,
		firstMetric
	);
	float firstContribution = weights.x * firstDepthOcclusion;

	if (indices.y < 0 || weights.y <= 1e-6) {
		return firstSample * firstContribution +
			fallbackSample * (1.0 - clamp(firstContribution, 0.0, 1.0));
	}

	int secondIndex = indices.y;
	float secondLayer = uReflectionProbeDataC[secondIndex].w;
	vec3 secondDirection = computeReflectionProbeParallaxDirection(
		secondIndex,
		worldPosition,
		normalizedDirection
	);
	vec3 secondSample = samplePrefilteredEnvSpecularLayer(
		secondDirection,
		roughness,
		secondLayer,
		float(probeCount)
	);
	float secondMetric = computeReflectionProbeMetric(secondIndex, worldPosition);
	float secondDepthOcclusion = computeReflectionProbeDepthOcclusion(
		secondIndex,
		secondMetric
	);
	float secondContribution = weights.y * secondDepthOcclusion;
	float combinedContribution = clamp(
		firstContribution + secondContribution,
		0.0,
		1.0
	);

	return
		firstSample * firstContribution +
		secondSample * secondContribution +
		fallbackSample * (1.0 - combinedContribution);
}
