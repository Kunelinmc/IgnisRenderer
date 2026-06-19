int getLocalLightProbeCount() {
	return clamp(uLocalLightProbeCount, 0, MAX_LOCAL_LIGHT_PROBES);
}

vec3 worldToLocalLightProbePoint(int probeIndex, vec3 worldPosition) {
	vec4 row0 = uLocalLightProbeWorldToProbeRow0[probeIndex];
	vec4 row1 = uLocalLightProbeWorldToProbeRow1[probeIndex];
	vec4 row2 = uLocalLightProbeWorldToProbeRow2[probeIndex];
	return vec3(
		dot(row0.xyz, worldPosition) + row0.w,
		dot(row1.xyz, worldPosition) + row1.w,
		dot(row2.xyz, worldPosition) + row2.w
	);
}

float computeLocalLightProbeMetric(int probeIndex, vec3 worldPosition) {
	vec3 localPosition = worldToLocalLightProbePoint(probeIndex, worldPosition);
	vec4 dataA = uLocalLightProbeDataA[probeIndex];
	float shape = uLocalLightProbeDataB[probeIndex].z;
	if (shape > 0.5) {
		return max(
			max(abs(localPosition.x) * dataA.x, abs(localPosition.y) * dataA.y),
			abs(localPosition.z) * dataA.z
		);
	}
	return length(localPosition) * dataA.w;
}

float computeLocalLightProbeWeight(int probeIndex, float metric) {
	float blendDistance = max(uLocalLightProbeDataB[probeIndex].x, 1e-5);
	float x = clamp((metric - 1.0) / blendDistance, 0.0, 1.0);
	return 1.0 - smoothstep(0.0, 1.0, x);
}

int getLocalLightProbePriority(int probeIndex) {
	return int(uLocalLightProbeDataB[probeIndex].y);
}

bool isBetterLocalLightProbeCandidate(
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

vec3 sampleLocalLightProbeCoeff(int probeIndex, int coeffIndex) {
	ivec2 texel = linearIndexToTexel(
		probeIndex * SH_COEFFICIENT_COUNT + coeffIndex,
		uLocalLightProbeCoeffsSize
	);
	return texelFetch(uLocalLightProbeCoeffs, texel, 0).rgb;
}

vec3 calculateIrradianceFromLocalLightProbe(int probeIndex, vec3 normal) {
	float basis[16];
	evalSHBasis(normal, basis);
	float c1 = PI;
	float c2 = (2.0 * PI) / 3.0;
	float c3 = PI / 4.0;
	vec3 result = vec3(0.0);
	for (int i = 0; i < SH_COEFFICIENT_COUNT; i++) {
		float factor = 0.0;
		if (i == 0) {
			factor = c1;
		} else if (i >= 1 && i < 4) {
			factor = c2;
		} else if (i >= 4 && i < 9) {
			factor = c3;
		}
		result += sampleLocalLightProbeCoeff(probeIndex, i) * basis[i] * factor;
	}
	return max(result, vec3(0.0));
}

vec3 sampleLocalLightProbeRadiance(int probeIndex, vec3 direction) {
	float basis[16];
	evalSHBasis(direction, basis);
	vec3 result = vec3(0.0);
	for (int i = 0; i < SH_COEFFICIENT_COUNT; i++) {
		result += sampleLocalLightProbeCoeff(probeIndex, i) * basis[i];
	}
	return max(result, vec3(0.0));
}

void selectTopTwoLocalLightProbes(
	vec3 worldPosition,
	out ivec2 indices,
	out vec2 rawWeights
) {
	indices = ivec2(-1);
	rawWeights = vec2(0.0);
	int probeCount = getLocalLightProbeCount();
	int bestPriority = -2147483647;

	for (int i = 0; i < MAX_LOCAL_LIGHT_PROBES; i++) {
		if (i >= probeCount) {
			break;
		}

		float metric = computeLocalLightProbeMetric(i, worldPosition);
		float weight = computeLocalLightProbeWeight(i, metric);
		if (weight <= 1e-6) {
			continue;
		}

		int priority = getLocalLightProbePriority(i);
		if (priority > bestPriority) {
			bestPriority = priority;
			indices = ivec2(i, -1);
			rawWeights = vec2(weight, 0.0);
			continue;
		}
		if (priority < bestPriority) {
			continue;
		}

		if (
			indices.x < 0 ||
			isBetterLocalLightProbeCandidate(weight, i, rawWeights.x, indices.x)
		) {
			indices.y = indices.x;
			rawWeights.y = rawWeights.x;
			indices.x = i;
			rawWeights.x = weight;
			continue;
		}

		if (
			indices.y < 0 ||
			isBetterLocalLightProbeCandidate(weight, i, rawWeights.y, indices.y)
		) {
			indices.y = i;
			rawWeights.y = weight;
		}
	}
}

vec4 sampleBlendedLocalLightProbeIrradiance(
	ivec2 indices,
	vec2 rawWeights,
	vec3 normal
) {
	float rawSum = rawWeights.x + max(rawWeights.y, 0.0);
	float coverage = clamp(rawSum, 0.0, 1.0);
	if (indices.x < 0 || coverage <= 1e-6) {
		return vec4(0.0);
	}

	float invWeight = 1.0 / max(rawSum, 1e-6);
	vec3 result =
		calculateIrradianceFromLocalLightProbe(indices.x, normal) *
		(rawWeights.x * invWeight);
	if (indices.y >= 0 && rawWeights.y > 1e-6) {
		result +=
			calculateIrradianceFromLocalLightProbe(indices.y, normal) *
			(rawWeights.y * invWeight);
	}
	return vec4(result, coverage);
}

vec4 sampleBlendedLocalLightProbeRadiance(
	ivec2 indices,
	vec2 rawWeights,
	vec3 direction
) {
	float rawSum = rawWeights.x + max(rawWeights.y, 0.0);
	float coverage = clamp(rawSum, 0.0, 1.0);
	if (indices.x < 0 || coverage <= 1e-6) {
		return vec4(0.0);
	}

	float invWeight = 1.0 / max(rawSum, 1e-6);
	vec3 result =
		sampleLocalLightProbeRadiance(indices.x, direction) *
		(rawWeights.x * invWeight);
	if (indices.y >= 0 && rawWeights.y > 1e-6) {
		result +=
			sampleLocalLightProbeRadiance(indices.y, direction) *
			(rawWeights.y * invWeight);
	}
	return vec4(result, coverage);
}

__WEBGL_IRRADIANCE_PROBE_GRID_FUNCTIONS__
