bool irradianceProbeGridEnabled() {
	return uIrradianceProbeGridEnabled != 0 &&
		uIrradianceProbeGridDataA.w > 0.5;
}

vec3 worldToIrradianceProbeGridPoint(vec3 worldPosition) {
	return vec3(
		dot(uIrradianceProbeGridWorldToGridRow0.xyz, worldPosition) +
			uIrradianceProbeGridWorldToGridRow0.w,
		dot(uIrradianceProbeGridWorldToGridRow1.xyz, worldPosition) +
			uIrradianceProbeGridWorldToGridRow1.w,
		dot(uIrradianceProbeGridWorldToGridRow2.xyz, worldPosition) +
			uIrradianceProbeGridWorldToGridRow2.w
	);
}

float computeIrradianceProbeGridCoverage(vec3 localPosition) {
	vec3 invHalfExtents = uIrradianceProbeGridDataB.xyz;
	float metric = max(
		max(
			abs(localPosition.x) * invHalfExtents.x,
			abs(localPosition.y) * invHalfExtents.y
		),
		abs(localPosition.z) * invHalfExtents.z
	);
	float blendDistance = max(uIrradianceProbeGridDataB.w, 1e-5);
	float x = clamp((metric - 1.0) / blendDistance, 0.0, 1.0);
	return 1.0 - smoothstep(0.0, 1.0, x);
}

float resolveIrradianceProbeGridAxis(
	float localValue,
	float invHalfExtent,
	int dimension
) {
	if (dimension <= 1) {
		return 0.0;
	}
	float normalized = clamp(localValue * invHalfExtent * 0.5 + 0.5, 0.0, 1.0);
	return normalized * float(dimension - 1);
}

int irradianceProbeGridCellIndex(int x, int y, int z, ivec3 dims) {
	return x + y * dims.x + z * dims.x * dims.y;
}

vec4 sampleIrradianceProbeGridCoeff(int cellIndex, int coeffIndex) {
	return texelFetch(uIrradianceProbeGridCoeffs, ivec2(coeffIndex, cellIndex), 0);
}

vec4 sampleIrradianceProbeGridIrradiance(vec3 worldPosition, vec3 normal) {
	if (!irradianceProbeGridEnabled()) {
		return vec4(0.0);
	}
	ivec3 dims = ivec3(
		max(int(floor(uIrradianceProbeGridDataA.x + 0.5)), 1),
		max(int(floor(uIrradianceProbeGridDataA.y + 0.5)), 1),
		max(int(floor(uIrradianceProbeGridDataA.z + 0.5)), 1)
	);
	int cellCount = max(int(floor(uIrradianceProbeGridDataA.w + 0.5)), 1);
	vec3 localPosition = worldToIrradianceProbeGridPoint(worldPosition);
	float coverage = computeIrradianceProbeGridCoverage(localPosition);
	if (coverage <= 1e-6) {
		return vec4(0.0);
	}

	float gridX = resolveIrradianceProbeGridAxis(
		localPosition.x,
		uIrradianceProbeGridDataB.x,
		dims.x
	);
	float gridY = resolveIrradianceProbeGridAxis(
		localPosition.y,
		uIrradianceProbeGridDataB.y,
		dims.y
	);
	float gridZ = resolveIrradianceProbeGridAxis(
		localPosition.z,
		uIrradianceProbeGridDataB.z,
		dims.z
	);
	int x0 = min(int(floor(gridX)), dims.x - 1);
	int y0 = min(int(floor(gridY)), dims.y - 1);
	int z0 = min(int(floor(gridZ)), dims.z - 1);
	int x1 = min(x0 + 1, dims.x - 1);
	int y1 = min(y0 + 1, dims.y - 1);
	int z1 = min(z0 + 1, dims.z - 1);
	float tx = gridX - float(x0);
	float ty = gridY - float(y0);
	float tz = gridZ - float(z0);
	float basis[16];
	evalSHBasis(normal, basis);
	float c1 = PI;
	float c2 = (2.0 * PI) / 3.0;
	float c3 = PI / 4.0;
	vec3 result = vec3(0.0);
	float totalWeight = 0.0;

	for (int corner = 0; corner < 8; corner++) {
		bool useX1 = (corner & 1) != 0;
		bool useY1 = (corner & 2) != 0;
		bool useZ1 = (corner & 4) != 0;
		int cellX = useX1 ? x1 : x0;
		int cellY = useY1 ? y1 : y0;
		int cellZ = useZ1 ? z1 : z0;
		float weightX = useX1 ? tx : 1.0 - tx;
		float weightY = useY1 ? ty : 1.0 - ty;
		float weightZ = useZ1 ? tz : 1.0 - tz;
		float weight = weightX * weightY * weightZ;
		if (weight <= 1e-6) {
			continue;
		}
		int cellIndex = irradianceProbeGridCellIndex(cellX, cellY, cellZ, dims);
		if (
			cellIndex < 0 ||
			cellIndex >= cellCount ||
			cellIndex >= int(floor(uIrradianceProbeGridCoeffsSize.y + 0.5))
		) {
			continue;
		}
		float valid = sampleIrradianceProbeGridCoeff(cellIndex, 0).w;
		if (valid <= 0.5) {
			continue;
		}
		for (int coeffIndex = 0; coeffIndex < SH_COEFFICIENT_COUNT; coeffIndex++) {
			float factor = 0.0;
			if (coeffIndex == 0) {
				factor = c1;
			} else if (coeffIndex >= 1 && coeffIndex < 4) {
				factor = c2;
			} else if (coeffIndex >= 4 && coeffIndex < 9) {
				factor = c3;
			}
			vec3 coeff = sampleIrradianceProbeGridCoeff(cellIndex, coeffIndex).rgb;
			result += coeff * basis[coeffIndex] * factor * weight;
		}
		totalWeight += weight;
	}

	if (totalWeight <= 1e-6) {
		return vec4(0.0);
	}
	return vec4(max(result / totalWeight, vec3(0.0)), clamp(coverage, 0.0, 1.0));
}

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
	vec3 fallback = mix(
		globalAmbientBase,
		localAmbientBase.rgb,
		localAmbientBase.w
	);
	vec4 gridAmbientBase = sampleIrradianceProbeGridIrradiance(
		worldPosition,
		normal
	);
	return mix(fallback, gridAmbientBase.rgb, gridAmbientBase.w);
}
