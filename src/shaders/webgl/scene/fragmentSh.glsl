vec3 sampleSHAmbientCoeff(int index) {
	return uSHAmbientCoeffs[index];
}

void evalSHBasis(vec3 direction, out float basis[16]) {
	float x = direction.x;
	float y = direction.y;
	float z = direction.z;

	basis[0] = 0.282095;
	basis[1] = 0.488603 * x;
	basis[2] = 0.488603 * y;
	basis[3] = 0.488603 * z;
	basis[4] = 1.092548 * x * z;
	basis[5] = 1.092548 * x * y;
	basis[6] = 0.315392 * (3.0 * y * y - 1.0);
	basis[7] = 1.092548 * y * z;
	basis[8] = 0.546274 * (x * x - z * z);
	basis[9] = 0.590835 * x * (x * x - 3.0 * z * z);
	basis[10] = 2.893641 * x * y * z;
	basis[11] = 0.457619 * x * (5.0 * y * y - 1.0);
	basis[12] = 0.373176 * y * (5.0 * y * y - 3.0);
	basis[13] = 0.457619 * z * (5.0 * y * y - 1.0);
	basis[14] = 1.446821 * y * (x * x - z * z);
	basis[15] = 0.590835 * z * (3.0 * x * x - z * z);
}

vec3 calculateIrradianceFromSH(vec3 normal) {
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
		result += sampleSHAmbientCoeff(i) * basis[i] * factor;
	}
	return max(result, vec3(0.0));
}

vec3 sampleSHRadiance(vec3 direction) {
	float basis[16];
	evalSHBasis(direction, basis);
	vec3 result = vec3(0.0);
	for (int i = 0; i < SH_COEFFICIENT_COUNT; i++) {
		result += sampleSHAmbientCoeff(i) * basis[i];
	}
	return max(result, vec3(0.0));
}
