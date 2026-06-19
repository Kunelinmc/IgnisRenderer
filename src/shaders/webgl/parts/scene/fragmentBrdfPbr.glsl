float distributionGGX(vec3 n, vec3 h, float roughness) {
	float a = roughness * roughness;
	float a2 = a * a;
	float nDotH = max(dot(n, h), 0.0);
	float nDotH2 = nDotH * nDotH;
	float denom = nDotH2 * (a2 - 1.0) + 1.0;
	return a2 / max(PI * denom * denom, 0.0001);
}

float geometrySchlickGGX(float nDotValue, float roughness) {
	float r = roughness + 1.0;
	float k = (r * r) / 8.0;
	return nDotValue / max(nDotValue * (1.0 - k) + k, 0.0001);
}

float geometrySmith(float nDotV, float nDotL, float roughness) {
	return geometrySchlickGGX(nDotV, roughness) *
		geometrySchlickGGX(nDotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
	return f0 + (vec3(1.0) - f0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float fresnelSchlickScalar(float cosTheta, float f0) {
	return f0 + (1.0 - f0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float iorToFresnel0(float transmittedIor, float incidentIor) {
	float value = (transmittedIor - incidentIor) /
		max(transmittedIor + incidentIor, EPSILON);
	return value * value;
}

vec3 fresnel0ToIor(vec3 f0) {
	vec3 sqrtF0 = sqrt(clamp(f0, vec3(0.0), vec3(0.9999)));
	return (vec3(1.0) + sqrtF0) / max(vec3(1.0) - sqrtF0, vec3(EPSILON));
}

vec3 evalIridescenceSensitivity(float opd, vec3 shift) {
	float phase = 2.0 * PI * opd * 1.0e-9;
	float phaseSq = phase * phase;
	vec3 val = vec3(5.4856e-13, 4.4201e-13, 5.2481e-13);
	vec3 pos = vec3(1.6810e6, 1.7953e6, 2.2084e6);
	vec3 variance = vec3(4.3278e9, 9.3046e9, 6.6121e9);
	vec3 xyz =
		val *
		sqrt(vec3(2.0 * PI) * variance) *
		cos(pos * phase + shift) *
		exp(-variance * phaseSq);
	xyz.x +=
		9.7470e-14 *
		sqrt(2.0 * PI * 4.5282e9) *
		cos(2.2399e6 * phase + shift.x) *
		exp(-4.5282e9 * phaseSq);
	xyz /= 1.0685e-7;

	const mat3 xyzToRec709 = mat3(
		3.2404542, -0.9692660, 0.0556434,
		-1.5371385, 1.8760108, -0.2040259,
		-0.4985314, 0.0415560, 1.0572252
	);
	return xyzToRec709 * xyz;
}

vec3 iridescentFresnel(
	float outsideIor,
	float iridescenceIor,
	vec3 baseF0,
	float iridescenceThickness,
	float cosTheta1
) {
	float filmIor = max(iridescenceIor, EPSILON);
	float cos1 = clamp(cosTheta1, 0.0, 1.0);
	float eta = outsideIor / filmIor;
	float sinTheta2Sq = eta * eta * (1.0 - cos1 * cos1);
	if (sinTheta2Sq > 1.0) {
		return vec3(1.0);
	}

	float cosTheta2 = sqrt(max(1.0 - sinTheta2Sq, 0.0));
	float r0 = iorToFresnel0(filmIor, outsideIor);
	float r12 = fresnelSchlickScalar(cos1, r0);
	float t121 = 1.0 - r12;
	vec3 baseIor = fresnel0ToIor(baseF0 + vec3(0.0001));
	vec3 r1 = vec3(
		iorToFresnel0(baseIor.x, filmIor),
		iorToFresnel0(baseIor.y, filmIor),
		iorToFresnel0(baseIor.z, filmIor)
	);
	vec3 r23 = fresnelSchlick(cosTheta2, r1);

	float phi12 = filmIor < outsideIor ? PI : 0.0;
	float phi21 = PI - phi12;
	vec3 phi23 = vec3(
		baseIor.x < filmIor ? PI : 0.0,
		baseIor.y < filmIor ? PI : 0.0,
		baseIor.z < filmIor ? PI : 0.0
	);
	vec3 phi = vec3(phi21) + phi23;
	float opd = 2.0 * filmIor * iridescenceThickness * cosTheta2;
	vec3 r123 = clamp(vec3(r12) * r23, vec3(1e-5), vec3(0.9999));
	vec3 sqrtR123 = sqrt(r123);
	vec3 rs = (t121 * t121) * r23 / (vec3(1.0) - r123);

	vec3 interference = vec3(r12) + rs;
	vec3 cm = rs - vec3(t121);
	for (int order = 1; order <= 2; order++) {
		cm *= sqrtR123;
		float orderValue = float(order);
		vec3 sensitivity = evalIridescenceSensitivity(
			orderValue * opd,
			orderValue * phi
		);
		interference += cm * 2.0 * sensitivity;
	}

	return max(interference, vec3(0.0));
}

vec3 resolveIridescenceFresnel(
	float cosTheta,
	vec3 baseF0,
	float iridescence,
	float iridescenceThickness,
	float iridescenceIor
) {
	vec3 base = fresnelSchlick(cosTheta, baseF0);
	float strength = clamp(iridescence, 0.0, 1.0);
	if (strength <= EPSILON || iridescenceThickness <= 0.0) {
		return base;
	}
	vec3 iridescent = iridescentFresnel(
		1.0,
		max(iridescenceIor, 1.0),
		clamp(baseF0, vec3(0.0), vec3(0.9999)),
		iridescenceThickness,
		cosTheta
	);
	return clamp(mix(base, iridescent, strength), vec3(0.0), vec3(1.0));
}

vec3 diffuseFresnelWeight(vec3 fresnel, float iridescence) {
	if (iridescence > EPSILON) {
		float fresnelMax = max(max(fresnel.r, fresnel.g), fresnel.b);
		return vec3(1.0 - fresnelMax);
	}
	return vec3(1.0) - fresnel;
}

float resolveTransmissionAlpha(
	float baseAlpha,
	float transmission,
	float fresnelAverage
) {
	float clampedTransmission = clamp(transmission, 0.0, 1.0);
	if (clampedTransmission <= EPSILON) {
		return clamp(baseAlpha, 0.0, 1.0);
	}
	float floorAlpha = max(
		TRANSMISSION_ALPHA_FLOOR,
		clamp(fresnelAverage, 0.0, 1.0)
	);
	float blended =
		baseAlpha * (1.0 - clampedTransmission) +
		floorAlpha * clampedTransmission;
	return clamp(max(floorAlpha, blended), 0.0, 1.0);
}
