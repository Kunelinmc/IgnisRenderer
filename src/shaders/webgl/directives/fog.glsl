const int IGNIS_FOG_MODE_LINEAR = 0;
const int IGNIS_FOG_MODE_EXP = 1;
const int IGNIS_FOG_MODE_EXP2 = 2;

float ignisFogLinear(float depth, float startDepth, float endDepth) {
	float safeRange = max(endDepth - startDepth, 1e-4);
	return clamp((depth - startDepth) / safeRange, 0.0, 1.0);
}

float ignisFogExp(float depth, float density) {
	float d = max(depth, 0.0) * max(density, 0.0);
	return 1.0 - exp(-d);
}

float ignisFogExp2(float depth, float density) {
	float d = max(depth, 0.0) * max(density, 0.0);
	return 1.0 - exp(-(d * d));
}

float ignisComputeFogFactor(
	int mode,
	float depth,
	float startDepth,
	float endDepth,
	float density,
	float strength
) {
	float fog = ignisFogLinear(depth, startDepth, endDepth);
	if (mode == IGNIS_FOG_MODE_EXP) {
		fog = ignisFogExp(depth, density);
	} else if (mode == IGNIS_FOG_MODE_EXP2) {
		fog = ignisFogExp2(depth, density);
	}
	return clamp(fog * max(strength, 0.0), 0.0, 1.0);
}
