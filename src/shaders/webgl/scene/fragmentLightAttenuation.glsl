float pointAttenuation(float distanceSq, float range) {
	float safeRange = max(range, 0.0001);
	float normalizedDistance = sqrt(max(distanceSq, 0.0)) / safeRange;
	float rangeFactor = clamp(1.0 - pow(normalizedDistance, 4.0), 0.0, 1.0);
	return (rangeFactor * rangeFactor) / max(distanceSq, 0.0001);
}

float spotAttenuation(float cosTheta, float outerCos, float innerCos) {
	return smoothstep(outerCos, max(innerCos, outerCos + EPSILON), cosTheta);
}
