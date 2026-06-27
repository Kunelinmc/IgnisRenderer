float pointAttenuation(float distanceSq, float range) {
	float rangeSq = max(range * range, EPSILON);
	float rangeFactor = distanceSq / rangeSq;
	float smoothFactor = max(0.0, 1.0 - rangeFactor * rangeFactor);
	return (smoothFactor * smoothFactor) / (distanceSq + 1.0);
}

float spotAttenuation(float cosTheta, float outerCos, float innerCos) {
	if (cosTheta < outerCos) {
		return 0.0;
	}
	float cutoffRange = max(innerCos - outerCos, EPSILON);
	return clamp((cosTheta - outerCos) / cutoffRange, 0.0, 1.0);
}
