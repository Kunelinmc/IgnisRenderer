#version 300 es
precision highp float;

in vec2 vUv;
uniform sampler2D uSourceMap;
uniform vec4 uBlurParams; // x: radius, y: sigma, z: width, w: height

out vec4 fragColor;

void main() {
	int radius = int(uBlurParams.x);
	float sigma = uBlurParams.y;
	vec2 texelSize = vec2(1.0 / uBlurParams.z, 1.0 / uBlurParams.w);

	vec3 colorSum = vec3(0.0);
	float weightSum = 0.0;

	// Loop bounds are static constants to satisfy WebGL compiler constraints.
	// Radius check is done dynamically inside the loops.
	for (int dy = -5; dy <= 5; dy++) {
		if (dy < -radius || dy > radius) { continue; }
		for (int dx = -5; dx <= 5; dx++) {
			if (dx < -radius || dx > radius) { continue; }

			float distSq = float(dx * dx + dy * dy);
			float weight = exp(-distSq / (2.0 * sigma * sigma));

			vec2 offset = vec2(float(dx), float(dy)) * texelSize;
			colorSum += texture(uSourceMap, vUv + offset).rgb * weight;
			weightSum += weight;
		}
	}

	vec3 finalColor = colorSum;
	if (weightSum > 0.0) {
		finalColor = colorSum / weightSum;
	}

	fragColor = vec4(finalColor, texture(uSourceMap, vUv).a);
}
