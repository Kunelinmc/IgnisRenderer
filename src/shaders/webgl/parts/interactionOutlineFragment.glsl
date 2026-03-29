#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform vec4 uOutlineColor;
uniform vec3 uOutlineParams;
uniform vec2 uViewportSize;
uniform int uCircleCount;
uniform vec4 uCircles[64];

out vec4 fragColor;

float computeShapeDistance(vec2 deltaPixel, int shape) {
	vec2 absDelta = abs(deltaPixel);
	if (shape == 1) {
		return max(absDelta.x, absDelta.y) * 1.41421356237;
	}
	if (shape == 2) {
		return absDelta.x + absDelta.y;
	}
	if (shape == 3) {
		return max(max(absDelta.x, absDelta.y), (absDelta.x + absDelta.y) * 0.70710678118);
	}
	return length(deltaPixel);
}

float computeOutlineMask(vec2 fragmentPixel) {
	float thickness = max(1.0, uOutlineParams.y);
	float feather = max(1.0, thickness * 0.75);
	int shape = int(floor(uOutlineParams.z + 0.5));
	float mask = 0.0;
	for (int i = 0; i < 64; i++) {
		if (i >= uCircleCount) {
			break;
		}
		vec4 circle = uCircles[i];
		float radius = max(0.0, circle.z);
		if (radius <= 0.0) {
			continue;
		}
		float shapeDistance = computeShapeDistance(fragmentPixel - circle.xy, shape);
		float edgeDistance = abs(shapeDistance - radius);
		float edgeAlpha = 1.0 - smoothstep(thickness, thickness + feather, edgeDistance);
		mask = max(mask, edgeAlpha);
	}
	return clamp(mask, 0.0, 1.0);
}

void main() {
	vec4 source = texture(uSourceMap, vUv);
	vec2 fragmentPixel = vec2(
		vUv.x * uViewportSize.x,
		(1.0 - vUv.y) * uViewportSize.y
	);
	float mask = computeOutlineMask(fragmentPixel);
	float alpha = clamp(uOutlineParams.x * mask, 0.0, 1.0);
	vec3 rgb = mix(source.rgb, uOutlineColor.rgb, alpha);
	fragColor = vec4(rgb, source.a);
}
