#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform vec4 uOutlineColor;
uniform vec2 uOutlineParams;
uniform vec2 uViewportSize;
uniform int uCircleCount;
uniform vec4 uCircles[64];

out vec4 fragColor;

float computeOutlineMask(vec2 fragmentPixel) {
	float thickness = max(1.0, uOutlineParams.y);
	float feather = max(1.0, thickness * 0.75);
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
		float edgeDistance = abs(length(fragmentPixel - circle.xy) - radius);
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
