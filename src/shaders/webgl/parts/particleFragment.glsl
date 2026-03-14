#version 300 es
precision highp float;

in vec2 vUv;
in vec4 vColor;
in vec2 vLocalUv;

uniform sampler2D uParticleMap;
uniform vec4 uUvTransformA;
uniform vec2 uUvTransformB;
uniform int uMapIsLinear;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragMotion;

vec3 srgbToLinear(vec3 c) {
	vec3 a = c / 12.92;
	vec3 b = pow((c + 0.055) / 1.055, vec3(2.4));
	return mix(b, a, lessThanEqual(c, vec3(0.04045)));
}

void main() {
	float radialDistance = distance(vLocalUv, vec2(0.5, 0.5));
	float radialMask = 1.0 - smoothstep(0.4, 0.5, radialDistance);

	vec2 scaledUv = vUv * uUvTransformA.xy;
	vec2 rotatedUv = vec2(
		scaledUv.x * uUvTransformB.x - scaledUv.y * uUvTransformB.y,
		scaledUv.x * uUvTransformB.y + scaledUv.y * uUvTransformB.x
	);
	vec2 finalUv = rotatedUv + uUvTransformA.zw;
	vec4 sampled = texture(uParticleMap, finalUv);
	if (uMapIsLinear == 0) {
		sampled.rgb = srgbToLinear(sampled.rgb);
	}

	vec4 color = sampled * vColor;
	color.a *= radialMask;
	if (color.a <= 0.001) {
		discard;
	}

	fragColor = vec4(max(color.rgb, vec3(0.0)), clamp(color.a, 0.0, 1.0));
	fragMotion = vec4(0, 0, 0, 1);
}