#version 300 es
precision highp float;
#import <ignis/color/srgb>
#import <ignis/postprocess/fog>

in vec2 vUv;
in vec4 vColor;
in vec2 vLocalUv;
in float vViewDepth;

uniform sampler2D uParticleMap;
uniform vec4 uUvTransformA;
uniform vec2 uUvTransformB;
uniform int uMapIsLinear;
uniform vec4 uFogParams0;
uniform vec4 uFogParams1;
uniform int uOITPassMode;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragMotion;

float resolveParticleOITWeight(float alpha, float viewDepth) {
	float clampedAlpha = clamp(alpha, 0.0, 1.0);
	float normalizedDepth = clamp(viewDepth / 400.0, 0.0, 1.0);
	float depthWeight = clamp(1.0 - normalizedDepth, 0.05, 1.0);
	float alphaWeight = max(clampedAlpha * 8.0 + 0.01, 0.01);
	float weight = alphaWeight * alphaWeight * alphaWeight * depthWeight;
	return clamp(weight, 1e-2, 3e3);
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

	int fogMode = int(floor(uFogParams0.x + 0.5));
	float fogFactor = ignisComputeFogFactor(
		fogMode,
		max(vViewDepth, 0.0),
		uFogParams0.y,
		uFogParams0.z,
		uFogParams0.w,
		uFogParams1.w
	);
	vec3 foggedColor = max(mix(color.rgb, uFogParams1.rgb, fogFactor), vec3(0.0));
	float finalAlpha = clamp(color.a, 0.0, 1.0);
	if (uOITPassMode == 1) {
		float weight = resolveParticleOITWeight(finalAlpha, max(vViewDepth, 0.0));
		fragColor = vec4(foggedColor * finalAlpha, finalAlpha) * weight;
		fragMotion = vec4(0.0);
		return;
	}
	if (uOITPassMode == 2) {
		fragColor = vec4(finalAlpha);
		fragMotion = vec4(0.0);
		return;
	}
	fragColor = vec4(foggedColor, finalAlpha);
	fragMotion = vec4(0, 0, 0, 1);
}
