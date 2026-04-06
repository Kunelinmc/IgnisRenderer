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

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragMotion;

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
	fragColor = vec4(foggedColor, clamp(color.a, 0.0, 1.0));
	fragMotion = vec4(0, 0, 0, 1);
}
