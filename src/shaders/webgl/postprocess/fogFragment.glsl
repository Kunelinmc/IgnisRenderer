#version 300 es
precision highp float;

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

in vec2 vUv;
uniform sampler2D uSceneColor;
uniform sampler2D uMotionDepthMap;
uniform vec4 uFogParams0;
uniform vec4 uFogParams1;

out vec4 fragColor;

void main() {
	vec4 scene = texture(uSceneColor, vUv);
	float linearDepth = max(texture(uMotionDepthMap, vUv).z, 0.0);
	int fogMode = int(floor(uFogParams0.x + 0.5));
	float fogFactor = ignisComputeFogFactor(
		fogMode,
		linearDepth,
		uFogParams0.y,
		uFogParams0.z,
		uFogParams0.w,
		uFogParams1.w
	);
	vec3 color = max(mix(scene.rgb, uFogParams1.rgb, fogFactor), vec3(0.0));
	float outputAlpha = clamp(scene.a, 0.0, 1.0) +
		fogFactor * (1.0 - clamp(scene.a, 0.0, 1.0));
	fragColor = vec4(color, outputAlpha);
}
