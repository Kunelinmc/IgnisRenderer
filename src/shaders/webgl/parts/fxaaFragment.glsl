#version 300 es
precision highp float;
#import <ignis/postprocess/luma-common>
#import <ignis/postprocess/fxaa>
#define IGNIS_LUMA_PROFILE bt601
#define IGNIS_LUMA_CLAMP false
#inject <ignis/postprocess/luma>(profile=IGNIS_LUMA_PROFILE, clamp=IGNIS_LUMA_CLAMP)

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform vec2 uTexelSize;

out vec4 fragColor;

void main() {
	vec3 rgbNW = texture(uSourceMap, vUv + vec2(-1.0, -1.0) * uTexelSize).rgb;
	vec3 rgbNE = texture(uSourceMap, vUv + vec2(1.0, -1.0) * uTexelSize).rgb;
	vec3 rgbSW = texture(uSourceMap, vUv + vec2(-1.0, 1.0) * uTexelSize).rgb;
	vec3 rgbSE = texture(uSourceMap, vUv + vec2(1.0, 1.0) * uTexelSize).rgb;
	vec3 rgbM = texture(uSourceMap, vUv).rgb;

	float lumaNW = luma(rgbNW);
	float lumaNE = luma(rgbNE);
	float lumaSW = luma(rgbSW);
	float lumaSE = luma(rgbSE);
	float lumaM = luma(rgbM);

	float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
	float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

	vec2 dir;
	dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
	dir.y = (lumaNW + lumaSW) - (lumaNE + lumaSE);

	float dirReduce = max(
		(lumaNW + lumaNE + lumaSW + lumaSE) *
			(0.25 * IGNIS_FXAA_EDGE_THRESHOLD_MIN),
		0.25 * IGNIS_FXAA_EDGE_THRESHOLD_MIN
	);
	float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
	dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * uTexelSize;

	vec3 rgbA = 0.5 * (
		texture(uSourceMap, vUv + dir * (1.0 / 3.0 - 0.5)).rgb +
		texture(uSourceMap, vUv + dir * (2.0 / 3.0 - 0.5)).rgb
	);
	vec3 rgbB = rgbA * 0.5 + 0.25 * (
		texture(uSourceMap, vUv + dir * -0.5).rgb +
		texture(uSourceMap, vUv + dir * 0.5).rgb
	);

	float lumaB = luma(rgbB);
	vec3 filtered =
		(lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
	fragColor = vec4(max(filtered, vec3(0.0)), 1.0);
}
