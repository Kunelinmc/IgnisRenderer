#version 300 es
precision highp float;

in vec2 vUv;
uniform sampler2D uSceneColor;
uniform sampler2D uDepthMap;
uniform sampler2D uNormalMap;
uniform vec2 uTexelSize;

out vec4 fragColor;

float saturate(float value) {
	return clamp(value, 0.0, 1.0);
}

void main() {
	vec4 scene = texture(uSceneColor, vUv);

	float depthCenter = texture(uDepthMap, vUv).r;
	float depthLeft = texture(uDepthMap, vUv - vec2(uTexelSize.x, 0.0)).r;
	float depthRight = texture(uDepthMap, vUv + vec2(uTexelSize.x, 0.0)).r;
	float depthUp = texture(uDepthMap, vUv - vec2(0.0, uTexelSize.y)).r;
	float depthDown = texture(uDepthMap, vUv + vec2(0.0, uTexelSize.y)).r;

	float depthDelta =
		abs(depthLeft - depthCenter) +
		abs(depthRight - depthCenter) +
		abs(depthUp - depthCenter) +
		abs(depthDown - depthCenter);
	float ao = 1.0 - saturate(depthDelta * 1.5);

	vec3 normal = normalize(texture(uNormalMap, vUv).xyz * 2.0 - 1.0);
	float nDotV = saturate(normal.z * 0.5 + 0.5);
	float fresnel = 1.0 - nDotV;
	fresnel = fresnel * fresnel * fresnel;
	float reflectionStability = 1.0 - saturate(depthDelta * 4.0);
	vec2 reflectionDir = normalize(normal.xy + vec2(1e-5, 0.0));
	vec2 reflectionStep = reflectionDir * uTexelSize * (4.0 + 8.0 * fresnel);
	vec2 reflectionUvA = clamp(vUv + reflectionStep, vec2(0.001), vec2(0.999));
	vec2 reflectionUvB = clamp(vUv + reflectionStep * 2.0, vec2(0.001), vec2(0.999));
	vec3 reflectionColor =
		(texture(uSceneColor, reflectionUvA).rgb +
		texture(uSceneColor, reflectionUvB).rgb) * 0.5;
	float reflectionWeight = fresnel * reflectionStability * 0.28;
	vec3 ssrColor = mix(scene.rgb, max(scene.rgb, reflectionColor), reflectionWeight);

	float nearDepthFog = depthCenter * 0.35;
	float farDepthFog = depthCenter * 0.02;
	float useNearDepthScale = step(depthCenter, 1.0);
	float fog = saturate(mix(farDepthFog, nearDepthFog, useNearDepthScale)) * 0.6;
	vec3 fogColor = mix(vec3(0.58, 0.64, 0.72), scene.rgb, 0.35);

	vec3 outColor = max(mix(ssrColor * ao, fogColor, fog), vec3(0.0));
	fragColor = vec4(outColor, scene.a);
}
