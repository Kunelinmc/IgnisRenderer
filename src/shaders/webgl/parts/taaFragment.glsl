#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uSceneColor;
uniform sampler2D uHistoryMap;
uniform sampler2D uMotionMap;
uniform sampler2D uMotionHistory;

uniform vec2 uTexelSize;
uniform float uHistoryWeight;
uniform float uDepthThreshold;
uniform float uMotionFactor;
uniform float uVarianceClampGamma;
uniform float uSharpen;
uniform float uHistoryValid;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragHistory;

const float PI = 3.14159265359;
const float EPSILON = 0.000001;

vec3 rgbToYCoCg(vec3 c) {
	float co = c.r - c.b;
	float t = c.b + co * 0.5;
	float cg = c.g - t;
	float y = t + cg * 0.5;
	return vec3(y, co, cg);
}

vec3 yCoCgToRgb(vec3 c) {
	float t = c.x - c.z * 0.5;
	float g = c.z + t;
	float b = t - c.y * 0.5;
	float r = b + c.y;
	return vec3(r, g, b);
}

float luma(vec3 c) {
	return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float minPositiveDepth(float a, float b) {
	if (a <= 0.0) return max(0.0, b);
	if (b <= 0.0) return max(0.0, a);
	return min(a, b);
}

float loadDepthMinCross(sampler2D tex, vec2 uv, vec2 texel) {
	float minDepth = texture(tex, uv).z;
	minDepth = minPositiveDepth(minDepth, texture(tex, uv + vec2(texel.x, 0.0)).z);
	minDepth = minPositiveDepth(minDepth, texture(tex, uv - vec2(texel.x, 0.0)).z);
	minDepth = minPositiveDepth(minDepth, texture(tex, uv + vec2(0.0, texel.y)).z);
	minDepth = minPositiveDepth(minDepth, texture(tex, uv - vec2(0.0, texel.y)).z);
	return minDepth;
}

void main() {
	vec4 curr = texture(uSceneColor, vUv);
	vec3 motionDepth = texture(uMotionMap, vUv).xyz;
	vec2 motion = motionDepth.xy;
	
	vec2 prevUv = vUv - vec2(motion.x * 0.5, -motion.y * 0.5);
	bool inside = prevUv.x >= 0.0 && prevUv.x <= 1.0 && prevUv.y >= 0.0 && prevUv.y <= 1.0;

	vec3 minYCoCg = vec3(1e9);
	vec3 maxYCoCg = vec3(-1e9);
	vec3 sumYCoCg = vec3(0.0);
	vec3 sumSqYCoCg = vec3(0.0);

	vec2 offsets[5];
	offsets[0] = vec2(0.0, 0.0);
	offsets[1] = vec2(-1.0, 0.0);
	offsets[2] = vec2(1.0, 0.0);
	offsets[3] = vec2(0.0, -1.0);
	offsets[4] = vec2(0.0, 1.0);

	for (int i = 0; i < 5; i++) {
		vec2 offset = offsets[i] * uTexelSize;
		vec3 ycocg = rgbToYCoCg(texture(uSceneColor, vUv + offset).rgb);
		minYCoCg = min(minYCoCg, ycocg);
		maxYCoCg = max(maxYCoCg, ycocg);
		sumYCoCg += ycocg;
		sumSqYCoCg += ycocg * ycocg;
	}

	vec3 meanYCoCg = sumYCoCg / 5.0;
	vec3 varianceYCoCg = max(sumSqYCoCg / 5.0 - meanYCoCg * meanYCoCg, vec3(0.0));
	vec3 sigmaYCoCg = sqrt(varianceYCoCg);
	float gamma = max(uVarianceClampGamma, 0.0);
	
	vec3 varianceMin = meanYCoCg - sigmaYCoCg * gamma;
	vec3 varianceMax = meanYCoCg + sigmaYCoCg * gamma;
	
	vec3 intersectionMin = max(minYCoCg, varianceMin);
	vec3 intersectionMax = min(maxYCoCg, varianceMax);
	vec3 clampMin = min(intersectionMin, intersectionMax);
	vec3 clampMax = max(intersectionMin, intersectionMax);

	vec4 hist = texture(uHistoryMap, prevUv);
	vec3 histYCoCg = clamp(rgbToYCoCg(hist.rgb), clampMin, clampMax);
	hist = vec4(max(yCoCgToRgb(histYCoCg), vec3(0.0)), hist.a);

	float currDepth = loadDepthMinCross(uMotionMap, vUv, uTexelSize);
	float prevDepth = loadDepthMinCross(uMotionHistory, prevUv, uTexelSize);
	
	bool hasDepth = currDepth > 0.0 && prevDepth > 0.0;
	float relDepthDiff = hasDepth ? abs(currDepth - prevDepth) / max(max(currDepth, prevDepth), 1e-4) : 1e6;
	float safeDepthThreshold = max(uDepthThreshold, 1e-4);
	float depthConfidence = hasDepth ? 1.0 - smoothstep(safeDepthThreshold * 0.5, safeDepthThreshold * 2.5, relDepthDiff) : 0.0;

	vec2 prevMotion = texture(uMotionHistory, prevUv).xy;
	vec2 forwardUv = prevUv + vec2(prevMotion.x * 0.5, -prevMotion.y * 0.5);
	vec2 reprojectionError = abs(forwardUv - vUv) / uTexelSize;
	float reprojectionErrorPx = max(reprojectionError.x, reprojectionError.y);
	float reprojectionConfidence = 1.0 - smoothstep(0.75, 3.0, reprojectionErrorPx);

	float currLuma = luma(curr.rgb);
	float histLuma = luma(hist.rgb);
	float lumaDiff = abs(currLuma - histLuma) / max(max(currLuma, histLuma), 1e-3);
	float colorConfidence = 1.0 - smoothstep(0.12, 0.7, lumaDiff);

	bool validBase = uHistoryValid > 0.5 && inside;
	float historyConfidence = validBase ? clamp(depthConfidence * reprojectionConfidence * colorConfidence, 0.0, 1.0) : 0.0;
	
	float motionMag = length(motion);
	float adaptive = clamp(uHistoryWeight * exp(-motionMag * uMotionFactor), 0.0, 0.96);
	float blend = clamp(adaptive * historyConfidence, 0.0, 0.96);
	
	vec4 temporalColor = mix(curr, hist, blend);

	vec4 left = texture(uSceneColor, vUv - vec2(uTexelSize.x, 0.0));
	vec4 right = texture(uSceneColor, vUv + vec2(uTexelSize.x, 0.0));
	vec4 up = texture(uSceneColor, vUv - vec2(0.0, uTexelSize.y));
	vec4 down = texture(uSceneColor, vUv + vec2(0.0, uTexelSize.y));
	vec4 blur = (left + right + up + down) * 0.25;
	
	float sharpenStrength = max(uSharpen, 0.0) * (1.0 - blend * 0.5);
	vec3 outRGB = max(temporalColor.rgb + (temporalColor.rgb - blur.rgb) * sharpenStrength, vec3(0.0));
	
	fragColor = vec4(outRGB, temporalColor.a);
	fragHistory = temporalColor;
}
