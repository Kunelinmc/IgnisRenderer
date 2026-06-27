#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform sampler2D uDepthMap;
uniform vec4 uInvSize; // fullInvW, fullInvH, aoInvW, aoInvH
uniform vec4 uBlurProj; // blurRadius, blurSharpness, tanHalfFov, aspect
uniform vec4 uPass; // blurDirX, blurDirY, isOrthographic, frameJitter

out vec4 fragColor;

void main() {
	ivec2 size = textureSize(uSourceMap, 0);
	ivec2 coord = ivec2(gl_FragCoord.xy);
	float centerDepth = texture(uDepthMap, vUv).z;
	int radius = clamp(int(uBlurProj.x + 0.5), 1, 4);
	float blurSharpness = max(uBlurProj.y, 1e-3);
	bool horizontal = abs(uPass.x) >= abs(uPass.y);
	ivec2 axis = horizontal ? ivec2(1, 0) : ivec2(0, 1);
	float sum = 0.0;
	float weightSum = 0.0;

	for (int tap = -4; tap <= 4; tap++) {
		if (abs(tap) > radius) {
			continue;
		}
		ivec2 sampleCoord = clamp(coord + axis * tap, ivec2(0), size - ivec2(1));
		vec2 sampleUv = (vec2(sampleCoord) + vec2(0.5)) * uInvSize.zw;
		float sampleDepth = texture(uDepthMap, sampleUv).z;
		float depthDelta = abs(sampleDepth - centerDepth);
		float bilateral = exp(-depthDelta * blurSharpness);
		float spatial = 1.0 - (abs(float(tap)) / float(radius + 1));
		float weight = bilateral * max(spatial, 0.0);
		float sampleAo = texelFetch(uSourceMap, sampleCoord, 0).x;
		sum += sampleAo * weight;
		weightSum += weight;
	}

	float ao = weightSum > 0.0 ?
		(sum / max(weightSum, 1e-4))
	:	texelFetch(uSourceMap, coord, 0).x;
	fragColor = vec4(vec3(ao), 1.0);
}
