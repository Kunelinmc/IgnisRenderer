#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform vec2 uTexelSize;
uniform vec4 uBloomParams; // x=threshold, y=softKnee, z=intensity, w=radius

out vec4 fragColor;

float luminance(vec3 color) {
	return dot(max(color, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
}

vec4 extractBloom(vec4 color) {
	float luma = luminance(color.rgb);
	float softKnee = max(uBloomParams.y, 1e-4);
	float soft = clamp(
		(luma - uBloomParams.x + softKnee) / (2.0 * softKnee),
		0.0,
		1.0
	);
	float contribution = max(luma - uBloomParams.x, 0.0) + soft * soft * softKnee;
	float scale = clamp(contribution / max(luma, 1e-4), 0.0, 1.0);
	return vec4(color.rgb * scale, clamp(color.a, 0.0, 1.0) * scale);
}

vec4 sampleBloom(vec2 offset) {
	vec2 radiusTexel = uTexelSize * max(uBloomParams.w, 0.5);
	return extractBloom(texture(uSourceMap, vUv + offset * radiusTexel));
}

void main() {
	vec4 source = texture(uSourceMap, vUv);
	vec4 bloom = sampleBloom(vec2(0.0, 0.0)) * 0.204164;
	bloom += sampleBloom(vec2(1.0, 0.0)) * 0.123841;
	bloom += sampleBloom(vec2(-1.0, 0.0)) * 0.123841;
	bloom += sampleBloom(vec2(0.0, 1.0)) * 0.123841;
	bloom += sampleBloom(vec2(0.0, -1.0)) * 0.123841;
	bloom += sampleBloom(vec2(1.0, 1.0)) * 0.07488;
	bloom += sampleBloom(vec2(-1.0, 1.0)) * 0.07488;
	bloom += sampleBloom(vec2(1.0, -1.0)) * 0.07488;
	bloom += sampleBloom(vec2(-1.0, -1.0)) * 0.07488;
	float intensity = max(uBloomParams.z, 0.0);
	vec3 outColor = max(source.rgb + bloom.rgb * intensity, vec3(0.0));
	float bloomCoverage = clamp(bloom.a * intensity, 0.0, 1.0);
	float outputAlpha = clamp(source.a, 0.0, 1.0) +
		bloomCoverage * (1.0 - clamp(source.a, 0.0, 1.0));
	fragColor = vec4(outColor, outputAlpha);
}
