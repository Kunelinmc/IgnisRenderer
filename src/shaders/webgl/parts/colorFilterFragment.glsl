#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform vec4 uFilterParams0;
uniform vec4 uFilterParams1;

out vec4 fragColor;

void main() {
	vec4 sampled = texture(uSourceMap, vUv);
	vec3 color = sampled.rgb;

	float brightness = uFilterParams0.x;
	float saturation = uFilterParams0.y;
	float contrast = uFilterParams0.z;
	float temperature = uFilterParams0.w;
	float tint = uFilterParams1.x;

	color += vec3(brightness);
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	color = mix(vec3(luma), color, saturation);
	color = (color - vec3(0.5)) * contrast + vec3(0.5);
	color += vec3(
		temperature * 0.1 + tint * 0.05,
		-tint * 0.1,
		-temperature * 0.1 + tint * 0.05
	);

	fragColor = vec4(clamp(color, vec3(0.0), vec3(1.0)), sampled.a);
}
