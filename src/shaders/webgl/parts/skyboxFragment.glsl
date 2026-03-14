#version 300 es
precision highp float;

const float PI = 3.14159265359;

in vec2 vNdc;

uniform sampler2D uSkyboxMap;
uniform vec4 uSkyboxBasisRight;
uniform vec4 uSkyboxBasisUp;
uniform vec3 uSkyboxBasisBackward;
uniform float uSkyboxIsOrthographic;
uniform int uSkyboxMapIsLinear;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragMotion;

vec3 srgbToLinear(vec3 c) {
	vec3 a = c / 12.92;
	vec3 b = pow((c + 0.055) / 1.055, vec3(2.4));
	return mix(b, a, lessThanEqual(c, vec3(0.04045)));
}

void main() {
	vec3 right = uSkyboxBasisRight.xyz;
	vec3 up = uSkyboxBasisUp.xyz;
	vec3 backward = uSkyboxBasisBackward.xyz;
	float tanHalfFov = uSkyboxBasisRight.w;
	float aspect = uSkyboxBasisUp.w;

	vec3 dir;
	if (uSkyboxIsOrthographic > 0.5) {
		dir = normalize(-backward);
	} else {
		float cx = vNdc.x * aspect * tanHalfFov;
		float cy = vNdc.y * tanHalfFov;
		dir = normalize(right * cx + up * cy - backward);
	}

	float phi = atan(dir.x, dir.z);
	float theta = acos(clamp(dir.y, -1.0, 1.0));
	vec2 uv = vec2((phi + PI) / (2.0 * PI), theta / PI);
	vec4 sampled = texture(uSkyboxMap, uv);
	vec3 sky = uSkyboxMapIsLinear == 1 ? sampled.rgb : srgbToLinear(sampled.rgb);
	fragColor = vec4(max(sky, vec3(0.0)), 1.0);
	fragMotion = vec4(0.0, 0.0, 0.0, 1.0);
}