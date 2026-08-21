#version 300 es
precision highp float;
#import <ignis/webgl/constants>
#import <ignis/color/srgb>

in vec2 vNdc;

uniform sampler2D uEnvironmentMap;
uniform vec4 uEnvironmentBasisRight;
uniform vec4 uEnvironmentBasisUp;
uniform vec3 uEnvironmentBasisBackward;
uniform float uEnvironmentIsOrthographic;
uniform int uEnvironmentMapIsLinear;
uniform vec3 uEnvironmentBackgroundTint;
uniform float uEnvironmentBackgroundExposure;
uniform float uEnvironmentBackgroundStrength;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragMotion;

void main() {
	vec3 right = uEnvironmentBasisRight.xyz;
	vec3 up = uEnvironmentBasisUp.xyz;
	vec3 backward = uEnvironmentBasisBackward.xyz;
	float tanHalfFov = uEnvironmentBasisRight.w;
	float aspect = uEnvironmentBasisUp.w;

	vec3 dir;
	if (uEnvironmentIsOrthographic > 0.5) {
		dir = normalize(-backward);
	} else {
		float cx = vNdc.x * aspect * tanHalfFov;
		float cy = vNdc.y * tanHalfFov;
		dir = normalize(right * cx + up * cy - backward);
	}

	float phi = atan(dir.x, dir.z);
	float theta = acos(clamp(dir.y, -1.0, 1.0));
	vec2 uv = vec2((phi + PI) / (2.0 * PI), theta / PI);
	vec4 sampled = texture(uEnvironmentMap, uv);
	vec3 sky = uEnvironmentMapIsLinear == 1 ? sampled.rgb : srgbToLinear(sampled.rgb);
	sky *= uEnvironmentBackgroundTint * uEnvironmentBackgroundExposure *
		uEnvironmentBackgroundStrength;
	fragColor = vec4(max(sky, vec3(0.0)), 1.0);
	fragMotion = vec4(0.0, 0.0, 0.0, 1.0);
}
