#version 300 es
precision highp float;
#import <ignis/postprocess/fog>

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
	fragColor = vec4(color, scene.a);
}
