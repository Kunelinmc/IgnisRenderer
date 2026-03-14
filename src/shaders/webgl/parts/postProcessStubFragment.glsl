#version 300 es
precision highp float;

in vec2 vUv;
uniform sampler2D uSceneColor;

out vec4 fragColor;

void main() {
	// TODO: Implement full effect logic
	fragColor = texture(uSceneColor, vUv);
}