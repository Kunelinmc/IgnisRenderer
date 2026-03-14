#version 300 es
precision highp float;

in vec2 vUv;
uniform sampler2D uSourceMap;

out vec4 fragColor;

void main() {
	fragColor = texture(uSourceMap, vUv);
}