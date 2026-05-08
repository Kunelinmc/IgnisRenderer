#version 300 es
precision highp float;

uniform vec3 uTransmittance;

out vec4 fragColor;

void main() {
	fragColor = vec4(clamp(uTransmittance, 0.0, 1.0), 1.0);
}
