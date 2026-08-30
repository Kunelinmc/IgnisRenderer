#version 300 es
precision highp float;
__IGNIS_WEBGL_DEPTH_DEFINES__

in vec2 vUv;
in vec2 vUv1;
in vec2 vUv2;
in vec2 vUv3;

layout(std140) uniform IgnisMaterialCommon {
	vec4 ignisBaseColor;
	vec4 ignisEmissive;
	vec4 ignisAlpha;
	vec4 ignisMaterialRenderParams;
#if WEBGL_DEPTH_BASE_MAP
	vec4 ignisBaseMapTransformA;
	vec4 ignisBaseMapTransformB;
#endif
};

#define uBaseColor ignisBaseColor
#if WEBGL_DEPTH_ALPHA_MASK
#define uAlpha ignisAlpha
#if WEBGL_DEPTH_BASE_MAP
uniform sampler2D uBaseMap;
#define uHasBaseMap 1
#define uBaseMapUV int(ignisBaseMapTransformB.z + 0.5)
#define uBaseMapTransformA ignisBaseMapTransformA
#define uBaseMapTransformB ignisBaseMapTransformB.xy
#endif
#endif

vec2 resolveUV(int uvSet) {
	if (uvSet == 1) return vUv1;
	if (uvSet == 2) return vUv2;
	if (uvSet >= 3) return vUv3;
	return vUv;
}

vec2 applyUVTransform(vec2 uv, vec4 transformA, vec2 transformB) {
	vec2 scaledUv = vec2(uv.x * transformA.x, uv.y * transformA.y);
	vec2 rotatedUv = vec2(
		scaledUv.x * transformB.x - scaledUv.y * transformB.y,
		scaledUv.x * transformB.y + scaledUv.y * transformB.x
	);
	return rotatedUv + transformA.zw;
}

void main() {
	float alpha = clamp(uBaseColor.a, 0.0, 1.0);
#if WEBGL_DEPTH_ALPHA_MASK && WEBGL_DEPTH_BASE_MAP
	if (uHasBaseMap == 1) {
		vec2 baseUv = applyUVTransform(
			resolveUV(uBaseMapUV),
			uBaseMapTransformA,
			uBaseMapTransformB
		);
		alpha *= texture(uBaseMap, baseUv).a;
	}
#endif

#if WEBGL_DEPTH_ALPHA_MASK
	if (uAlpha.y > 0.5 && alpha < uAlpha.x) {
		discard;
	}
#endif
}
