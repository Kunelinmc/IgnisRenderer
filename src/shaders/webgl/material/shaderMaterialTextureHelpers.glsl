vec2 ignisSelectShaderMaterialUv(
	vec2 uv0,
	vec2 uv1,
	vec2 uv2,
	vec2 uv3,
	int uvSet
) {
	if (uvSet == 1) return uv1;
	if (uvSet == 2) return uv2;
	if (uvSet >= 3) return uv3;
	return uv0;
}

vec4 ignisDecodeShaderMaterialSample(vec4 sampled, bool linear) {
	if (linear) return sampled;
	vec3 rgb = mix(
		pow((sampled.rgb + vec3(0.055)) / vec3(1.055), vec3(2.4)),
		sampled.rgb / vec3(12.92),
		lessThanEqual(sampled.rgb, vec3(0.04045))
	);
	return vec4(rgb, sampled.a);
}
