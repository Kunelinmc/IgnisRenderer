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
