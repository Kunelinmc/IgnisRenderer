vec4 fetchClusterHeader(int clusterIndex) {
	ivec2 texel = linearIndexToTexel(clusterIndex, uClusterHeaderTexSize);
	return texelFetch(uClusterHeaderTexture, texel, 0);
}

int fetchClusterListLightIndex(int listIndex) {
	int texelIndex = listIndex / 4;
	int component = listIndex - texelIndex * 4;
	ivec2 texel = linearIndexToTexel(texelIndex, uClusterIndexTexSize);
	vec4 packed = texelFetch(uClusterIndexTexture, texel, 0);
	if (component == 0) return int(floor(packed.x + 0.5));
	if (component == 1) return int(floor(packed.y + 0.5));
	if (component == 2) return int(floor(packed.z + 0.5));
	return int(floor(packed.w + 0.5));
}

vec4 fetchClusterLightRow(int lightIndex, int row) {
	int texelIndex = lightIndex * 4 + row;
	ivec2 texel = linearIndexToTexel(texelIndex, uClusterLightTexSize);
	return texelFetch(uClusterLightTexture, texel, 0);
}

bool resolveClusterSpan(out int offset, out int count, out int maxLightsPerCluster) {
	offset = 0;
	count = 0;
	maxLightsPerCluster = 0;
	if (uEnableClusteredLighting == 0) {
		return false;
	}
	int tilesX = max(int(floor(uClusterParams0.z + 0.5)), 1);
	int tilesY = max(int(floor(uClusterParams0.w + 0.5)), 1);
	int zSlices = max(int(floor(uClusterParams1.x + 0.5)), 1);
	maxLightsPerCluster = max(int(floor(uClusterParams1.y + 0.5)), 1);
	float logScale = uClusterParams1.z;
	float logBias = uClusterParams1.w;

	float width = max(uClusterParams0.x, 1.0);
	float height = max(uClusterParams0.y, 1.0);
	float xNorm = clamp(gl_FragCoord.x / width, 0.0, 0.999999);
	float yNorm = clamp(gl_FragCoord.y / height, 0.0, 0.999999);
	int tileX = clamp(int(floor(xNorm * float(tilesX))), 0, tilesX - 1);
	int tileY = clamp(int(floor(yNorm * float(tilesY))), 0, tilesY - 1);
	float viewDepth = max(vViewDepth, 1e-4);
	int slice = clamp(
		int(floor(log(viewDepth) * logScale + logBias)),
		0,
		zSlices - 1
	);
	int clusterIndex = tileX + tileY * tilesX + slice * tilesX * tilesY;
	vec4 header = fetchClusterHeader(clusterIndex);
	offset = max(0, int(floor(header.x + 0.5)));
	count = max(0, int(floor(header.y + 0.5)));
	return count > 0;
}
