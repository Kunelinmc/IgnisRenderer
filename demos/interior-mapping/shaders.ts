export const INTERIOR_MAPPING_VERTEX_WGSL = `
struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
}

struct ModelUniforms {
	modelMatrix: mat4x4<f32>,
	prevModelMatrix: mat4x4<f32>,
	normalMatrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) uv0: vec2<f32>,
	@location(2) normal: vec3<f32>,
	@location(3) tangent: vec4<f32>,
}

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) worldPosition: vec3<f32>,
	@location(1) worldNormal: vec3<f32>,
	@location(2) uv0: vec2<f32>,
	@location(3) worldTangent: vec4<f32>,
}

@vertex
fn vsMain(input: VertexInput) -> VertexOutput {
	var output: VertexOutput;
	let worldPos = model.modelMatrix * vec4<f32>(input.position, 1.0);
	output.position = frame.viewProjection * worldPos;
	output.worldPosition = worldPos.xyz;
	output.worldNormal = normalize((model.normalMatrix * vec4<f32>(input.normal, 0.0)).xyz);

	let tangentWorld = (model.normalMatrix * vec4<f32>(input.tangent.xyz, 0.0)).xyz;
	output.worldTangent = vec4<f32>(
		normalize(tangentWorld),
		input.tangent.w
	);

	output.uv0 = input.uv0;
	return output;
}
`;

export const INTERIOR_MAPPING_FRAGMENT_WGSL = `
struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

#inject <ignis/material/uniform-block>

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) worldPosition: vec3<f32>,
	@location(1) worldNormal: vec3<f32>,
	@location(2) uv0: vec2<f32>,
	@location(3) worldTangent: vec4<f32>,
}

fn sampleRoomAtlas(panel: vec2<u32>, uv: vec2<f32>) -> vec4<f32> {
	let atlasGrid = vec2<f32>(3.0, 2.0);
	let dimensions = vec2<f32>(textureDimensions(ignisShaderTexture_roomAtlas));
	let texelMargin = vec2<f32>(0.5) / max(dimensions, vec2<f32>(1.0));
	let panelMin = vec2<f32>(f32(panel.x), f32(panel.y)) / atlasGrid;
	let panelMax = vec2<f32>(f32(panel.x + 1u), f32(panel.y + 1u)) / atlasGrid;
	let atlasUv = mix(
		panelMin + texelMargin,
		panelMax - texelMargin,
		clamp(vec2<f32>(uv.x, 1.0 - uv.y), vec2<f32>(0.0), vec2<f32>(1.0))
	);
	return ignisSampleTextureLevel_roomAtlas(
		atlasUv,
		atlasUv,
		atlasUv,
		atlasUv,
		0.0
	);
}

@fragment
fn fsMainSingle(input: VertexOutput) -> @location(0) vec4<f32> {
	let viewDirWorld = normalize(input.worldPosition - frame.cameraPosition.xyz);

	let N = normalize(input.worldNormal);
	let T = normalize(input.worldTangent.xyz);
	let B = normalize(cross(N, T) * input.worldTangent.w);

	let aspect = ignisShaderUniforms.roomAspect;
	let rdRaw = vec3<f32>(
		dot(viewDirWorld, T),
		dot(viewDirWorld, B),
		-dot(viewDirWorld, N)
	);
	let rdAspect = vec3<f32>(rdRaw.x / aspect, rdRaw.y, rdRaw.z);

	if (rdAspect.z <= 0.0) {
		return vec4<f32>(0.04, 0.04, 0.04, 1.0);
	}

	let tiling = ignisShaderUniforms.roomTiling;
	let uvsScaled = input.uv0 * tiling;
	let ro = vec3<f32>(fract(uvsScaled.x), fract(uvsScaled.y), 0.0);
	let depth = ignisShaderUniforms.roomDepth;

	let EPS = 1e-5;
	let rdx = select(
		rdAspect.x,
		sign(rdAspect.x + 0.5) * EPS,
		abs(rdAspect.x) < EPS
	);
	let rdy = select(
		rdAspect.y,
		sign(rdAspect.y + 0.5) * EPS,
		abs(rdAspect.y) < EPS
	);
	let rdz = rdAspect.z;

	let tx = (select(0.0, 1.0, rdx > 0.0) - ro.x) / rdx;
	let ty = (select(0.0, 1.0, rdy > 0.0) - ro.y) / rdy;
	let tz = (depth - ro.z) / rdz;

	let t = min(tx, min(ty, tz));
	if (t < 0.0) {
		return vec4<f32>(0.04, 0.04, 0.04, 1.0);
	}
	let hit = ro + t * rdAspect;

	let hitX = t < (tx + EPS) && tx <= (ty + EPS) && tx <= (tz + EPS);
	let hitY = !hitX && t < (ty + EPS) && ty <= (tz + EPS);
	var baseColor = vec3<f32>(0.0);
	var faceU = 0.0;
	var faceV = 0.0;
	var atlasPanel = vec2<u32>(2u, 1u);
	var atlasUv = vec2<f32>(0.5);

	if (hitX) {
		let wallDepthU = hit.z / depth;
		if (rdx > 0.0) {
			faceU = 1.0 - wallDepthU;
			atlasPanel = vec2<u32>(2u, 0u);
		} else {
			faceU = wallDepthU;
			atlasPanel = vec2<u32>(0u, 0u);
		}
		faceV = hit.y;
		atlasUv = vec2<f32>(faceU, 1.0 - faceV);
	} else if (hitY) {
		faceU = hit.x;
		faceV = hit.z / depth;
		if (rdy > 0.0) {
			atlasPanel = vec2<u32>(0u, 1u);
		} else {
			atlasPanel = vec2<u32>(1u, 1u);
		}
		atlasUv = vec2<f32>(faceU, faceV);
	} else {
		faceU = hit.x;
		faceV = hit.y;
		atlasPanel = vec2<u32>(1u, 0u);
		atlasUv = vec2<f32>(faceU, 1.0 - faceV);
	}

	baseColor = sampleRoomAtlas(atlasPanel, atlasUv).rgb;

	var color = baseColor;

	let foregroundUv = vec2<f32>(ro.x, 1.0 - ro.y);
	let foreground = sampleRoomAtlas(vec2<u32>(2u, 1u), foregroundUv);
	color = mix(color, foreground.rgb, clamp(foreground.a, 0.0, 1.0));

	return vec4<f32>(color, 1.0);
}
`;

export const INTERIOR_MAPPING_VERTEX_GLSL = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv0;
layout(location = 6) in vec4 aTangent;

uniform mat4 uModel;
uniform mat4 uViewProjection;
uniform mat3 uNormalMatrix;

out vec3 vWorldPosition;
out vec3 vWorldNormal;
out vec2 vUv0;
out vec4 vWorldTangent;

void main() {
	vec4 worldPos = uModel * vec4(aPosition, 1.0);
	gl_Position = uViewProjection * worldPos;
	vWorldPosition = worldPos.xyz;
	vWorldNormal = normalize(uNormalMatrix * aNormal);

	vec3 tangentWorld = uNormalMatrix * aTangent.xyz;
	vWorldTangent = vec4(normalize(tangentWorld), aTangent.w);
	vUv0 = aUv0;
}
`;

export const INTERIOR_MAPPING_FRAGMENT_GLSL = `#version 300 es
precision highp float;

in vec3 vWorldPosition;
in vec3 vWorldNormal;
in vec2 vUv0;
in vec4 vWorldTangent;

uniform vec3 uCameraPosition;

layout(location = 0) out vec4 fragColor;

vec4 ignisSampleTextureLevel_roomAtlas(
	vec2 uv0,
	vec2 uv1,
	vec2 uv2,
	vec2 uv3,
	float lod
);

vec4 sampleRoomAtlas(ivec2 panel, vec2 uv) {
	vec2 atlasGrid = vec2(3.0, 2.0);
	vec2 dimensions = vec2(textureSize(uShaderTex_roomAtlas, 0));
	vec2 texelMargin = vec2(0.5) / max(dimensions, vec2(1.0));
	vec2 panelMin = vec2(panel) / atlasGrid;
	vec2 panelMax = vec2(panel + ivec2(1)) / atlasGrid;
	vec2 atlasUv = mix(
		panelMin + texelMargin,
		panelMax - texelMargin,
		clamp(vec2(uv.x, 1.0 - uv.y), vec2(0.0), vec2(1.0))
	);
	return ignisSampleTextureLevel_roomAtlas(atlasUv, atlasUv, atlasUv, atlasUv, 0.0);
}

void main() {
	vec3 viewDirWorld = normalize(vWorldPosition - uCameraPosition);

	vec3 N = normalize(vWorldNormal);
	vec3 T = normalize(vWorldTangent.xyz);
	vec3 B = normalize(cross(N, T) * vWorldTangent.w);

	float aspect = uShaderUniform_roomAspect;
	vec3 rdRaw = vec3(
		dot(viewDirWorld, T),
		dot(viewDirWorld, B),
		-dot(viewDirWorld, N)
	);
	vec3 rdAspect = vec3(rdRaw.x / aspect, rdRaw.y, rdRaw.z);

	if (rdAspect.z <= 0.0) {
		fragColor = vec4(0.04, 0.04, 0.04, 1.0);
		return;
	}

	vec2 tiling = uShaderUniform_roomTiling;
	vec2 uvsScaled = vUv0 * tiling;
	vec3 ro = vec3(fract(uvsScaled.x), fract(uvsScaled.y), 0.0);
	float depth = uShaderUniform_roomDepth;

	float EPS = 1e-5;
	float rdx =
		abs(rdAspect.x) < EPS ? sign(rdAspect.x + 0.5) * EPS : rdAspect.x;
	float rdy =
		abs(rdAspect.y) < EPS ? sign(rdAspect.y + 0.5) * EPS : rdAspect.y;
	float rdz = rdAspect.z;

	float tx = ((rdx > 0.0 ? 1.0 : 0.0) - ro.x) / rdx;
	float ty = ((rdy > 0.0 ? 1.0 : 0.0) - ro.y) / rdy;
	float tz = (depth - ro.z) / rdz;

	float t = min(tx, min(ty, tz));
	if (t < 0.0) {
		fragColor = vec4(0.04, 0.04, 0.04, 1.0);
		return;
	}
	vec3 hit = ro + t * rdAspect;

	bool hitX = t < (tx + EPS) && tx <= (ty + EPS) && tx <= (tz + EPS);
	bool hitY = !hitX && t < (ty + EPS) && ty <= (tz + EPS);
	vec3 baseColor = vec3(0.0);
	float faceU = 0.0;
	float faceV = 0.0;
	ivec2 atlasPanel = ivec2(2, 1);
	vec2 atlasUv = vec2(0.5);

	if (hitX) {
		float wallDepthU = hit.z / depth;
		if (rdx > 0.0) {
			faceU = 1.0 - wallDepthU;
			atlasPanel = ivec2(2, 0);
		} else {
			faceU = wallDepthU;
			atlasPanel = ivec2(0, 0);
		}
		faceV = hit.y;
		atlasUv = vec2(faceU, 1.0 - faceV);
	} else if (hitY) {
		faceU = hit.x;
		faceV = hit.z / depth;
		if (rdy > 0.0) {
			atlasPanel = ivec2(0, 1);
		} else {
			atlasPanel = ivec2(1, 1);
		}
		atlasUv = vec2(faceU, faceV);
	} else {
		faceU = hit.x;
		faceV = hit.y;
		atlasPanel = ivec2(1, 0);
		atlasUv = vec2(faceU, 1.0 - faceV);
	}

	baseColor = sampleRoomAtlas(atlasPanel, atlasUv).rgb;

	vec3 color = baseColor;

	vec2 foregroundUv = vec2(ro.x, 1.0 - ro.y);
	vec4 foreground = sampleRoomAtlas(ivec2(2, 1), foregroundUv);
	color = mix(color, foreground.rgb, clamp(foreground.a, 0.0, 1.0));

	fragColor = vec4(color, 1.0);
}
`;
