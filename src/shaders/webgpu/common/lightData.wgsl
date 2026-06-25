struct DirectionalLightData {
	direction: vec4<f32>,
	color: vec4<f32>,
}

struct PointLightData {
	positionRange: vec4<f32>,
	color: vec4<f32>,
}

struct SpotLightData {
	positionRange: vec4<f32>,
	directionOuter: vec4<f32>,
	colorInner: vec4<f32>,
}

struct AreaLightData {
	positionRange: vec4<f32>,
	rightWidth: vec4<f32>,
	upHeight: vec4<f32>,
	normalAreaScale: vec4<f32>,
	color: vec4<f32>,
}

struct ShadowData {
	viewProjection: mat4x4<f32>,
	cascadeViewProjections: array<mat4x4<f32>, 4>,
	cascadeSplits: array<vec4<f32>, 4>,
	paramsA: vec4<f32>,
	paramsB: vec4<f32>,
	paramsC: vec4<f32>,
	paramsD: vec4<f32>,
	paramsE: vec4<f32>,
	paramsF: vec4<f32>,
}

struct ReflectionProbeData {
	worldToProbeRow0: vec4<f32>,
	worldToProbeRow1: vec4<f32>,
	worldToProbeRow2: vec4<f32>,
	probeToWorldRow0: vec4<f32>,
	probeToWorldRow1: vec4<f32>,
	probeToWorldRow2: vec4<f32>,
	dataA: vec4<f32>,
	dataB: vec4<f32>,
	dataC: vec4<f32>,
}
