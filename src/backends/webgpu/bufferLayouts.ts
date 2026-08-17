import type { VertexBufferLayout } from "../types";
import {
	MAX_AREA_LIGHTS,
	MAX_DIRECTIONAL_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
} from "../constants";

import {
	WEBGPU_CLUSTERED_PARAMS_FLOATS,
	WEBGPU_FRAME_CAMERA_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_LIGHT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_SHADOW_UNIFORM_BYTE_SIZE,
	WEBGPU_MODEL_UNIFORM_BYTE_SIZE,
	WEBGPU_PARTICLE_ATTR_INSTANCE_COLOR,
	WEBGPU_PARTICLE_ATTR_INSTANCE_POSITION_SIZE,
	WEBGPU_PARTICLE_ATTR_INSTANCE_RECEIVE_SHADOW,
	WEBGPU_PARTICLE_ATTR_INSTANCE_ROTATION,
	WEBGPU_PARTICLE_ATTR_INSTANCE_UV_RECT,
	WEBGPU_PARTICLE_ATTR_QUAD_POSITION,
	WEBGPU_PARTICLE_ATTR_QUAD_UV,
	WEBGPU_PARTICLE_INSTANCE_STRIDE,
	WEBGPU_PARTICLE_QUAD_VERTEX_STRIDE,
	WEBGPU_PARTICLE_UV_UNIFORM_SIZE,
	WEBGPU_SCENE_ATTR_JOINTS0,
	WEBGPU_SCENE_ATTR_JOINTS1,
	WEBGPU_SCENE_ATTR_NORMAL,
	WEBGPU_SCENE_ATTR_POSITION,
	WEBGPU_SCENE_ATTR_TANGENT,
	WEBGPU_SCENE_ATTR_UV0,
	WEBGPU_SCENE_ATTR_UV1,
	WEBGPU_SCENE_ATTR_UV2,
	WEBGPU_SCENE_ATTR_UV3,
	WEBGPU_SCENE_ATTR_WEIGHTS0,
	WEBGPU_SCENE_ATTR_WEIGHTS1,
	WEBGPU_SCENE_VERTEX_FLOAT_OFFSET,
	WEBGPU_SCENE_VERTEX_STRIDE,
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
	WEBGPU_VOLUMETRIC_LIGHT_STRIDE_FLOATS,
} from "./constants";
import {
	StructuredBufferLayout,
	arrayOf,
	mat4x4f32,
	scalar,
	structOf,
	vec,
	type BufferFieldSchema,
} from "./StructuredBufferLayout";

const F32 = scalar("f32");
const U32 = scalar("u32");
const VEC2_F32 = vec(2, "f32");
const VEC3_F32 = vec(3, "f32");
const VEC4_F32 = vec(4, "f32");
const VEC4_U32 = vec(4, "u32");
const MAT4X4_F32 = mat4x4f32();

const DIRECTIONAL_LIGHT_SCHEMA = structOf([
	{ name: "direction", type: VEC4_F32 },
	{ name: "color", type: VEC4_F32 },
]);

const POINT_LIGHT_SCHEMA = structOf([
	{ name: "positionRange", type: VEC4_F32 },
	{ name: "color", type: VEC4_F32 },
]);

const SPOT_LIGHT_SCHEMA = structOf([
	{ name: "positionRange", type: VEC4_F32 },
	{ name: "directionOuter", type: VEC4_F32 },
	{ name: "colorInner", type: VEC4_F32 },
]);

const AREA_LIGHT_SCHEMA = structOf([
	{ name: "positionRange", type: VEC4_F32 },
	{ name: "rightWidth", type: VEC4_F32 },
	{ name: "upHeight", type: VEC4_F32 },
	{ name: "normalAreaScale", type: VEC4_F32 },
	{ name: "color", type: VEC4_F32 },
]);

const SHADOW_DATA_SCHEMA = structOf([
	{ name: "viewProjection", type: MAT4X4_F32 },
	{ name: "cascadeViewProjections", type: arrayOf(MAT4X4_F32, 4) },
	{ name: "cascadeSplits", type: arrayOf(VEC4_F32, 4) },
	{ name: "paramsA", type: VEC4_F32 },
	{ name: "paramsB", type: VEC4_F32 },
	{ name: "paramsC", type: VEC4_F32 },
	{ name: "paramsD", type: VEC4_F32 },
	{ name: "paramsE", type: VEC4_F32 },
	{ name: "paramsF", type: VEC4_F32 },
]);

const REFLECTION_PROBE_SCHEMA = structOf([
	{ name: "worldToProbeRow0", type: VEC4_F32 },
	{ name: "worldToProbeRow1", type: VEC4_F32 },
	{ name: "worldToProbeRow2", type: VEC4_F32 },
	{ name: "probeToWorldRow0", type: VEC4_F32 },
	{ name: "probeToWorldRow1", type: VEC4_F32 },
	{ name: "probeToWorldRow2", type: VEC4_F32 },
	{ name: "dataA", type: VEC4_F32 },
	{ name: "dataB", type: VEC4_F32 },
	{ name: "dataC", type: VEC4_F32 },
]);

const VOLUMETRIC_LIGHT_RECORD_SCHEMA = structOf([
	{ name: "positionRange", type: VEC4_F32 },
	{ name: "directionOuter", type: VEC4_F32 },
	{ name: "colorInner", type: VEC4_F32 },
]);

const VOLUMETRIC_LIGHT_LAYOUT_CACHE = new Map<number, StructuredBufferLayout>();

export const WEBGPU_SCENE_VERTEX_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "position", type: vec(3, "f32") },
		{ name: "normal", type: vec(3, "f32") },
		{ name: "uv0", type: VEC2_F32 },
		{ name: "tangent", type: VEC4_F32 },
		{ name: "uv1", type: VEC2_F32 },
		{ name: "joints0", type: VEC4_F32 },
		{ name: "weights0", type: VEC4_F32 },
		{ name: "joints1", type: VEC4_F32 },
		{ name: "weights1", type: VEC4_F32 },
		{ name: "uv2", type: VEC2_F32 },
		{ name: "uv3", type: VEC2_F32 },
	]),
	"vertex"
);

WEBGPU_SCENE_VERTEX_LAYOUT.assertByteSize(
	WEBGPU_SCENE_VERTEX_STRIDE,
	"scene vertex"
);
assertFloatOffsets(
	WEBGPU_SCENE_VERTEX_LAYOUT,
	WEBGPU_SCENE_VERTEX_FLOAT_OFFSET,
	"scene vertex"
);

const PARTICLE_QUAD_VERTEX_STRUCT_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "position", type: VEC2_F32 },
		{ name: "uv", type: VEC2_F32 },
	]),
	"vertex"
);
PARTICLE_QUAD_VERTEX_STRUCT_LAYOUT.assertByteSize(
	WEBGPU_PARTICLE_QUAD_VERTEX_STRIDE,
	"particle quad vertex"
);

const PARTICLE_INSTANCE_STRUCT_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "positionSize", type: VEC4_F32 },
		{ name: "color", type: VEC4_F32 },
		{ name: "uvRect", type: VEC4_F32 },
		{ name: "rotation", type: F32 },
		{ name: "receiveShadow", type: F32 },
		{ name: "padding0", type: F32 },
		{ name: "padding1", type: F32 },
	]),
	"vertex"
);
PARTICLE_INSTANCE_STRUCT_LAYOUT.assertByteSize(
	WEBGPU_PARTICLE_INSTANCE_STRIDE,
	"particle instance vertex"
);

export const WEBGPU_PARTICLE_UV_UNIFORM_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "transformA", type: VEC4_F32 },
		{ name: "transformB", type: VEC4_F32 },
	]),
	"uniform"
);
WEBGPU_PARTICLE_UV_UNIFORM_LAYOUT.assertByteSize(
	WEBGPU_PARTICLE_UV_UNIFORM_SIZE,
	"particle uv transform"
);

export const WEBGPU_PARTICLE_VERTEX_LAYOUTS: VertexBufferLayout[] = [
	PARTICLE_QUAD_VERTEX_STRUCT_LAYOUT.createVertexBufferLayout({
		arrayStride: WEBGPU_PARTICLE_QUAD_VERTEX_STRIDE,
		stepMode: "vertex",
		attributes: [
			{
				path: "position",
				shaderLocation: WEBGPU_PARTICLE_ATTR_QUAD_POSITION,
			},
			{
				path: "uv",
				shaderLocation: WEBGPU_PARTICLE_ATTR_QUAD_UV,
			},
		],
	}),
	PARTICLE_INSTANCE_STRUCT_LAYOUT.createVertexBufferLayout({
		arrayStride: WEBGPU_PARTICLE_INSTANCE_STRIDE,
		stepMode: "instance",
		attributes: [
			{
				path: "positionSize",
				shaderLocation: WEBGPU_PARTICLE_ATTR_INSTANCE_POSITION_SIZE,
			},
			{
				path: "color",
				shaderLocation: WEBGPU_PARTICLE_ATTR_INSTANCE_COLOR,
			},
			{
				path: "uvRect",
				shaderLocation: WEBGPU_PARTICLE_ATTR_INSTANCE_UV_RECT,
			},
			{
				path: "rotation",
				shaderLocation: WEBGPU_PARTICLE_ATTR_INSTANCE_ROTATION,
			},
			{
				path: "receiveShadow",
				shaderLocation: WEBGPU_PARTICLE_ATTR_INSTANCE_RECEIVE_SHADOW,
			},
		],
	}),
];

export const WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "viewProjection", type: MAT4X4_F32 },
		{ name: "prevViewProjection", type: MAT4X4_F32 },
		{ name: "cameraPosition", type: VEC4_F32 },
		{ name: "environmentBasisRight", type: VEC4_F32 },
		{ name: "environmentBasisUp", type: VEC4_F32 },
		{ name: "environmentBasisBackward", type: VEC4_F32 },
		{ name: "ambientColor", type: VEC4_F32 },
		{ name: "lightCounts", type: VEC4_F32 },
		{ name: "options", type: VEC4_F32 },
		{ name: "environmentOptionsA", type: VEC4_F32 },
		{ name: "environmentOptionsB", type: VEC4_F32 },
		{ name: "taaJitterCurrentPrev", type: VEC4_F32 },
	]),
	"uniform"
);
WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT.assertByteSize(
	WEBGPU_FRAME_CAMERA_UNIFORM_BYTE_SIZE,
	"FrameCameraUniforms"
);

export const WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT = new StructuredBufferLayout(
	structOf([
		{
			name: "directionalLights",
			type: arrayOf(DIRECTIONAL_LIGHT_SCHEMA, MAX_DIRECTIONAL_LIGHTS),
		},
		{
			name: "pointLights",
			type: arrayOf(POINT_LIGHT_SCHEMA, MAX_POINT_LIGHTS),
		},
		{
			name: "spotLights",
			type: arrayOf(SPOT_LIGHT_SCHEMA, MAX_SPOT_LIGHTS),
		},
		{ name: "areaLightCounts", type: VEC4_F32 },
		{
			name: "areaLights",
			type: arrayOf(AREA_LIGHT_SCHEMA, MAX_AREA_LIGHTS),
		},
	]),
	"uniform"
);
WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT.assertByteSize(
	WEBGPU_FRAME_LIGHT_UNIFORM_BYTE_SIZE,
	"FrameLightUniforms"
);

export const WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT = new StructuredBufferLayout(
	structOf([
		{
			name: "directionalShadows",
			type: arrayOf(SHADOW_DATA_SCHEMA, MAX_DIRECTIONAL_LIGHTS),
		},
		{
			name: "spotShadows",
			type: arrayOf(SHADOW_DATA_SCHEMA, MAX_SPOT_LIGHTS),
		},
	]),
	"uniform"
);
WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT.assertByteSize(
	WEBGPU_FRAME_SHADOW_UNIFORM_BYTE_SIZE,
	"FrameShadowUniforms"
);

export const WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT = new StructuredBufferLayout(
	structOf([
		{
			name: "shAmbientCoeffs",
			type: arrayOf(VEC4_F32, WEBGPU_SH_COEFFICIENT_COUNT),
		},
		{
			name: "reflectionProbes",
			type: arrayOf(REFLECTION_PROBE_SCHEMA, MAX_REFLECTION_PROBES),
		},
		{ name: "localLightProbeCounts", type: VEC4_F32 },
		{
			name: "localLightProbeWorldToProbeRow0",
			type: arrayOf(VEC4_F32, MAX_LOCAL_LIGHT_PROBES),
		},
		{
			name: "localLightProbeWorldToProbeRow1",
			type: arrayOf(VEC4_F32, MAX_LOCAL_LIGHT_PROBES),
		},
		{
			name: "localLightProbeWorldToProbeRow2",
			type: arrayOf(VEC4_F32, MAX_LOCAL_LIGHT_PROBES),
		},
		{
			name: "localLightProbeDataA",
			type: arrayOf(VEC4_F32, MAX_LOCAL_LIGHT_PROBES),
		},
		{
			name: "localLightProbeDataB",
			type: arrayOf(VEC4_F32, MAX_LOCAL_LIGHT_PROBES),
		},
		{
			name: "localLightProbeSHAmbientCoeffs",
			type: arrayOf(
				VEC4_F32,
				MAX_LOCAL_LIGHT_PROBES * WEBGPU_SH_COEFFICIENT_COUNT
			),
		},
		{ name: "irradianceProbeGridWorldToGridRow0", type: VEC4_F32 },
		{ name: "irradianceProbeGridWorldToGridRow1", type: VEC4_F32 },
		{ name: "irradianceProbeGridWorldToGridRow2", type: VEC4_F32 },
		{ name: "irradianceProbeGridDataA", type: VEC4_F32 },
		{ name: "irradianceProbeGridDataB", type: VEC4_F32 },
		{ name: "irradianceProbeGridDataC", type: VEC4_F32 },
	]),
	"uniform"
);
WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT.assertByteSize(
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_BYTE_SIZE,
	"FrameEnvironmentUniforms"
);

export const WEBGPU_MODEL_UNIFORM_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "modelMatrix", type: MAT4X4_F32 },
		{ name: "prevModelMatrix", type: MAT4X4_F32 },
		{ name: "normalMatrix", type: MAT4X4_F32 },
		{ name: "baseColorFactor", type: VEC4_F32 },
		{ name: "emissiveFactor", type: VEC4_F32 },
		{ name: "surfaceParams0", type: VEC4_F32 },
		{ name: "surfaceParams1", type: VEC4_F32 },
		{ name: "surfaceParams2", type: VEC4_F32 },
		{ name: "surfaceParams3", type: VEC4_F32 },
		{ name: "specularColorFactor", type: VEC4_F32 },
		{ name: "phongAmbientShininess", type: VEC4_F32 },
		{ name: "phongSpecularShading", type: VEC4_F32 },
		{ name: "sheenColorClearcoatNormalScale", type: VEC4_F32 },
		{ name: "attenuationColor", type: VEC4_F32 },
		{ name: "anisotropyParams", type: VEC4_F32 },
		{ name: "anisotropyTextureTransformA", type: VEC4_F32 },
		{ name: "anisotropyTextureTransformB", type: VEC4_F32 },
		{ name: "materialFlags", type: VEC4_F32 },
		{ name: "pbrMasks", type: VEC4_U32 },
		{ name: "nodeRenderLayers", type: VEC4_F32 },
		{
			name: "textureTransformA",
			type: arrayOf(VEC4_F32, WEBGPU_TEXTURE_SLOT_COUNT),
		},
		{
			name: "textureTransformB",
			type: arrayOf(VEC4_F32, WEBGPU_TEXTURE_SLOT_COUNT),
		},
	]),
	"uniform"
);
WEBGPU_MODEL_UNIFORM_LAYOUT.assertByteSize(
	WEBGPU_MODEL_UNIFORM_BYTE_SIZE,
	"ModelUniforms"
);

export const WEBGPU_CLUSTER_GRID_PARAMS_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "screenWidth", type: U32 },
		{ name: "screenHeight", type: U32 },
		{ name: "tilesX", type: U32 },
		{ name: "tilesY", type: U32 },
		{ name: "zSlices", type: U32 },
		{ name: "clusterCount", type: U32 },
		{ name: "near", type: F32 },
		{ name: "far", type: F32 },
		{ name: "logScale", type: F32 },
		{ name: "logBias", type: F32 },
		{ name: "lightCount", type: U32 },
		{ name: "maxLightsPerCluster", type: U32 },
	]),
	"uniform"
);
WEBGPU_CLUSTER_GRID_PARAMS_LAYOUT.assertByteSize(
	WEBGPU_CLUSTERED_PARAMS_FLOATS * 4,
	"ClusterGridParams"
);

/**
 * Creates the shared scene vertex layout consumed by scene render pipelines.
 *
 * @returns A WebGPU vertex buffer layout derived from `WEBGPU_SCENE_VERTEX_LAYOUT`.
 */
export function createWebGPUSceneVertexBufferLayout(): VertexBufferLayout {
	return WEBGPU_SCENE_VERTEX_LAYOUT.createVertexBufferLayout({
		attributes: [
			{ path: "position", shaderLocation: WEBGPU_SCENE_ATTR_POSITION },
			{ path: "uv0", shaderLocation: WEBGPU_SCENE_ATTR_UV0 },
			{ path: "normal", shaderLocation: WEBGPU_SCENE_ATTR_NORMAL },
			{ path: "tangent", shaderLocation: WEBGPU_SCENE_ATTR_TANGENT },
			{ path: "uv1", shaderLocation: WEBGPU_SCENE_ATTR_UV1 },
			{ path: "joints0", shaderLocation: WEBGPU_SCENE_ATTR_JOINTS0 },
			{ path: "weights0", shaderLocation: WEBGPU_SCENE_ATTR_WEIGHTS0 },
			{ path: "joints1", shaderLocation: WEBGPU_SCENE_ATTR_JOINTS1 },
			{ path: "weights1", shaderLocation: WEBGPU_SCENE_ATTR_WEIGHTS1 },
			{ path: "uv2", shaderLocation: WEBGPU_SCENE_ATTR_UV2 },
			{ path: "uv3", shaderLocation: WEBGPU_SCENE_ATTR_UV3 },
		],
	});
}

/**
 * Creates the reduced scene vertex layout required by shadow-depth pipelines.
 *
 * @returns A WebGPU vertex buffer layout with position and animation attributes.
 */
export function createWebGPUShadowVertexBufferLayout(): VertexBufferLayout {
	return WEBGPU_SCENE_VERTEX_LAYOUT.createVertexBufferLayout({
		attributes: [
			{ path: "position", shaderLocation: WEBGPU_SCENE_ATTR_POSITION },
			{ path: "joints0", shaderLocation: WEBGPU_SCENE_ATTR_JOINTS0 },
			{ path: "weights0", shaderLocation: WEBGPU_SCENE_ATTR_WEIGHTS0 },
			{ path: "joints1", shaderLocation: WEBGPU_SCENE_ATTR_JOINTS1 },
			{ path: "weights1", shaderLocation: WEBGPU_SCENE_ATTR_WEIGHTS1 },
		],
	});
}

/**
 * Creates a runtime uniform buffer layout for `ShaderMaterial` bindings.
 *
 * @param fields - WGSL field names and structured buffer schemas in shader
 * declaration order.
 * @returns A uniform-address-space layout for the provided shader fields.
 * @throws If a field schema is invalid for `StructuredBufferLayout`.
 */
export function createWebGPUShaderMaterialUniformLayout(
	fields: readonly BufferFieldSchema[]
): StructuredBufferLayout {
	return new StructuredBufferLayout(
		structOf(fields.map((field) => ({ name: field.name, type: field.type }))),
		"uniform"
	);
}

/**
 * Resolves the storage buffer layout for volumetric light records.
 *
 * @param count - Number of light records to expose in the storage array.
 * @returns A cached storage-address-space layout sized for `count` records.
 * @throws If `count` produces a layout that does not match the expected stride.
 */
export function getWebGPUVolumetricLightLayout(
	count: number
): StructuredBufferLayout {
	const cached = VOLUMETRIC_LIGHT_LAYOUT_CACHE.get(count);
	if (cached) {
		return cached;
	}

	const layout = new StructuredBufferLayout(
		arrayOf(VOLUMETRIC_LIGHT_RECORD_SCHEMA, count),
		"storage"
	);
	layout.assertByteSize(
		count * WEBGPU_VOLUMETRIC_LIGHT_STRIDE_FLOATS * 4,
		"VolumetricLightBuffer"
	);
	VOLUMETRIC_LIGHT_LAYOUT_CACHE.set(count, layout);
	return layout;
}

function assertFloatOffsets(
	layout: StructuredBufferLayout,
	offsets: Record<string, number>,
	label: string
): void {
	for (const [field, expectedOffset] of Object.entries(offsets)) {
		const actualOffset = layout.byteOffsetOf(field) / 4;
		if (actualOffset !== expectedOffset) {
			throw new Error(
				`${label} field "${field}" float offset mismatch: ` +
					`expected ${expectedOffset}, got ${actualOffset}.`
			);
		}
	}
}
