import type { VertexBufferLayout } from "../types";
import { PARTICLE_QUAD_VERTICES } from "../constants";

export const WEBGPU_PARTICLE_BINDING_TEXTURE = 0;
export const WEBGPU_PARTICLE_BINDING_SAMPLER = 1;
export const WEBGPU_PARTICLE_BINDING_UV_TRANSFORM = 2;

export const WEBGPU_PARTICLE_ATTR_QUAD_POSITION = 0;
export const WEBGPU_PARTICLE_ATTR_QUAD_UV = 1;
export const WEBGPU_PARTICLE_ATTR_INSTANCE_POSITION_SIZE = 2;
export const WEBGPU_PARTICLE_ATTR_INSTANCE_COLOR = 3;
export const WEBGPU_PARTICLE_ATTR_INSTANCE_UV_RECT = 4;
export const WEBGPU_PARTICLE_ATTR_INSTANCE_ROTATION = 5;
export const WEBGPU_PARTICLE_ATTR_INSTANCE_RECEIVE_SHADOW = 6;

export const WEBGPU_PARTICLE_QUAD_FLOATS_PER_VERTEX = 4;
export const WEBGPU_PARTICLE_QUAD_VERTEX_STRIDE =
	WEBGPU_PARTICLE_QUAD_FLOATS_PER_VERTEX * 4;

export const WEBGPU_PARTICLE_INSTANCE_FLOATS = 16;
export const WEBGPU_PARTICLE_INSTANCE_STRIDE =
	WEBGPU_PARTICLE_INSTANCE_FLOATS * 4;

export const WEBGPU_PARTICLE_UV_UNIFORM_FLOATS = 8;
export const WEBGPU_PARTICLE_UV_UNIFORM_SIZE =
	WEBGPU_PARTICLE_UV_UNIFORM_FLOATS * 4;

export const WEBGPU_PARTICLE_VERTEX_LAYOUTS: VertexBufferLayout[] = [
	{
		arrayStride: WEBGPU_PARTICLE_QUAD_VERTEX_STRIDE,
		stepMode: "vertex",
		attributes: [
			{
				shaderLocation: WEBGPU_PARTICLE_ATTR_QUAD_POSITION,
				offset: 0,
				format: "float32x2",
			},
			{
				shaderLocation: WEBGPU_PARTICLE_ATTR_QUAD_UV,
				offset: 8,
				format: "float32x2",
			},
		],
	},
	{
		arrayStride: WEBGPU_PARTICLE_INSTANCE_STRIDE,
		stepMode: "instance",
		attributes: [
			{
				shaderLocation: WEBGPU_PARTICLE_ATTR_INSTANCE_POSITION_SIZE,
				offset: 0,
				format: "float32x4",
			},
			{
				shaderLocation: WEBGPU_PARTICLE_ATTR_INSTANCE_COLOR,
				offset: 16,
				format: "float32x4",
			},
			{
				shaderLocation: WEBGPU_PARTICLE_ATTR_INSTANCE_UV_RECT,
				offset: 32,
				format: "float32x4",
			},
			{
				shaderLocation: WEBGPU_PARTICLE_ATTR_INSTANCE_ROTATION,
				offset: 48,
				format: "float32",
			},
			{
				shaderLocation: WEBGPU_PARTICLE_ATTR_INSTANCE_RECEIVE_SHADOW,
				offset: 52,
				format: "float32",
			},
		],
	},
];

export const WEBGPU_PARTICLE_QUAD_VERTICES = PARTICLE_QUAD_VERTICES;
