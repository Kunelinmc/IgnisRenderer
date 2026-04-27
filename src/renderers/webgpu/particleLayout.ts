import { PARTICLE_QUAD_VERTICES } from "../constants";
import {
	scalar,
	structOf,
	StructuredBufferLayout,
	vec,
} from "./StructuredBufferLayout";

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

const PARTICLE_QUAD_VERTEX_STRUCT_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "position", type: vec(2, "f32") },
		{ name: "uv", type: vec(2, "f32") },
	]),
	"vertex"
);
PARTICLE_QUAD_VERTEX_STRUCT_LAYOUT.assertByteSize(16, "particle quad vertex");

const PARTICLE_INSTANCE_STRUCT_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "positionSize", type: vec(4, "f32") },
		{ name: "color", type: vec(4, "f32") },
		{ name: "uvRect", type: vec(4, "f32") },
		{ name: "rotation", type: scalar("f32") },
		{ name: "receiveShadow", type: scalar("f32") },
		{ name: "padding0", type: scalar("f32") },
		{ name: "padding1", type: scalar("f32") },
	]),
	"vertex"
);
PARTICLE_INSTANCE_STRUCT_LAYOUT.assertByteSize(64, "particle instance vertex");

const PARTICLE_UV_UNIFORM_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "transformA", type: vec(4, "f32") },
		{ name: "transformB", type: vec(4, "f32") },
	]),
	"uniform"
);
PARTICLE_UV_UNIFORM_LAYOUT.assertByteSize(32, "particle uv transform");

export const WEBGPU_PARTICLE_QUAD_FLOATS_PER_VERTEX =
	PARTICLE_QUAD_VERTEX_STRUCT_LAYOUT.byteSize / 4;
export const WEBGPU_PARTICLE_QUAD_VERTEX_STRIDE =
	WEBGPU_PARTICLE_QUAD_FLOATS_PER_VERTEX * 4;

export const WEBGPU_PARTICLE_INSTANCE_FLOATS =
	PARTICLE_INSTANCE_STRUCT_LAYOUT.byteSize / 4;
export const WEBGPU_PARTICLE_INSTANCE_STRIDE =
	WEBGPU_PARTICLE_INSTANCE_FLOATS * 4;

export const WEBGPU_PARTICLE_UV_UNIFORM_FLOATS =
	PARTICLE_UV_UNIFORM_LAYOUT.byteSize / 4;
export const WEBGPU_PARTICLE_UV_UNIFORM_SIZE =
	WEBGPU_PARTICLE_UV_UNIFORM_FLOATS * 4;

export const WEBGPU_PARTICLE_VERTEX_LAYOUTS = [
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

export const WEBGPU_PARTICLE_QUAD_VERTICES = PARTICLE_QUAD_VERTICES;
