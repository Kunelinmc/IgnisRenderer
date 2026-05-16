import type { VertexBufferLayout } from "../types";

import {
	StructuredBufferLayout,
	structOf,
	vec,
} from "./StructuredBufferLayout";

export const WEBGPU_SCENE_VERTEX_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "position", type: vec(3, "f32") },
		{ name: "normal", type: vec(3, "f32") },
		{ name: "uv0", type: vec(2, "f32") },
		{ name: "tangent", type: vec(4, "f32") },
		{ name: "uv1", type: vec(2, "f32") },
		{ name: "joints0", type: vec(4, "f32") },
		{ name: "weights0", type: vec(4, "f32") },
		{ name: "joints1", type: vec(4, "f32") },
		{ name: "weights1", type: vec(4, "f32") },
		{ name: "uv2", type: vec(2, "f32") },
		{ name: "uv3", type: vec(2, "f32") },
	]),
	"vertex"
);

export const WEBGPU_SCENE_VERTEX_STRIDE = WEBGPU_SCENE_VERTEX_LAYOUT.byteSize;
export const WEBGPU_SCENE_VERTEX_FLOATS = WEBGPU_SCENE_VERTEX_STRIDE / 4;

export const WEBGPU_SCENE_VERTEX_FLOAT_OFFSET = {
	position: WEBGPU_SCENE_VERTEX_LAYOUT.byteOffsetOf("position") / 4,
	normal: WEBGPU_SCENE_VERTEX_LAYOUT.byteOffsetOf("normal") / 4,
	uv0: WEBGPU_SCENE_VERTEX_LAYOUT.byteOffsetOf("uv0") / 4,
	tangent: WEBGPU_SCENE_VERTEX_LAYOUT.byteOffsetOf("tangent") / 4,
	uv1: WEBGPU_SCENE_VERTEX_LAYOUT.byteOffsetOf("uv1") / 4,
	joints0: WEBGPU_SCENE_VERTEX_LAYOUT.byteOffsetOf("joints0") / 4,
	weights0: WEBGPU_SCENE_VERTEX_LAYOUT.byteOffsetOf("weights0") / 4,
	joints1: WEBGPU_SCENE_VERTEX_LAYOUT.byteOffsetOf("joints1") / 4,
	weights1: WEBGPU_SCENE_VERTEX_LAYOUT.byteOffsetOf("weights1") / 4,
	uv2: WEBGPU_SCENE_VERTEX_LAYOUT.byteOffsetOf("uv2") / 4,
	uv3: WEBGPU_SCENE_VERTEX_LAYOUT.byteOffsetOf("uv3") / 4,
} as const;

/**
 * Creates the shared scene vertex layout consumed by scene render pipelines.
 *
 * @returns A WebGPU vertex buffer layout derived from `WEBGPU_SCENE_VERTEX_LAYOUT`.
 */
export function createWebGPUSceneVertexBufferLayout(): VertexBufferLayout {
	return WEBGPU_SCENE_VERTEX_LAYOUT.createVertexBufferLayout({
		attributes: [
			{ path: "position", shaderLocation: 0 },
			{ path: "uv0", shaderLocation: 1 },
			{ path: "normal", shaderLocation: 2 },
			{ path: "tangent", shaderLocation: 3 },
			{ path: "uv1", shaderLocation: 4 },
			{ path: "joints0", shaderLocation: 5 },
			{ path: "weights0", shaderLocation: 6 },
			{ path: "joints1", shaderLocation: 7 },
			{ path: "weights1", shaderLocation: 8 },
			{ path: "uv2", shaderLocation: 9 },
			{ path: "uv3", shaderLocation: 10 },
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
			{ path: "position", shaderLocation: 0 },
			{ path: "joints0", shaderLocation: 5 },
			{ path: "weights0", shaderLocation: 6 },
			{ path: "joints1", shaderLocation: 7 },
			{ path: "weights1", shaderLocation: 8 },
		],
	});
}
