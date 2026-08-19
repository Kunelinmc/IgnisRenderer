import type { IPrimitiveGeometry } from "../../core/types";
import { float32ToFloat16Bits, float16BitsToFloat32 } from "../../foundation/Float16";
import type { VertexAttribute, VertexBufferLayout, VertexFormat } from "../types";
import {
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
} from "./constants";

export const WEBGPU_GEOMETRY_POSITION_SLOT = 0;
export const WEBGPU_GEOMETRY_SURFACE_SLOT = 1;
export const WEBGPU_GEOMETRY_SKIN_SLOT = 2;
export const WEBGPU_GEOMETRY_DEFAULT_SLOT = 3;

export const WEBGPU_GEOMETRY_DEFAULT_BUFFER_SIZE = 84;
export const WEBGPU_UV_HALF_MAX_ABS_ERROR = 1 / 4096;

export type WebGPUSkinProfile = "static" | "skin4" | "skin8";

export interface WebGPUPackedVertexStream {
	readonly data: ArrayBufferView;
	readonly layout: VertexBufferLayout;
	readonly byteLength: number;
}

export interface WebGPUPackedVertexGeometry {
	readonly position: WebGPUPackedVertexStream;
	readonly surface: WebGPUPackedVertexStream | null;
	readonly skin: WebGPUPackedVertexStream | null;
	readonly defaultData: Uint8Array;
	readonly sceneLayouts: readonly VertexBufferLayout[];
	readonly shadowLayouts: readonly VertexBufferLayout[];
	readonly layoutKey: string;
	readonly shadowLayoutKey: string;
	readonly skinProfile: WebGPUSkinProfile;
	readonly vertexByteLength: number;
}

interface PackingAttribute {
	readonly name: string;
	readonly shaderLocation: number;
	readonly format: VertexFormat;
	readonly components: 2 | 3 | 4;
	readonly source: ArrayLike<number>;
	readonly sourceComponents: 2 | 3 | 4;
	offset: number;
}

const EMPTY_VERTEX_LAYOUT: VertexBufferLayout = {
	arrayStride: 0,
	attributes: [],
};

const DEFAULT_ATTRIBUTE = {
	normal: {
		shaderLocation: WEBGPU_SCENE_ATTR_NORMAL,
		format: "float32x3",
		offset: 0,
	},
	tangent: {
		shaderLocation: WEBGPU_SCENE_ATTR_TANGENT,
		format: "snorm16x4",
		offset: 12,
	},
	uv0: {
		shaderLocation: WEBGPU_SCENE_ATTR_UV0,
		format: "float16x2",
		offset: 20,
	},
	uv1: {
		shaderLocation: WEBGPU_SCENE_ATTR_UV1,
		format: "float16x2",
		offset: 24,
	},
	uv2: {
		shaderLocation: WEBGPU_SCENE_ATTR_UV2,
		format: "float16x2",
		offset: 28,
	},
	uv3: {
		shaderLocation: WEBGPU_SCENE_ATTR_UV3,
		format: "float16x2",
		offset: 32,
	},
	joints0: {
		shaderLocation: WEBGPU_SCENE_ATTR_JOINTS0,
		format: "float32x4",
		offset: 36,
	},
	weights0: {
		shaderLocation: WEBGPU_SCENE_ATTR_WEIGHTS0,
		format: "unorm16x4",
		offset: 52,
	},
	joints1: {
		shaderLocation: WEBGPU_SCENE_ATTR_JOINTS1,
		format: "float32x4",
		offset: 60,
	},
	weights1: {
		shaderLocation: WEBGPU_SCENE_ATTR_WEIGHTS1,
		format: "unorm16x4",
		offset: 76,
	},
} as const satisfies Record<string, VertexAttribute>;

/** @internal Packs backend-private WebGPU scene vertex streams. */
export function packWebGPUVertexGeometry(
	geometry: IPrimitiveGeometry,
	vertexCount: number
): WebGPUPackedVertexGeometry {
	const positionLayout: VertexBufferLayout = {
		arrayStride: 12,
		attributes: [{
			shaderLocation: WEBGPU_SCENE_ATTR_POSITION,
			format: "float32x3",
			offset: 0,
		}],
	};
	const position: WebGPUPackedVertexStream = {
		data: geometry.positions,
		layout: positionLayout,
		byteLength: geometry.positions.byteLength,
	};

	const surfaceAttributes = createSurfaceAttributes(geometry);
	const skinProfile = resolveSkinProfile(geometry);
	const skinAttributes = createSkinAttributes(geometry, skinProfile);
	const surface = packAttributes(surfaceAttributes, vertexCount);
	const skin = packAttributes(skinAttributes, vertexCount);
	const defaultAttributes = createDefaultAttributes(geometry, skinProfile);
	const defaultLayout: VertexBufferLayout = {
		arrayStride: 0,
		attributes: defaultAttributes,
	};
	const sceneLayouts = [
		positionLayout,
		surface?.layout ?? EMPTY_VERTEX_LAYOUT,
		skin?.layout ?? EMPTY_VERTEX_LAYOUT,
		defaultLayout,
	];
	const shadowDefaultLocations = new Set([
		WEBGPU_SCENE_ATTR_JOINTS0,
		WEBGPU_SCENE_ATTR_WEIGHTS0,
		WEBGPU_SCENE_ATTR_JOINTS1,
		WEBGPU_SCENE_ATTR_WEIGHTS1,
	]);
	const shadowLayouts = [
		positionLayout,
		EMPTY_VERTEX_LAYOUT,
		skin?.layout ?? EMPTY_VERTEX_LAYOUT,
		{
			arrayStride: 0,
			attributes: defaultAttributes.filter((attribute) =>
				shadowDefaultLocations.has(attribute.shaderLocation)
			),
		},
	];

	return {
		position,
		surface,
		skin,
		defaultData: new Uint8Array(WEBGPU_GEOMETRY_DEFAULT_BUFFER_SIZE),
		sceneLayouts,
		shadowLayouts,
		layoutKey: createVertexLayoutKey(sceneLayouts),
		shadowLayoutKey: createVertexLayoutKey(shadowLayouts),
		skinProfile,
		vertexByteLength:
			position.byteLength + (surface?.byteLength ?? 0) + (skin?.byteLength ?? 0),
	};
}

/** @internal Creates a stable cache key for a complete vertex-buffer layout. */
export function createVertexLayoutKey(
	layouts: readonly VertexBufferLayout[]
): string {
	return layouts.map((layout, slot) => {
		const attributes = layout.attributes
			.map((attribute) =>
				`${attribute.shaderLocation}:${attribute.format}:${attribute.offset}`
			)
			.join(",");
		return `${slot}@${layout.arrayStride}[${attributes}]`;
	}).join("|");
}

function createSurfaceAttributes(
	geometry: IPrimitiveGeometry
): PackingAttribute[] {
	const attributes: PackingAttribute[] = [];
	if (geometry.normals) {
		attributes.push(createAttribute(
			"normal",
			WEBGPU_SCENE_ATTR_NORMAL,
			"float32x3",
			3,
			geometry.normals
		));
	}
	if (geometry.tangents) {
		attributes.push(createAttribute(
			"tangent",
			WEBGPU_SCENE_ATTR_TANGENT,
			canPackSnorm16(geometry.tangents) ? "snorm16x4" : "float32x4",
			4,
			geometry.tangents
		));
	}
	addUvAttribute(attributes, "uv0", WEBGPU_SCENE_ATTR_UV0, geometry.uv0);
	addUvAttribute(attributes, "uv1", WEBGPU_SCENE_ATTR_UV1, geometry.uv1);
	addUvAttribute(attributes, "uv2", WEBGPU_SCENE_ATTR_UV2, geometry.uv2);
	addUvAttribute(attributes, "uv3", WEBGPU_SCENE_ATTR_UV3, geometry.uv3);
	return attributes;
}

function createSkinAttributes(
	geometry: IPrimitiveGeometry,
	profile: WebGPUSkinProfile
): PackingAttribute[] {
	if (profile === "static") return [];
	const empty = new Float32Array(0);
	const attributes = [
		createAttribute(
			"joints0",
			WEBGPU_SCENE_ATTR_JOINTS0,
			"float32x4",
			4,
			geometry.joints0 ?? empty
		),
		createAttribute(
			"weights0",
			WEBGPU_SCENE_ATTR_WEIGHTS0,
			geometry.weights0 && canPackUnorm16(geometry.weights0) ?
				"unorm16x4" : "float32x4",
			4,
			geometry.weights0 ?? empty
		),
	];
	if (profile === "skin8") {
		attributes.push(
			createAttribute(
				"joints1",
				WEBGPU_SCENE_ATTR_JOINTS1,
				"float32x4",
				4,
				geometry.joints1 ?? empty
			),
			createAttribute(
				"weights1",
				WEBGPU_SCENE_ATTR_WEIGHTS1,
				geometry.weights1 && canPackUnorm16(geometry.weights1) ?
					"unorm16x4" : "float32x4",
				4,
				geometry.weights1 ?? empty
			)
		);
	}
	return attributes;
}

function createDefaultAttributes(
	geometry: IPrimitiveGeometry,
	profile: WebGPUSkinProfile
): VertexAttribute[] {
	const attributes: VertexAttribute[] = [];
	if (!geometry.normals) attributes.push(DEFAULT_ATTRIBUTE.normal);
	if (!geometry.tangents) attributes.push(DEFAULT_ATTRIBUTE.tangent);
	if (!geometry.uv0) attributes.push(DEFAULT_ATTRIBUTE.uv0);
	if (!geometry.uv1) attributes.push(DEFAULT_ATTRIBUTE.uv1);
	if (!geometry.uv2) attributes.push(DEFAULT_ATTRIBUTE.uv2);
	if (!geometry.uv3) attributes.push(DEFAULT_ATTRIBUTE.uv3);
	if (profile === "static") {
		attributes.push(
			DEFAULT_ATTRIBUTE.joints0,
			DEFAULT_ATTRIBUTE.weights0,
			DEFAULT_ATTRIBUTE.joints1,
			DEFAULT_ATTRIBUTE.weights1
		);
	} else if (profile === "skin4") {
		attributes.push(DEFAULT_ATTRIBUTE.joints1, DEFAULT_ATTRIBUTE.weights1);
	}
	return attributes;
}

function resolveSkinProfile(geometry: IPrimitiveGeometry): WebGPUSkinProfile {
	if (geometry.joints1 || geometry.weights1) return "skin8";
	if (geometry.joints0 || geometry.weights0) return "skin4";
	return "static";
}

function addUvAttribute(
	attributes: PackingAttribute[],
	name: string,
	shaderLocation: number,
	source: Float32Array | null | undefined
): void {
	if (!source) return;
	attributes.push(createAttribute(
		name,
		shaderLocation,
		canPackFloat16Uv(source) ? "float16x2" : "float32x2",
		2,
		source
	));
}

function createAttribute(
	name: string,
	shaderLocation: number,
	format: VertexFormat,
	components: 2 | 3 | 4,
	source: ArrayLike<number>
): PackingAttribute {
	return {
		name,
		shaderLocation,
		format,
		components,
		source,
		sourceComponents: components,
		offset: 0,
	};
}

function packAttributes(
	attributes: PackingAttribute[],
	vertexCount: number
): WebGPUPackedVertexStream | null {
	if (attributes.length === 0) return null;
	let stride = 0;
	for (const attribute of attributes) {
		attribute.offset = stride;
		stride += vertexFormatByteSize(attribute.format);
	}
	const data = new Uint8Array(vertexCount * stride);
	const view = new DataView(data.buffer);
	for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
		const vertexOffset = vertexIndex * stride;
		for (const attribute of attributes) {
			writeAttribute(view, vertexOffset + attribute.offset, attribute, vertexIndex);
		}
	}
	const layout: VertexBufferLayout = {
		arrayStride: stride,
		attributes: attributes.map((attribute) => ({
			shaderLocation: attribute.shaderLocation,
			format: attribute.format,
			offset: attribute.offset,
		})),
	};
	return { data, layout, byteLength: data.byteLength };
}

function writeAttribute(
	view: DataView,
	byteOffset: number,
	attribute: PackingAttribute,
	vertexIndex: number
): void {
	const sourceOffset = vertexIndex * attribute.sourceComponents;
	for (let component = 0; component < attribute.components; component++) {
		const value = Number(attribute.source[sourceOffset + component] ?? 0);
		switch (attribute.format) {
			case "float16x2":
				view.setUint16(
					byteOffset + component * 2,
					float32ToFloat16Bits(value),
					true
				);
				break;
			case "snorm16x4":
				view.setInt16(
					byteOffset + component * 2,
					Math.round(Math.max(-1, Math.min(1, value)) * 32767),
					true
				);
				break;
			case "unorm16x4":
				view.setUint16(
					byteOffset + component * 2,
					Math.round(Math.max(0, Math.min(1, value)) * 65535),
					true
				);
				break;
			default:
				view.setFloat32(byteOffset + component * 4, value, true);
				break;
		}
	}
}

function canPackFloat16Uv(source: Float32Array): boolean {
	for (let i = 0; i < source.length; i++) {
		const value = source[i];
		if (!Number.isFinite(value)) return false;
		const roundTrip = float16BitsToFloat32(float32ToFloat16Bits(value));
		if (Math.abs(roundTrip - value) > WEBGPU_UV_HALF_MAX_ABS_ERROR) return false;
	}
	return true;
}

function canPackSnorm16(source: Float32Array): boolean {
	for (let i = 0; i < source.length; i++) {
		const value = source[i];
		if (!Number.isFinite(value) || value < -1 || value > 1) return false;
	}
	return true;
}

function canPackUnorm16(source: Float32Array): boolean {
	for (let i = 0; i < source.length; i++) {
		const value = source[i];
		if (!Number.isFinite(value) || value < 0 || value > 1) return false;
	}
	return true;
}

function vertexFormatByteSize(format: VertexFormat): number {
	switch (format) {
		case "float16x2":
			return 4;
		case "snorm16x4":
		case "unorm16x4":
		case "float32x2":
			return 8;
		case "float32x3":
			return 12;
		case "float32x4":
			return 16;
		default:
			throw new Error(`Unsupported WebGPU geometry vertex format "${format}".`);
	}
}
