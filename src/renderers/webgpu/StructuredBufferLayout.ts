import type { VertexBufferLayout, VertexFormat } from "../types";

export type BufferAddressSpace = "uniform" | "storage" | "vertex";
export type BufferScalarType = "f32" | "u32" | "i32";

export interface BufferScalarSchema {
	kind: "scalar";
	scalar: BufferScalarType;
}

export interface BufferVectorSchema {
	kind: "vector";
	scalar: BufferScalarType;
	size: 2 | 3 | 4;
}

export interface BufferMatrixSchema {
	kind: "matrix";
	scalar: "f32";
	columns: number;
	rows: number;
}

export interface BufferArraySchema {
	kind: "array";
	element: BufferTypeSchema;
	length: number;
}

export interface BufferFieldSchema {
	name: string;
	type: BufferTypeSchema;
}

export interface BufferStructSchema {
	kind: "struct";
	fields: BufferFieldSchema[];
}

export type BufferTypeSchema =
	| BufferScalarSchema
	| BufferVectorSchema
	| BufferMatrixSchema
	| BufferArraySchema
	| BufferStructSchema;

interface BufferScalarLayout {
	kind: "scalar";
	scalar: BufferScalarType;
	align: number;
	size: number;
}

interface BufferVectorLayout {
	kind: "vector";
	scalar: BufferScalarType;
	components: 2 | 3 | 4;
	align: number;
	size: number;
}

interface BufferMatrixLayout {
	kind: "matrix";
	scalar: "f32";
	columns: number;
	rows: number;
	align: number;
	size: number;
	stride: number;
}

interface BufferArrayLayout {
	kind: "array";
	element: BufferTypeLayout;
	length: number;
	align: number;
	size: number;
	stride: number;
}

interface BufferStructFieldLayout {
	name: string;
	type: BufferTypeLayout;
	offset: number;
	align: number;
}

interface BufferStructLayout {
	kind: "struct";
	fields: BufferStructFieldLayout[];
	fieldMap: Map<string, BufferStructFieldLayout>;
	align: number;
	size: number;
}

type BufferTypeLayout =
	| BufferScalarLayout
	| BufferVectorLayout
	| BufferMatrixLayout
	| BufferArrayLayout
	| BufferStructLayout;

export type BufferPath = string | number | readonly (string | number)[];

export type Matrix4Like =
	| number[][]
	| {
			elements: number[][];
	  };

export interface StructuredVertexAttribute {
	path: BufferPath;
	shaderLocation: number;
	format?: VertexFormat;
}

export interface StructuredVertexLayoutOptions {
	attributes: StructuredVertexAttribute[];
	arrayStride?: number;
	stepMode?: "vertex" | "instance";
}

export function scalar(scalarType: BufferScalarType): BufferScalarSchema {
	return {
		kind: "scalar",
		scalar: scalarType,
	};
}

export function vec(
	size: 2 | 3 | 4,
	scalarType: BufferScalarType = "f32"
): BufferVectorSchema {
	return {
		kind: "vector",
		scalar: scalarType,
		size,
	};
}

export function mat4x4f32(): BufferMatrixSchema {
	return {
		kind: "matrix",
		scalar: "f32",
		columns: 4,
		rows: 4,
	};
}

export function arrayOf(
	element: BufferTypeSchema,
	length: number
): BufferArraySchema {
	return {
		kind: "array",
		element,
		length,
	};
}

export function structOf(fields: BufferFieldSchema[]): BufferStructSchema {
	return {
		kind: "struct",
		fields,
	};
}

export class StructuredBufferLayout {
	private _addressSpace: BufferAddressSpace;
	private _schema: BufferTypeSchema;
	private _layout: BufferTypeLayout;

	constructor(
		schema: BufferTypeSchema,
		addressSpace: BufferAddressSpace = "uniform"
	) {
		this._schema = schema;
		this._addressSpace = addressSpace;
		this._layout = resolveTypeLayout(schema, addressSpace);
	}

	public get addressSpace(): BufferAddressSpace {
		return this._addressSpace;
	}

	public get byteSize(): number {
		return this._layout.size;
	}

	public get rootSchema(): BufferTypeSchema {
		return this._schema;
	}

	public get rootLayout(): BufferTypeLayout {
		return this._layout;
	}

	public byteOffsetOf(path: BufferPath): number {
		return resolvePathLayout(this._layout, path).byteOffset;
	}

	public byteSizeOf(path: BufferPath): number {
		return resolvePathLayout(this._layout, path).layout.size;
	}

	public createWriter(): StructuredBufferWriter {
		return new StructuredBufferWriter(this);
	}

	public createVertexBufferLayout(
		options: StructuredVertexLayoutOptions
	): VertexBufferLayout {
		if (this._addressSpace !== "vertex") {
			throw new Error(
				`StructuredBufferLayout.createVertexBufferLayout() requires addressSpace "vertex", received "${this._addressSpace}".`
			);
		}
		if (!options || !Array.isArray(options.attributes)) {
			throw new Error(
				"StructuredBufferLayout.createVertexBufferLayout() requires an attributes array."
			);
		}
		if (options.attributes.length <= 0) {
			throw new Error(
				"StructuredBufferLayout.createVertexBufferLayout() requires at least one attribute."
			);
		}

		const attributes: VertexBufferLayout["attributes"] = [];
		const seenLocations = new Set<number>();
		let requiredStride = 0;
		for (const attribute of options.attributes) {
			const shaderLocation = assertNonNegativeInteger(
				attribute.shaderLocation,
				"vertex.shaderLocation"
			);
			if (seenLocations.has(shaderLocation)) {
				throw new Error(
					`StructuredBufferLayout.createVertexBufferLayout() duplicate shaderLocation ${shaderLocation}.`
				);
			}
			seenLocations.add(shaderLocation);

			const resolved = resolvePathLayout(this._layout, attribute.path);
			const format =
				attribute.format ??
				inferVertexFormatFromLayout(resolved.layout, attribute.path);
			assertVertexFormatCompatible(
				format,
				resolved.layout,
				attribute.path
			);
			const byteSize = vertexFormatByteSize(format);
			requiredStride = Math.max(requiredStride, resolved.byteOffset + byteSize);
			attributes.push({
				shaderLocation,
				offset: resolved.byteOffset,
				format,
			});
		}

		const arrayStride =
			options.arrayStride !== undefined ? options.arrayStride : this.byteSize;
		assertNonNegativeInteger(arrayStride, "vertex.arrayStride");
		if (arrayStride <= 0) {
			throw new Error(
				`StructuredBufferLayout.createVertexBufferLayout() requires arrayStride > 0, received ${arrayStride}.`
			);
		}
		if (arrayStride < requiredStride) {
			throw new Error(
				`StructuredBufferLayout.createVertexBufferLayout() arrayStride ${arrayStride} is smaller than required ${requiredStride}.`
			);
		}

		return {
			arrayStride,
			stepMode: options.stepMode ?? "vertex",
			attributes,
		};
	}

	public assertByteSize(expectedBytes: number, label = "buffer"): void {
		if (!Number.isInteger(expectedBytes) || expectedBytes < 0) {
			throw new Error(
				`StructuredBufferLayout.assertByteSize(${label}) expected a non-negative integer, received ${expectedBytes}.`
			);
		}
		if (this.byteSize !== expectedBytes) {
			throw new Error(
				`${label} schema byte size mismatch: expected ${expectedBytes}, got ${this.byteSize}.`
			);
		}
	}
}

export class StructuredBufferWriter {
	private _layout: StructuredBufferLayout;
	private _buffer: ArrayBuffer;
	private _f32: Float32Array<ArrayBuffer>;
	private _u32: Uint32Array<ArrayBuffer>;
	private _i32: Int32Array<ArrayBuffer>;

	constructor(layout: StructuredBufferLayout) {
		this._layout = layout;
		this._buffer = new ArrayBuffer(layout.byteSize);
		this._f32 = new Float32Array(this._buffer);
		this._u32 = new Uint32Array(this._buffer);
		this._i32 = new Int32Array(this._buffer);
	}

	public get byteLength(): number {
		return this._buffer.byteLength;
	}

	public clear(): void {
		this._u32.fill(0);
	}

	public expectByteLength(expectedBytes: number, label = "buffer"): void {
		if (!Number.isInteger(expectedBytes) || expectedBytes < 0) {
			throw new Error(
				`StructuredBufferWriter.expectByteLength(${label}) expected a non-negative integer, received ${expectedBytes}.`
			);
		}
		if (this.byteLength !== expectedBytes) {
			throw new Error(
				`${label} writer byte size mismatch: expected ${expectedBytes}, got ${this.byteLength}.`
			);
		}
	}

	public writeF32(path: BufferPath, value: number): void {
		const resolved = this._resolvePath(path);
		if (resolved.layout.kind !== "scalar" || resolved.layout.scalar !== "f32") {
			throw new Error(
				`writeF32 expected scalar<f32> at ${formatPath(path)}, got ${describeLayout(resolved.layout)}.`
			);
		}
		this._writeScalarF32(resolved.byteOffset, value);
	}

	public writeU32(path: BufferPath, value: number): void {
		const resolved = this._resolvePath(path);
		if (resolved.layout.kind !== "scalar" || resolved.layout.scalar !== "u32") {
			throw new Error(
				`writeU32 expected scalar<u32> at ${formatPath(path)}, got ${describeLayout(resolved.layout)}.`
			);
		}
		this._writeScalarU32(resolved.byteOffset, value);
	}

	public writeI32(path: BufferPath, value: number): void {
		const resolved = this._resolvePath(path);
		if (resolved.layout.kind !== "scalar" || resolved.layout.scalar !== "i32") {
			throw new Error(
				`writeI32 expected scalar<i32> at ${formatPath(path)}, got ${describeLayout(resolved.layout)}.`
			);
		}
		this._writeScalarI32(resolved.byteOffset, value);
	}

	public writeVec(path: BufferPath, values: readonly number[]): void {
		const resolved = this._resolvePath(path);
		if (resolved.layout.kind !== "vector") {
			throw new Error(
				`writeVec expected vector at ${formatPath(path)}, got ${describeLayout(resolved.layout)}.`
			);
		}
		if (values.length !== resolved.layout.components) {
			throw new Error(
				`writeVec length mismatch at ${formatPath(path)}: expected ${resolved.layout.components}, got ${values.length}.`
			);
		}

		for (let i = 0; i < resolved.layout.components; i++) {
			const byteOffset = resolved.byteOffset + i * 4;
			switch (resolved.layout.scalar) {
				case "u32":
					this._writeScalarU32(byteOffset, values[i]);
					break;
				case "i32":
					this._writeScalarI32(byteOffset, values[i]);
					break;
				default:
					this._writeScalarF32(byteOffset, values[i]);
					break;
			}
		}
	}

	public writeMat4(path: BufferPath, matrix: Matrix4Like): void {
		const resolved = this._resolvePath(path);
		if (resolved.layout.kind !== "matrix") {
			throw new Error(
				`writeMat4 expected matrix at ${formatPath(path)}, got ${describeLayout(resolved.layout)}.`
			);
		}
		if (
			resolved.layout.scalar !== "f32" ||
			resolved.layout.columns !== 4 ||
			resolved.layout.rows !== 4
		) {
			throw new Error(
				`writeMat4 only supports mat4x4<f32> at ${formatPath(path)}, got ${describeLayout(resolved.layout)}.`
			);
		}

		const rows = resolveMatrixRows(matrix);
		for (let column = 0; column < resolved.layout.columns; column++) {
			for (let row = 0; row < resolved.layout.rows; row++) {
				const value = rows[row]?.[column];
				if (!Number.isFinite(value)) {
					throw new Error(
						`writeMat4 encountered invalid value at row ${row}, column ${column} for ${formatPath(path)}.`
					);
				}
				const byteOffset =
					resolved.byteOffset + column * resolved.layout.stride + row * 4;
				this._writeScalarF32(byteOffset, value);
			}
		}
	}

	public toArrayBuffer(): ArrayBuffer {
		return this._buffer;
	}

	public toFloat32Array(): Float32Array<ArrayBuffer> {
		return this._f32;
	}

	private _resolvePath(path: BufferPath): {
		layout: BufferTypeLayout;
		byteOffset: number;
	} {
		return resolvePathLayout(this._layout.rootLayout, path);
	}

	private _writeScalarF32(byteOffset: number, value: number): void {
		const validated = validateFiniteNumber(value, "f32");
		this._writeWithAlignment(byteOffset);
		this._f32[byteOffset >> 2] = Math.fround(validated);
	}

	private _writeScalarU32(byteOffset: number, value: number): void {
		const validated = validateUnsignedInteger(value);
		this._writeWithAlignment(byteOffset);
		this._u32[byteOffset >> 2] = validated >>> 0;
	}

	private _writeScalarI32(byteOffset: number, value: number): void {
		const validated = validateSignedInteger(value);
		this._writeWithAlignment(byteOffset);
		this._i32[byteOffset >> 2] = validated | 0;
	}

	private _writeWithAlignment(byteOffset: number): void {
		if (!Number.isInteger(byteOffset) || byteOffset < 0 || (byteOffset & 0x3) !== 0) {
			throw new Error(
				`StructuredBufferWriter attempted invalid scalar write at byte offset ${byteOffset}.`
			);
		}
		if (byteOffset + 4 > this._buffer.byteLength) {
			throw new Error(
				`StructuredBufferWriter out-of-bounds write at byte offset ${byteOffset} (byteLength ${this._buffer.byteLength}).`
			);
		}
	}
}

function resolveTypeLayout(
	schema: BufferTypeSchema,
	addressSpace: BufferAddressSpace
): BufferTypeLayout {
	switch (schema.kind) {
		case "scalar":
			return {
				kind: "scalar",
				scalar: schema.scalar,
				align: 4,
				size: 4,
			};
		case "vector": {
			const align =
				addressSpace === "vertex" ? 4
				: schema.size === 2 ? 8
				: schema.size === 3 ? 16
				: 16;
			return {
				kind: "vector",
				scalar: schema.scalar,
				components: schema.size,
				align,
				size: schema.size * 4,
			};
		}
		case "matrix": {
			const columnVectorLayout = resolveTypeLayout(
				{
					kind: "vector",
					scalar: schema.scalar,
					size: schema.rows as 2 | 3 | 4,
				},
				addressSpace
			);
			if (columnVectorLayout.kind !== "vector") {
				throw new Error("Internal error: matrix column layout is not a vector.");
			}
			const stride = alignTo(
				columnVectorLayout.size,
				columnVectorLayout.align
			);
			return {
				kind: "matrix",
				scalar: "f32",
				columns: schema.columns,
				rows: schema.rows,
				align: columnVectorLayout.align,
				size:
					schema.columns <= 0 ? 0
					:	stride * (schema.columns - 1) + columnVectorLayout.size,
				stride,
			};
		}
		case "array": {
			if (!Number.isInteger(schema.length) || schema.length < 0) {
				throw new Error(
					`Array schema length must be a non-negative integer, received ${schema.length}.`
				);
			}
			const elementLayout = resolveTypeLayout(schema.element, addressSpace);
			const elementAlign =
				addressSpace === "uniform" ? Math.max(16, elementLayout.align) : elementLayout.align;
			const stride = alignTo(elementLayout.size, elementAlign);
			const size =
				schema.length === 0 ? 0 : stride * (schema.length - 1) + elementLayout.size;
			return {
				kind: "array",
				element: elementLayout,
				length: schema.length,
				align: elementAlign,
				size,
				stride,
			};
		}
		case "struct": {
			const fields: BufferStructFieldLayout[] = [];
			const fieldMap = new Map<string, BufferStructFieldLayout>();
			let cursor = 0;
			let structAlign = 1;

			for (const field of schema.fields) {
				if (!field.name) {
					throw new Error("Struct schema field name must be non-empty.");
				}
				if (fieldMap.has(field.name)) {
					throw new Error(
						`Struct schema contains duplicate field name "${field.name}".`
					);
				}
				const fieldLayout = resolveTypeLayout(field.type, addressSpace);
				const fieldAlign = resolveMemberAlign(fieldLayout, addressSpace);
				const offset = alignTo(cursor, fieldAlign);
				const fieldLayoutEntry: BufferStructFieldLayout = {
					name: field.name,
					type: fieldLayout,
					offset,
					align: fieldAlign,
				};
				fields.push(fieldLayoutEntry);
				fieldMap.set(field.name, fieldLayoutEntry);
				cursor = offset + fieldLayout.size;
				structAlign = Math.max(structAlign, fieldAlign);
			}

			if (addressSpace === "uniform") {
				structAlign = Math.max(16, structAlign);
			}

			return {
				kind: "struct",
				fields,
				fieldMap,
				align: structAlign,
				size: alignTo(cursor, structAlign),
			};
		}
		default:
			throw new Error(`Unsupported schema kind: ${(schema as { kind: string }).kind}`);
	}
}

function resolveMemberAlign(
	layout: BufferTypeLayout,
	addressSpace: BufferAddressSpace
): number {
	if (addressSpace !== "uniform") {
		return layout.align;
	}
	if (layout.kind === "array" || layout.kind === "struct") {
		return Math.max(16, layout.align);
	}
	return layout.align;
}

function resolvePathLayout(
	rootLayout: BufferTypeLayout,
	path: BufferPath
): {
	layout: BufferTypeLayout;
	byteOffset: number;
} {
	const segments = normalizePath(path);
	let currentLayout = rootLayout;
	let byteOffset = 0;

	for (const segment of segments) {
		if (currentLayout.kind === "struct") {
			if (typeof segment !== "string") {
				throw new Error(
					`Path ${formatPath(path)} expected struct field segment but received ${String(segment)}.`
				);
			}
			const field = currentLayout.fieldMap.get(segment);
			if (!field) {
				throw new Error(
					`Path ${formatPath(path)} references unknown struct field "${segment}".`
				);
			}
			byteOffset += field.offset;
			currentLayout = field.type;
			continue;
		}

		if (currentLayout.kind === "array") {
			if (typeof segment !== "number" || !Number.isInteger(segment)) {
				throw new Error(
					`Path ${formatPath(path)} expected array index segment but received ${String(segment)}.`
				);
			}
			if (segment < 0 || segment >= currentLayout.length) {
				throw new Error(
					`Path ${formatPath(path)} array index ${segment} is out of bounds (length ${currentLayout.length}).`
				);
			}
			byteOffset += segment * currentLayout.stride;
			currentLayout = currentLayout.element;
			continue;
		}

		throw new Error(
			`Path ${formatPath(path)} attempts to descend into non-composite type ${describeLayout(currentLayout)}.`
		);
	}

	return {
		layout: currentLayout,
		byteOffset,
	};
}

function inferVertexFormatFromLayout(
	layout: BufferTypeLayout,
	path: BufferPath
): VertexFormat {
	if (layout.kind === "scalar") {
		switch (layout.scalar) {
			case "f32":
				return "float32";
			case "u32":
				return "uint32";
			default:
				throw new Error(
					`Unable to infer VertexFormat for ${formatPath(path)} from ${describeLayout(layout)}. Please provide an explicit format.`
				);
		}
	}
	if (layout.kind === "vector") {
		if (layout.scalar === "f32") {
			switch (layout.components) {
				case 2:
					return "float32x2";
				case 3:
					return "float32x3";
				default:
					return "float32x4";
			}
		}
		if (layout.scalar === "u32") {
			switch (layout.components) {
				case 2:
					return "uint32x2";
				case 3:
					return "uint32x3";
				default:
					return "uint32x4";
			}
		}
	}
	throw new Error(
		`Unable to infer VertexFormat for ${formatPath(path)} from ${describeLayout(layout)}.`
	);
}

function assertVertexFormatCompatible(
	format: VertexFormat,
	layout: BufferTypeLayout,
	path: BufferPath
): void {
	const info = describeVertexFormat(format);
	if (layout.kind === "scalar") {
		if (info.kind === "unorm8x4" && layout.size === info.byteSize) {
			return;
		}
		if (info.components !== 1) {
			throw new Error(
				`Vertex format ${format} is incompatible with ${describeLayout(layout)} at ${formatPath(path)}.`
			);
		}
		if (
			(info.kind === "f32" && layout.scalar === "f32") ||
			(info.kind === "u32" && layout.scalar === "u32")
		) {
			return;
		}
		throw new Error(
			`Vertex format ${format} is incompatible with ${describeLayout(layout)} at ${formatPath(path)}.`
		);
	}

	if (layout.kind === "vector") {
		if (info.kind === "unorm8x4") {
			throw new Error(
				`Vertex format ${format} is incompatible with ${describeLayout(layout)} at ${formatPath(path)}.`
			);
		}
		if (info.components !== layout.components) {
			throw new Error(
				`Vertex format ${format} component count mismatch for ${describeLayout(layout)} at ${formatPath(path)}.`
			);
		}
		if (
			(info.kind === "f32" && layout.scalar === "f32") ||
			(info.kind === "u32" && layout.scalar === "u32")
		) {
			return;
		}
		throw new Error(
			`Vertex format ${format} is incompatible with ${describeLayout(layout)} at ${formatPath(path)}.`
		);
	}

	throw new Error(
		`Vertex format ${format} requires scalar or vector layout at ${formatPath(path)}, got ${describeLayout(layout)}.`
	);
}

function vertexFormatByteSize(format: VertexFormat): number {
	return describeVertexFormat(format).byteSize;
}

function describeVertexFormat(format: VertexFormat): {
	kind: "f32" | "u32" | "unorm8x4";
	components: 1 | 2 | 3 | 4;
	byteSize: number;
} {
	switch (format) {
		case "float32":
			return { kind: "f32", components: 1, byteSize: 4 };
		case "float32x2":
			return { kind: "f32", components: 2, byteSize: 8 };
		case "float32x3":
			return { kind: "f32", components: 3, byteSize: 12 };
		case "float32x4":
			return { kind: "f32", components: 4, byteSize: 16 };
		case "uint32":
			return { kind: "u32", components: 1, byteSize: 4 };
		case "uint32x2":
			return { kind: "u32", components: 2, byteSize: 8 };
		case "uint32x3":
			return { kind: "u32", components: 3, byteSize: 12 };
		case "uint32x4":
			return { kind: "u32", components: 4, byteSize: 16 };
		case "unorm8x4":
			return { kind: "unorm8x4", components: 4, byteSize: 4 };
		default:
			throw new Error(`Unsupported VertexFormat "${format}".`);
	}
}

function normalizePath(path: BufferPath): Array<string | number> {
	if (typeof path === "string" || typeof path === "number") {
		return [path];
	}
	return Array.from(path);
}

function formatPath(path: BufferPath): string {
	const segments = normalizePath(path);
	if (segments.length === 0) {
		return "<root>";
	}
	let result = "";
	for (const segment of segments) {
		if (typeof segment === "number") {
			result += `[${segment}]`;
		} else if (result.length === 0) {
			result += segment;
		} else {
			result += `.${segment}`;
		}
	}
	return result;
}

function describeLayout(layout: BufferTypeLayout): string {
	switch (layout.kind) {
		case "scalar":
			return `scalar<${layout.scalar}>`;
		case "vector":
			return `vec${layout.components}<${layout.scalar}>`;
		case "matrix":
			return `mat${layout.columns}x${layout.rows}<${layout.scalar}>`;
		case "array":
			return `array<${describeLayout(layout.element)}, ${layout.length}>`;
		case "struct":
			return "struct";
		default:
			return "unknown";
	}
}

function resolveMatrixRows(matrix: Matrix4Like): number[][] {
	if (Array.isArray(matrix)) {
		return matrix;
	}
	const elements = matrix.elements;
	if (!Array.isArray(elements)) {
		throw new Error("Matrix object must provide a two-dimensional elements array.");
	}
	return elements;
}

function validateFiniteNumber(value: number, scalarType: string): number {
	if (!Number.isFinite(value)) {
		throw new Error(`Expected ${scalarType} finite number, received ${value}.`);
	}
	return value;
}

function validateUnsignedInteger(value: number): number {
	if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
		throw new Error(
			`Expected u32 integer in [0, 4294967295], received ${value}.`
		);
	}
	return value;
}

function validateSignedInteger(value: number): number {
	if (
		!Number.isInteger(value) ||
		value < -0x80000000 ||
		value > 0x7fffffff
	) {
		throw new Error(
			`Expected i32 integer in [-2147483648, 2147483647], received ${value}.`
		);
	}
	return value;
}

function assertNonNegativeInteger(value: number, label: string): number {
	if (!Number.isInteger(value) || value < 0) {
		throw new Error(
			`${label} must be a non-negative integer, received ${value}.`
		);
	}
	return value;
}

function alignTo(value: number, alignment: number): number {
	if (!Number.isFinite(value) || !Number.isFinite(alignment)) {
		throw new Error(
			`alignTo() requires finite values, received value=${value}, alignment=${alignment}.`
		);
	}
	if (!Number.isInteger(value) || !Number.isInteger(alignment)) {
		throw new Error(
			`alignTo() requires integer values, received value=${value}, alignment=${alignment}.`
		);
	}
	if (value < 0 || alignment <= 0) {
		throw new Error(
			`alignTo() requires value >= 0 and alignment > 0, received value=${value}, alignment=${alignment}.`
		);
	}
	const remainder = value % alignment;
	return remainder === 0 ? value : value + alignment - remainder;
}
