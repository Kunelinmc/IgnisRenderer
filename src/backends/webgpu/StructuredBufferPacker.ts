import {
	StructuredBufferLayout,
	StructuredBufferWriter,
	type BufferPath,
	type Matrix4Like,
} from "./StructuredBufferLayout";

export type StructuredBufferPackerOutput = "arrayBuffer" | "float32Array";

type PathPrefix = readonly (string | number)[];

type StructuredBufferPackerResult<TOutput extends StructuredBufferPackerOutput> =
	TOutput extends "float32Array" ? Float32Array<ArrayBuffer> : ArrayBuffer;

type ValueResolver<TInput, TValue> = (
	input: TInput
) => TValue | null | undefined;

type ArrayValueResolver<TInput, TValue> = (
	input: TInput,
	index: number
) => TValue | null | undefined;

/**
 * Describes one structured buffer write rule for a packer input.
 *
 * Fields validate their target paths when the packer is created, then write
 * values into a `StructuredBufferWriter` during packing.
 */
export interface StructuredBufferPackerField<TInput> {
	readonly label: string;
	validate(layout: StructuredBufferLayout, prefix?: PathPrefix): void;
	write(
		writer: StructuredBufferWriter,
		input: TInput,
		prefix?: PathPrefix
	): void;
}

/**
 * Options used to create a structured buffer packer.
 */
export interface StructuredBufferPackerOptions<
	TInput,
	TOutput extends StructuredBufferPackerOutput = "arrayBuffer",
> {
	label: string;
	layout: StructuredBufferLayout;
	clearBeforePack?: boolean;
	output?: TOutput;
	fields: StructuredBufferPackerField<TInput>[];
}

/**
 * Packs typed input objects into a `StructuredBufferLayout` writer.
 *
 * The packer owns no GPU resources. It only coordinates field resolvers,
 * writer clearing, and output view selection.
 */
export class StructuredBufferPacker<
	TInput,
	TOutput extends StructuredBufferPackerOutput = "arrayBuffer",
> {
	private _label: string;
	private _layout: StructuredBufferLayout;
	private _clearBeforePack: boolean;
	private _output: StructuredBufferPackerOutput;
	private _fields: StructuredBufferPackerField<TInput>[];

	constructor(options: StructuredBufferPackerOptions<TInput, TOutput>) {
		this._label = options.label;
		this._layout = options.layout;
		this._clearBeforePack = options.clearBeforePack ?? true;
		this._output = options.output ?? "arrayBuffer";
		this._fields = options.fields.slice();

		for (const field of this._fields) {
			try {
				field.validate(this._layout);
			} catch (error) {
				const detail = error instanceof Error ? error.message : String(error);
				throw new Error(
					`${this._label} packer field "${field.label}" validation failed: ${detail}`
				);
			}
		}
	}

	/**
	 * Creates a reusable writer for this packer's layout.
	 *
	 * @returns A zero-initialized writer sized to the layout byte length.
	 */
	public createWriter(): StructuredBufferWriter {
		const writer = this._layout.createWriter();
		writer.expectByteLength(this._layout.byteSize, this._label);
		return writer;
	}

	/**
	 * Packs an input into a newly allocated writer.
	 *
	 * @param input - Source value consumed by the configured field resolvers.
	 * @returns The configured output view for the writer-owned buffer.
	 */
	public pack(input: TInput): StructuredBufferPackerResult<TOutput> {
		return this.packInto(this.createWriter(), input);
	}

	/**
	 * Packs an input into an existing writer.
	 *
	 * @param writer - Reusable writer created for the same layout.
	 * @param input - Source value consumed by the configured field resolvers.
	 * @returns The configured output view for the writer-owned buffer.
	 */
	public packInto(
		writer: StructuredBufferWriter,
		input: TInput
	): StructuredBufferPackerResult<TOutput> {
		writer.expectByteLength(this._layout.byteSize, this._label);
		if (this._clearBeforePack) {
			writer.clear();
		}
		for (const field of this._fields) {
			field.write(writer, input);
		}
		return this._toOutput(writer);
	}

	private _toOutput(
		writer: StructuredBufferWriter
	): StructuredBufferPackerResult<TOutput> {
		if (this._output === "float32Array") {
			return writer.toFloat32Array() as StructuredBufferPackerResult<TOutput>;
		}
		return writer.toArrayBuffer() as StructuredBufferPackerResult<TOutput>;
	}
}

/**
 * Creates a reusable structured buffer packer.
 *
 * @param options - Layout, output mode, and field rules for the packer.
 * @returns A packer that can allocate writers or reuse caller-owned writers.
 */
export function createStructuredBufferPacker<
	TInput,
	TOutput extends StructuredBufferPackerOutput = "arrayBuffer",
>(
	options: StructuredBufferPackerOptions<TInput, TOutput>
): StructuredBufferPacker<TInput, TOutput> {
	return new StructuredBufferPacker(options);
}

/**
 * Creates an `f32` scalar field rule.
 *
 * @param path - Target scalar path in the layout.
 * @param resolver - Returns the scalar value to write; `null`/`undefined` skips.
 */
export function f32<TInput>(
	path: BufferPath,
	resolver: ValueResolver<TInput, number>
): StructuredBufferPackerField<TInput> {
	return createScalarField(path, "f32", resolver);
}

/**
 * Creates a `u32` scalar field rule.
 *
 * @param path - Target scalar path in the layout.
 * @param resolver - Returns the scalar value to write; `null`/`undefined` skips.
 */
export function u32<TInput>(
	path: BufferPath,
	resolver: ValueResolver<TInput, number>
): StructuredBufferPackerField<TInput> {
	return createScalarField(path, "u32", resolver);
}

/**
 * Creates an `i32` scalar field rule.
 *
 * @param path - Target scalar path in the layout.
 * @param resolver - Returns the scalar value to write; `null`/`undefined` skips.
 */
export function i32<TInput>(
	path: BufferPath,
	resolver: ValueResolver<TInput, number>
): StructuredBufferPackerField<TInput> {
	return createScalarField(path, "i32", resolver);
}

/**
 * Creates an `f32` scalar field rule from a boolean value.
 *
 * @param path - Target scalar path in the layout.
 * @param resolver - Returns the boolean value to write; `null`/`undefined` skips.
 */
export function boolF32<TInput>(
	path: BufferPath,
	resolver: ValueResolver<TInput, boolean>
): StructuredBufferPackerField<TInput> {
	return createScalarField(path, "f32", (input): number | null | undefined => {
		const value = resolver(input);
		if (value == null) {
			return null;
		}
		return value ? 1 : 0;
	});
}

/**
 * Creates a `vec2` field rule.
 *
 * @param path - Target vector path in the layout.
 * @param resolver - Returns the vector value to write; `null`/`undefined` skips.
 */
export function vec2<TInput>(
	path: BufferPath,
	resolver: ValueResolver<TInput, readonly number[]>
): StructuredBufferPackerField<TInput> {
	return createVectorField(path, resolver);
}

/**
 * Creates a `vec3` field rule.
 *
 * @param path - Target vector path in the layout.
 * @param resolver - Returns the vector value to write; `null`/`undefined` skips.
 */
export function vec3<TInput>(
	path: BufferPath,
	resolver: ValueResolver<TInput, readonly number[]>
): StructuredBufferPackerField<TInput> {
	return createVectorField(path, resolver);
}

/**
 * Creates a `vec4` field rule.
 *
 * @param path - Target vector path in the layout.
 * @param resolver - Returns the vector value to write; `null`/`undefined` skips.
 */
export function vec4<TInput>(
	path: BufferPath,
	resolver: ValueResolver<TInput, readonly number[]>
): StructuredBufferPackerField<TInput> {
	return createVectorField(path, resolver);
}

/**
 * Creates a `mat4x4<f32>` field rule.
 *
 * @param path - Target matrix path in the layout.
 * @param resolver - Returns row-major matrix rows; `null`/`undefined` skips.
 */
export function mat4<TInput>(
	path: BufferPath,
	resolver: ValueResolver<TInput, Matrix4Like>
): StructuredBufferPackerField<TInput> {
	return {
		label: formatPath(path),
		validate: (layout, prefix = []) => {
			layout.byteSizeOf(prefixPath(prefix, path));
		},
		write: (writer, input, prefix = []) => {
			const value = resolver(input);
			if (value == null) {
				return;
			}
			writer.writeMat4(prefixPath(prefix, path), value);
		},
	};
}

/**
 * Creates a fixed-length array of `vec4` field rule.
 *
 * @param path - Target array path in the layout.
 * @param length - Number of elements to visit.
 * @param resolver - Returns each vector value; `null`/`undefined` skips.
 */
export function arrayVec4<TInput>(
	path: BufferPath,
	length: number,
	resolver: ArrayValueResolver<TInput, readonly number[]>
): StructuredBufferPackerField<TInput> {
	assertNonNegativeInteger(length, `arrayVec4(${formatPath(path)}).length`);
	return {
		label: formatPath(path),
		validate: (layout, prefix = []) => {
			validateArrayElementPath(layout, prefix, path, length);
		},
		write: (writer, input, prefix = []) => {
			const base = prefixPathSegments(prefix, path);
			for (let index = 0; index < length; index++) {
				const value = resolver(input, index);
				if (value == null) {
					continue;
				}
				writer.writeVec([...base, index], value);
			}
		},
	};
}

/**
 * Creates a fixed-length array of struct field rule.
 *
 * @param path - Target array path in the layout.
 * @param length - Number of elements to visit.
 * @param elementResolver - Returns each element input; `null`/`undefined` skips.
 * @param fields - Field rules relative to each array element.
 */
export function arrayStruct<TInput, TElement>(
	path: BufferPath,
	length: number,
	elementResolver: ArrayValueResolver<TInput, TElement>,
	fields: StructuredBufferPackerField<TElement>[]
): StructuredBufferPackerField<TInput> {
	assertNonNegativeInteger(length, `arrayStruct(${formatPath(path)}).length`);
	return {
		label: formatPath(path),
		validate: (layout, prefix = []) => {
			validateArrayElementPath(layout, prefix, path, length);
			if (length <= 0) {
				return;
			}
			const elementPrefix = [...prefixPathSegments(prefix, path), 0];
			for (const field of fields) {
				field.validate(layout, elementPrefix);
			}
		},
		write: (writer, input, prefix = []) => {
			const base = prefixPathSegments(prefix, path);
			for (let index = 0; index < length; index++) {
				const element = elementResolver(input, index);
				if (element == null) {
					continue;
				}
				const elementPrefix = [...base, index];
				for (const field of fields) {
					field.write(writer, element, elementPrefix);
				}
			}
		},
	};
}

/**
 * Creates a custom field rule for writes that need full writer access.
 *
 * @param label - Human-readable field label for diagnostics.
 * @param writerCallback - Callback invoked during packing.
 */
export function custom<TInput>(
	label: string,
	writerCallback: (
		writer: StructuredBufferWriter,
		input: TInput
	) => void
): StructuredBufferPackerField<TInput> {
	return {
		label,
		validate: () => {},
		write: (writer, input) => {
			writerCallback(writer, input);
		},
	};
}

function createScalarField<TInput>(
	path: BufferPath,
	scalarType: "f32" | "u32" | "i32",
	resolver: ValueResolver<TInput, number>
): StructuredBufferPackerField<TInput> {
	return {
		label: formatPath(path),
		validate: (layout, prefix = []) => {
			layout.byteSizeOf(prefixPath(prefix, path));
		},
		write: (writer, input, prefix = []) => {
			const value = resolver(input);
			if (value == null) {
				return;
			}
			const targetPath = prefixPath(prefix, path);
			if (scalarType === "u32") {
				writer.writeU32(targetPath, value);
				return;
			}
			if (scalarType === "i32") {
				writer.writeI32(targetPath, value);
				return;
			}
			writer.writeF32(targetPath, value);
		},
	};
}

function createVectorField<TInput>(
	path: BufferPath,
	resolver: ValueResolver<TInput, readonly number[]>
): StructuredBufferPackerField<TInput> {
	return {
		label: formatPath(path),
		validate: (layout, prefix = []) => {
			layout.byteSizeOf(prefixPath(prefix, path));
		},
		write: (writer, input, prefix = []) => {
			const value = resolver(input);
			if (value == null) {
				return;
			}
			writer.writeVec(prefixPath(prefix, path), value);
		},
	};
}

function validateArrayElementPath(
	layout: StructuredBufferLayout,
	prefix: PathPrefix,
	path: BufferPath,
	length: number
): void {
	const base = prefixPathSegments(prefix, path);
	if (length <= 0) {
		layout.byteSizeOf(base);
		return;
	}
	layout.byteSizeOf([...base, 0]);
}

function prefixPath(prefix: PathPrefix, path: BufferPath): BufferPath {
	if (prefix.length <= 0) {
		return path;
	}
	return [...prefix, ...normalizePath(path)];
}

function prefixPathSegments(
	prefix: PathPrefix,
	path: BufferPath
): Array<string | number> {
	return [...prefix, ...normalizePath(path)];
}

function normalizePath(path: BufferPath): Array<string | number> {
	if (typeof path === "string" || typeof path === "number") {
		return [path];
	}
	return Array.from(path);
}

function formatPath(path: BufferPath): string {
	const segments = normalizePath(path);
	if (segments.length <= 0) {
		return "<root>";
	}
	let result = "";
	for (const segment of segments) {
		if (typeof segment === "number") {
			result += `[${segment}]`;
		} else if (result.length <= 0) {
			result += segment;
		} else {
			result += `.${segment}`;
		}
	}
	return result;
}

function assertNonNegativeInteger(value: number, label: string): void {
	if (!Number.isInteger(value) || value < 0) {
		throw new Error(`${label} must be a non-negative integer, received ${value}.`);
	}
}
