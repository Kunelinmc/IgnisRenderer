import { Texture } from "../core/Texture";
import type { Environment } from "../core/Environment";
import { Logger } from "../foundation/Logger";
import { clamp } from "../maths/Common";
import { Loader } from "./Loader";

export interface EXRParseOptions {
	/**
	 * Alpha value used when the EXR has no alpha channel.
	 */
	defaultAlpha?: number;
}

export interface EXREnvironmentApplyOptions {
	/**
	 * When true, assigns the texture to `environment.backgroundTexture`.
	 */
	background?: boolean;
	/**
	 * When true, assigns the texture to `environment.iblTexture`.
	 */
	ibl?: boolean;
}

export interface EXRLoadEnvironmentOptions
	extends EXRParseOptions,
		EXREnvironmentApplyOptions {}

export type EXREnvironmentTarget =
	| Environment
	| {
			environment: Environment;
	  };

interface EXRBox2i {
	xMin: number;
	yMin: number;
	xMax: number;
	yMax: number;
}

interface EXRChannel {
	name: string;
	pixelType: EXRPixelType;
	xSampling: number;
	ySampling: number;
}

interface EXRHeader {
	channels: EXRChannel[];
	compression: EXRCompression;
	dataWindow: EXRBox2i;
	displayWindow: EXRBox2i | null;
	lineOrder: EXRLineOrder;
	type: string | null;
	chunkCount: number | null;
}

interface EXRComponentMap {
	r: number;
	g: number;
	b: number;
	a: number;
}

const EXR_MAGIC = 20000630;
const VERSION_TILED_FLAG = 1 << 9;
const VERSION_DEEP_FLAG = 1 << 11;
const VERSION_MULTIPART_FLAG = 1 << 12;
const DEFAULT_ALPHA = 1;

enum EXRPixelType {
	UINT = 0,
	HALF = 1,
	FLOAT = 2,
}

enum EXRCompression {
	NO_COMPRESSION = 0,
	RLE_COMPRESSION = 1,
	ZIPS_COMPRESSION = 2,
	ZIP_COMPRESSION = 3,
	PIZ_COMPRESSION = 4,
	PXR24_COMPRESSION = 5,
	B44_COMPRESSION = 6,
	B44A_COMPRESSION = 7,
	DWAA_COMPRESSION = 8,
	DWAB_COMPRESSION = 9,
	HTJ2K256_COMPRESSION = 10,
	HTJ2K32_COMPRESSION = 11,
}

enum EXRLineOrder {
	INCREASING_Y = 0,
	DECREASING_Y = 1,
	RANDOM_Y = 2,
}

/**
 * Loads scanline OpenEXR environment maps into HDR `Texture` instances.
 */
export class EXRLoader extends Loader {
	constructor() {
		super();
	}

	/**
	 * Loads an OpenEXR texture from `url`.
	 *
	 * @param url URL, request path, or data URL passed to `fetch`.
	 * @param options Parse options. `defaultAlpha` is used when no alpha channel
	 * is present.
	 * @returns A linear HDR `Texture` backed by `Float32Array` RGBA pixels.
	 * @remarks The loader supports single-part scanline EXR files with
	 * uncompressed, RLE, ZIPS, or ZIP pixel data. ZIP and ZIPS require
	 * `DecompressionStream` at runtime. On load failure a 1x1 black HDR fallback
	 * texture is returned and marked as a load-error fallback. The method emits
	 * normal `Loader` lifecycle events and uses the shared loader cache.
	 */
	public async load(
		url: string,
		options: EXRParseOptions = {}
	): Promise<Texture> {
		try {
			const texture = await this._loadCached(
				createParseCacheKey(url, options),
				async () => {
					const buffer = await this._fetchWithProgress(url);
					return this.parseAsync(buffer, options);
				}
			);
			this.emit("load", texture);
			return texture;
		} catch (error) {
			this.emit("error", error);
			Logger.error([`EXRLoader: Failed to load ${url}`, error], {
				scope: "EXRLoader",
			});
			return createLoadErrorFallbackTexture();
		}
	}

	/**
	 * Loads an OpenEXR texture and assigns it to an environment target.
	 *
	 * @param url URL, request path, or data URL passed to `fetch`.
	 * @param target Either a `Scene`-like object with `.environment` or the
	 * `Environment` instance itself.
	 * @param options Parse and assignment options. By default the loaded texture
	 * is assigned to both `backgroundTexture` and `iblTexture`.
	 * @returns The loaded `Texture`, including the fallback texture if loading
	 * failed.
	 * @remarks Assignment triggers the environment change notifications normally
	 * emitted by `Environment` setters. Passing `{ background: false }` or
	 * `{ ibl: false }` suppresses the corresponding assignment.
	 */
	public async loadEnvironment(
		url: string,
		target: EXREnvironmentTarget,
		options: EXRLoadEnvironmentOptions = {}
	): Promise<Texture> {
		const texture = await this.load(url, options);
		this.applyToEnvironment(target, texture, options);
		return texture;
	}

	/**
	 * Assigns an existing EXR texture to an environment target.
	 *
	 * @param target Either a `Scene`-like object with `.environment` or the
	 * `Environment` instance itself.
	 * @param texture Texture to assign. It should be a linear HDR texture created
	 * by `EXRLoader.parse`, `EXRLoader.parseAsync`, or `EXRLoader.load`.
	 * @param options Assignment options. By default both background and IBL slots
	 * are updated.
	 * @returns The same `texture` instance for call chaining.
	 * @remarks This method mutates `environment.backgroundTexture` and/or
	 * `environment.iblTexture`, causing the owning scene to become dirty through
	 * the existing `Environment` change event path.
	 */
	public applyToEnvironment(
		target: EXREnvironmentTarget,
		texture: Texture,
		options: EXREnvironmentApplyOptions = {}
	): Texture {
		const environment = resolveEnvironmentTarget(target);
		if (options.background !== false) {
			environment.backgroundTexture = texture;
		}
		if (options.ibl !== false) {
			environment.iblTexture = texture;
		}
		return texture;
	}

	/**
	 * Parses uncompressed or RLE-compressed OpenEXR bytes synchronously.
	 *
	 * @param buffer OpenEXR file bytes.
	 * @param options Parse options. `defaultAlpha` is used when no alpha channel
	 * is present.
	 * @returns A linear HDR `Texture` backed by `Float32Array` RGBA pixels.
	 * @remarks This synchronous parser cannot inflate ZIP/ZIPS chunks. Use
	 * `parseAsync` or `load` for ZIP/ZIPS files. The parser supports single-part
	 * scanline images with `R`, `G`, `B`, and optional `A` channels using `HALF`,
	 * `FLOAT`, or `UINT` sample types and unit sampling.
	 */
	public parse(
		buffer: ArrayBuffer,
		options: EXRParseOptions = {}
	): Texture {
		const parser = new EXRScanlineParser(buffer, options);
		return parser.parseSync();
	}

	/**
	 * Parses OpenEXR bytes asynchronously.
	 *
	 * @param buffer OpenEXR file bytes.
	 * @param options Parse options. `defaultAlpha` is used when no alpha channel
	 * is present.
	 * @returns A linear HDR `Texture` backed by `Float32Array` RGBA pixels.
	 * @remarks Supports single-part scanline EXR files with uncompressed, RLE,
	 * ZIPS, or ZIP pixel data. ZIP and ZIPS require `DecompressionStream`; if the
	 * runtime does not provide it, parsing rejects with an actionable error.
	 */
	public async parseAsync(
		buffer: ArrayBuffer,
		options: EXRParseOptions = {}
	): Promise<Texture> {
		const parser = new EXRScanlineParser(buffer, options);
		return parser.parseAsync();
	}
}

class EXRScanlineParser {
	private _reader: BinaryReader;
	private _defaultAlpha: number;
	private _header: EXRHeader | null = null;
	private _width = 0;
	private _height = 0;
	private _bytesPerLine = 0;
	private _output: Float32Array = new Float32Array(0);
	private _componentMap: EXRComponentMap = {
		r: -1,
		g: -1,
		b: -1,
		a: -1,
	};

	constructor(buffer: ArrayBuffer, options: EXRParseOptions) {
		this._reader = new BinaryReader(buffer);
		this._defaultAlpha = sanitizeDefaultAlpha(options.defaultAlpha);
	}

	public parseSync(): Texture {
		this._parsePreamble();
		this._decodeChunksSync();
		return this._createTexture();
	}

	public async parseAsync(): Promise<Texture> {
		this._parsePreamble();
		await this._decodeChunksAsync();
		return this._createTexture();
	}

	private _parsePreamble(): void {
		const magic = this._reader.readUint32();
		if (magic !== EXR_MAGIC) {
			throw new Error("EXRLoader: Invalid OpenEXR magic number.");
		}

		const versionField = this._reader.readUint32();
		const version = versionField & 0xff;
		if (version !== 2) {
			throw new Error(
				`EXRLoader: Unsupported OpenEXR version ${version}; expected version 2.`
			);
		}
		if ((versionField & VERSION_MULTIPART_FLAG) !== 0) {
			throw new Error("EXRLoader: Multi-part EXR files are not supported.");
		}
		if ((versionField & VERSION_DEEP_FLAG) !== 0) {
			throw new Error("EXRLoader: Deep EXR files are not supported.");
		}
		if ((versionField & VERSION_TILED_FLAG) !== 0) {
			throw new Error("EXRLoader: Tiled EXR files are not supported.");
		}

		this._header = this._readHeader();
		this._validateHeader(this._header);
		this._initializeOutput(this._header);
		this._skipOffsetTable(this._header);
	}

	private _readHeader(): EXRHeader {
		const header: EXRHeader = {
			channels: [],
			compression: EXRCompression.NO_COMPRESSION,
			dataWindow: { xMin: 0, yMin: 0, xMax: -1, yMax: -1 },
			displayWindow: null,
			lineOrder: EXRLineOrder.INCREASING_Y,
			type: null,
			chunkCount: null,
		};

		while (!this._reader.isEOF()) {
			const name = this._reader.readNullTerminatedString();
			if (name.length === 0) {
				break;
			}
			const type = this._reader.readNullTerminatedString();
			const size = this._reader.readInt32();
			const valueOffset = this._reader.offset;

			switch (name) {
				case "channels":
					if (type === "chlist") {
						header.channels = this._readChannels(valueOffset, size);
					}
					break;
				case "compression":
					if (type === "compression" && size >= 1) {
						header.compression = this._reader.view.getUint8(valueOffset);
					}
					break;
				case "dataWindow":
					if (type === "box2i" && size === 16) {
						header.dataWindow = this._readBox2i(valueOffset);
					}
					break;
				case "displayWindow":
					if (type === "box2i" && size === 16) {
						header.displayWindow = this._readBox2i(valueOffset);
					}
					break;
				case "lineOrder":
					if (type === "lineOrder" && size >= 1) {
						header.lineOrder = this._reader.view.getUint8(valueOffset);
					}
					break;
				case "type":
					if (type === "string") {
						header.type = this._readSizedString(valueOffset, size);
					}
					break;
				case "chunkCount":
					if (type === "int" && size === 4) {
						header.chunkCount = this._reader.view.getInt32(valueOffset, true);
					}
					break;
				default:
					break;
			}

			this._reader.seek(valueOffset + size);
		}

		return header;
	}

	private _readChannels(offset: number, size: number): EXRChannel[] {
		const reader = this._reader.sliceReader(offset, size);
		const channels: EXRChannel[] = [];
		while (!reader.isEOF()) {
			const name = reader.readNullTerminatedString();
			if (name.length === 0) {
				break;
			}
			const pixelType = reader.readInt32() as EXRPixelType;
			reader.skip(1);
			reader.skip(3);
			const xSampling = reader.readInt32();
			const ySampling = reader.readInt32();
			channels.push({
				name,
				pixelType,
				xSampling,
				ySampling,
			});
		}
		return channels;
	}

	private _readBox2i(offset: number): EXRBox2i {
		const view = this._reader.view;
		return {
			xMin: view.getInt32(offset, true),
			yMin: view.getInt32(offset + 4, true),
			xMax: view.getInt32(offset + 8, true),
			yMax: view.getInt32(offset + 12, true),
		};
	}

	private _readSizedString(offset: number, size: number): string {
		const bytes = new Uint8Array(this._reader.buffer, offset, size);
		let result = "";
		for (let i = 0; i < bytes.length; i++) {
			if (bytes[i] === 0) break;
			result += String.fromCharCode(bytes[i]);
		}
		return result;
	}

	private _validateHeader(header: EXRHeader): void {
		if (header.type && header.type !== "scanlineimage") {
			throw new Error(
				`EXRLoader: Unsupported EXR image type "${header.type}".`
			);
		}
		if (
			header.lineOrder !== EXRLineOrder.INCREASING_Y &&
			header.lineOrder !== EXRLineOrder.DECREASING_Y
		) {
			throw new Error("EXRLoader: RANDOM_Y scanline order is not supported.");
		}
		if (!isSupportedCompression(header.compression)) {
			throw new Error(
				`EXRLoader: Unsupported EXR compression method ${header.compression}.`
			);
		}
		if (header.channels.length === 0) {
			throw new Error("EXRLoader: EXR file has no channels.");
		}

		const dataWindow = header.dataWindow;
		const width = dataWindow.xMax - dataWindow.xMin + 1;
		const height = dataWindow.yMax - dataWindow.yMin + 1;
		if (width <= 0 || height <= 0) {
			throw new Error(
				`EXRLoader: Invalid dataWindow ${dataWindow.xMin},${dataWindow.yMin} to ${dataWindow.xMax},${dataWindow.yMax}.`
			);
		}

		for (const channel of header.channels) {
			if (!isSupportedPixelType(channel.pixelType)) {
				throw new Error(
					`EXRLoader: Unsupported pixel type ${channel.pixelType} for channel "${channel.name}".`
				);
			}
			if (channel.xSampling !== 1 || channel.ySampling !== 1) {
				throw new Error(
					`EXRLoader: Channel "${channel.name}" uses sampling ${channel.xSampling}x${channel.ySampling}; only 1x1 sampling is supported.`
				);
			}
		}

		const componentMap = resolveComponentMap(header.channels);
		if (componentMap.r < 0 || componentMap.g < 0 || componentMap.b < 0) {
			throw new Error(
				"EXRLoader: EXR image must provide R, G, and B channels."
			);
		}
	}

	private _initializeOutput(header: EXRHeader): void {
		const dataWindow = header.dataWindow;
		this._width = dataWindow.xMax - dataWindow.xMin + 1;
		this._height = dataWindow.yMax - dataWindow.yMin + 1;
		this._bytesPerLine = calculateBytesPerLine(header.channels, this._width);
		this._componentMap = resolveComponentMap(header.channels);
		this._output = new Float32Array(this._width * this._height * 4);
		for (let index = 3; index < this._output.length; index += 4) {
			this._output[index] = this._defaultAlpha;
		}
	}

	private _skipOffsetTable(header: EXRHeader): void {
		const chunkCount = this._resolveChunkCount(header);
		this._reader.skip(chunkCount * 8);
	}

	private _decodeChunksSync(): void {
		const header = this._requireHeader();
		const chunkCount = this._resolveChunkCount(header);
		for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
			const chunk = this._readChunkHeader(header);
			const decoded = this._decodeChunkDataSync(
				header,
				chunk.payload,
				chunk.expectedByteLength
			);
			this._copyChunkPixels(header, decoded, chunk.y, chunk.rowCount);
		}
	}

	private async _decodeChunksAsync(): Promise<void> {
		const header = this._requireHeader();
		const chunkCount = this._resolveChunkCount(header);
		for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
			const chunk = this._readChunkHeader(header);
			const decoded = await this._decodeChunkDataAsync(
				header,
				chunk.payload,
				chunk.expectedByteLength
			);
			this._copyChunkPixels(header, decoded, chunk.y, chunk.rowCount);
		}
	}

	private _readChunkHeader(header: EXRHeader): {
		y: number;
		rowCount: number;
		expectedByteLength: number;
		payload: Uint8Array;
	} {
		const y = this._reader.readInt32();
		const pixelDataSize = this._reader.readUint32();
		const payload = this._reader.readBytes(pixelDataSize);
		const rowIndex = y - header.dataWindow.yMin;
		if (rowIndex < 0 || rowIndex >= this._height) {
			throw new Error(`EXRLoader: Scanline chunk y=${y} is outside dataWindow.`);
		}
		const rowCount = Math.min(
			getScanlinesPerChunk(header.compression),
			this._height - rowIndex
		);
		return {
			y,
			rowCount,
			expectedByteLength: rowCount * this._bytesPerLine,
			payload,
		};
	}

	private _decodeChunkDataSync(
		header: EXRHeader,
		payload: Uint8Array,
		expectedByteLength: number
	): Uint8Array {
		if (payload.byteLength === expectedByteLength) {
			return payload;
		}

		switch (header.compression) {
			case EXRCompression.NO_COMPRESSION:
				throw new Error(
					`EXRLoader: Uncompressed chunk has ${payload.byteLength} bytes; expected ${expectedByteLength}.`
				);
			case EXRCompression.RLE_COMPRESSION:
				return decodePredictor(deinterleave(decodeRLE(payload, expectedByteLength)));
			case EXRCompression.ZIPS_COMPRESSION:
			case EXRCompression.ZIP_COMPRESSION:
				throw new Error(
					"EXRLoader: ZIP/ZIPS compressed EXR data requires parseAsync() or load()."
				);
			default:
				throw new Error(
					`EXRLoader: Unsupported EXR compression method ${header.compression}.`
				);
		}
	}

	private async _decodeChunkDataAsync(
		header: EXRHeader,
		payload: Uint8Array,
		expectedByteLength: number
	): Promise<Uint8Array> {
		if (payload.byteLength === expectedByteLength) {
			return payload;
		}

		switch (header.compression) {
			case EXRCompression.NO_COMPRESSION:
				throw new Error(
					`EXRLoader: Uncompressed chunk has ${payload.byteLength} bytes; expected ${expectedByteLength}.`
				);
			case EXRCompression.RLE_COMPRESSION:
				return decodePredictor(deinterleave(decodeRLE(payload, expectedByteLength)));
			case EXRCompression.ZIPS_COMPRESSION:
			case EXRCompression.ZIP_COMPRESSION: {
				const inflated = await inflateZlib(payload);
				if (inflated.byteLength !== expectedByteLength) {
					throw new Error(
						`EXRLoader: Inflated ZIP chunk has ${inflated.byteLength} bytes; expected ${expectedByteLength}.`
					);
				}
				return decodePredictor(deinterleave(inflated));
			}
			default:
				throw new Error(
					`EXRLoader: Unsupported EXR compression method ${header.compression}.`
				);
		}
	}

	private _copyChunkPixels(
		header: EXRHeader,
		data: Uint8Array,
		yStart: number,
		rowCount: number
	): void {
		if (data.byteLength < rowCount * this._bytesPerLine) {
			throw new Error(
				`EXRLoader: Scanline chunk is truncated (${data.byteLength} bytes).`
			);
		}

		let sourceOffset = 0;
		for (let localY = 0; localY < rowCount; localY++) {
			const imageY = yStart + localY;
			const outputY = imageY - header.dataWindow.yMin;
			for (let channelIndex = 0; channelIndex < header.channels.length; channelIndex++) {
				const channel = header.channels[channelIndex];
				const component = resolveComponentIndex(this._componentMap, channelIndex);
				const sampleSize = getPixelTypeByteSize(channel.pixelType);
				if (component >= 0) {
					this._copyChannelRow(data, sourceOffset, outputY, component, channel);
				}
				sourceOffset += this._width * sampleSize;
			}
		}
	}

	private _copyChannelRow(
		data: Uint8Array,
		sourceOffset: number,
		outputY: number,
		component: number,
		channel: EXRChannel
	): void {
		const sampleSize = getPixelTypeByteSize(channel.pixelType);
		const outputBase = outputY * this._width * 4 + component;
		for (let x = 0; x < this._width; x++) {
			const value = readPixelValue(data, sourceOffset + x * sampleSize, channel.pixelType);
			this._output[outputBase + x * 4] = sanitizePixelValue(value);
		}
	}

	private _createTexture(): Texture {
		const texture = new Texture(this._output, this._width, this._height, "HDR");
		texture.wrapS = "Repeat";
		texture.wrapT = "Clamp";
		texture.minFilter = "Linear";
		texture.magFilter = "Linear";
		texture.mipmaps = [this._output];
		texture.data = this._output;
		return texture;
	}

	private _resolveChunkCount(header: EXRHeader): number {
		if (header.chunkCount !== null) {
			if (!Number.isInteger(header.chunkCount) || header.chunkCount <= 0) {
				throw new Error(
					`EXRLoader: Invalid chunkCount ${header.chunkCount}.`
				);
			}
			return header.chunkCount;
		}
		return Math.ceil(this._height / getScanlinesPerChunk(header.compression));
	}

	private _requireHeader(): EXRHeader {
		if (!this._header) {
			throw new Error("EXRLoader: Parser was not initialized.");
		}
		return this._header;
	}
}

class BinaryReader {
	public readonly buffer: ArrayBuffer;
	public readonly view: DataView;
	public offset: number;
	private _end: number;

	constructor(buffer: ArrayBuffer, offset = 0, byteLength = buffer.byteLength) {
		this.buffer = buffer;
		this.view = new DataView(buffer, offset, byteLength);
		this.offset = offset;
		this._end = offset + byteLength;
	}

	public isEOF(): boolean {
		return this.offset >= this._end;
	}

	public seek(offset: number): void {
		if (offset < 0 || offset > this._end) {
			throw new Error(`EXRLoader: Read offset ${offset} is outside buffer.`);
		}
		this.offset = offset;
	}

	public skip(byteLength: number): void {
		this.seek(this.offset + byteLength);
	}

	public readUint32(): number {
		this._require(4);
		const value = this.view.getUint32(this.offset - this.view.byteOffset, true);
		this.offset += 4;
		return value;
	}

	public readInt32(): number {
		this._require(4);
		const value = this.view.getInt32(this.offset - this.view.byteOffset, true);
		this.offset += 4;
		return value;
	}

	public readBytes(byteLength: number): Uint8Array {
		if (!Number.isInteger(byteLength) || byteLength < 0) {
			throw new Error(`EXRLoader: Invalid byte length ${byteLength}.`);
		}
		this._require(byteLength);
		const bytes = new Uint8Array(this.buffer, this.offset, byteLength);
		this.offset += byteLength;
		return bytes;
	}

	public readNullTerminatedString(): string {
		let result = "";
		while (this.offset < this._end) {
			const value = this.view.getUint8(this.offset - this.view.byteOffset);
			this.offset++;
			if (value === 0) {
				return result;
			}
			result += String.fromCharCode(value);
		}
		throw new Error("EXRLoader: Unterminated string in EXR header.");
	}

	public sliceReader(offset: number, byteLength: number): BinaryReader {
		if (offset < 0 || offset + byteLength > this._end) {
			throw new Error("EXRLoader: Attribute payload is outside buffer.");
		}
		return new BinaryReader(this.buffer, offset, byteLength);
	}

	private _require(byteLength: number): void {
		if (this.offset + byteLength > this._end) {
			throw new Error("EXRLoader: Unexpected end of EXR data.");
		}
	}
}

function resolveEnvironmentTarget(target: EXREnvironmentTarget): Environment {
	const maybeScene = target as { environment?: Environment };
	const environment = maybeScene.environment ?? (target as Environment);
	if (
		!environment ||
		typeof environment !== "object" ||
		!("backgroundTexture" in environment) ||
		!("iblTexture" in environment)
	) {
		throw new Error(
			"EXRLoader: loadEnvironment/applyToEnvironment requires an Environment or Scene-like target."
		);
	}
	return environment;
}

function createLoadErrorFallbackTexture(): Texture {
	const texture = new Texture(new Float32Array([0, 0, 0, 1]), 1, 1, "HDR");
	texture.wrapS = "Repeat";
	texture.wrapT = "Clamp";
	texture.markAsLoadErrorFallback();
	return texture;
}

function createParseCacheKey(url: string, options: EXRParseOptions): string {
	return `texture:${url}:defaultAlpha=${sanitizeDefaultAlpha(options.defaultAlpha)}`;
}

function sanitizeDefaultAlpha(value: number | undefined): number {
	if (!Number.isFinite(value)) {
		return DEFAULT_ALPHA;
	}
	return clamp(value, 0, Number.MAX_VALUE);
}

function isSupportedCompression(value: number): boolean {
	return (
		value === EXRCompression.NO_COMPRESSION ||
		value === EXRCompression.RLE_COMPRESSION ||
		value === EXRCompression.ZIPS_COMPRESSION ||
		value === EXRCompression.ZIP_COMPRESSION
	);
}

function isSupportedPixelType(value: number): boolean {
	return (
		value === EXRPixelType.UINT ||
		value === EXRPixelType.HALF ||
		value === EXRPixelType.FLOAT
	);
}

function resolveComponentMap(channels: EXRChannel[]): EXRComponentMap {
	return {
		r: findChannelIndex(channels, "R"),
		g: findChannelIndex(channels, "G"),
		b: findChannelIndex(channels, "B"),
		a: findChannelIndex(channels, "A"),
	};
}

function findChannelIndex(channels: EXRChannel[], component: string): number {
	const exactIndex = channels.findIndex((channel) => channel.name === component);
	if (exactIndex >= 0) {
		return exactIndex;
	}
	const suffix = `.${component}`;
	return channels.findIndex((channel) => channel.name.endsWith(suffix));
}

function resolveComponentIndex(
	componentMap: EXRComponentMap,
	channelIndex: number
): number {
	if (componentMap.r === channelIndex) return 0;
	if (componentMap.g === channelIndex) return 1;
	if (componentMap.b === channelIndex) return 2;
	if (componentMap.a === channelIndex) return 3;
	return -1;
}

function calculateBytesPerLine(channels: EXRChannel[], width: number): number {
	let bytes = 0;
	for (const channel of channels) {
		bytes += width * getPixelTypeByteSize(channel.pixelType);
	}
	return bytes;
}

function getPixelTypeByteSize(pixelType: EXRPixelType): number {
	switch (pixelType) {
		case EXRPixelType.UINT:
		case EXRPixelType.FLOAT:
			return 4;
		case EXRPixelType.HALF:
			return 2;
		default:
			throw new Error(`EXRLoader: Unsupported pixel type ${pixelType}.`);
	}
}

function getScanlinesPerChunk(compression: EXRCompression): number {
	switch (compression) {
		case EXRCompression.ZIP_COMPRESSION:
			return 16;
		case EXRCompression.NO_COMPRESSION:
		case EXRCompression.RLE_COMPRESSION:
		case EXRCompression.ZIPS_COMPRESSION:
			return 1;
		default:
			throw new Error(
				`EXRLoader: Unsupported EXR compression method ${compression}.`
			);
	}
}

function readPixelValue(
	data: Uint8Array,
	offset: number,
	pixelType: EXRPixelType
): number {
	switch (pixelType) {
		case EXRPixelType.UINT:
			return readUint32LE(data, offset);
		case EXRPixelType.FLOAT:
			return readFloat32LE(data, offset);
		case EXRPixelType.HALF:
			return halfToFloat(readUint16LE(data, offset));
		default:
			return 0;
	}
}

function sanitizePixelValue(value: number): number {
	if (!Number.isFinite(value)) {
		return 0;
	}
	return Math.max(0, value);
}

function readUint16LE(data: Uint8Array, offset: number): number {
	return data[offset] | (data[offset + 1] << 8);
}

function readUint32LE(data: Uint8Array, offset: number): number {
	return (
		(data[offset] |
			(data[offset + 1] << 8) |
			(data[offset + 2] << 16) |
			(data[offset + 3] << 24)) >>>
		0
	);
}

function readFloat32LE(data: Uint8Array, offset: number): number {
	const view = new DataView(data.buffer, data.byteOffset + offset, 4);
	return view.getFloat32(0, true);
}

function halfToFloat(bits: number): number {
	const sign = (bits & 0x8000) ? -1 : 1;
	const exponent = (bits >> 10) & 0x1f;
	const fraction = bits & 0x03ff;

	if (exponent === 0) {
		if (fraction === 0) {
			return sign < 0 ? -0 : 0;
		}
		return sign * Math.pow(2, -14) * (fraction / 1024);
	}
	if (exponent === 31) {
		return fraction === 0 ? sign * Infinity : NaN;
	}
	return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function decodeRLE(data: Uint8Array, expectedByteLength: number): Uint8Array {
	const decoded = new Uint8Array(expectedByteLength);
	let sourceOffset = 0;
	let targetOffset = 0;

	while (sourceOffset < data.byteLength && targetOffset < expectedByteLength) {
		const control = data[sourceOffset++];
		if (control < 128) {
			const count = control + 1;
			if (sourceOffset + count > data.byteLength) {
				throw new Error("EXRLoader: RLE literal run exceeds chunk payload.");
			}
			decoded.set(
				data.subarray(sourceOffset, sourceOffset + count),
				targetOffset
			);
			sourceOffset += count;
			targetOffset += count;
			continue;
		}

		const count = 257 - control;
		if (sourceOffset >= data.byteLength) {
			throw new Error("EXRLoader: RLE repeat run is missing a value.");
		}
		const value = data[sourceOffset++];
		decoded.fill(value, targetOffset, targetOffset + count);
		targetOffset += count;
	}

	if (targetOffset !== expectedByteLength) {
		throw new Error(
			`EXRLoader: RLE decoded ${targetOffset} bytes; expected ${expectedByteLength}.`
		);
	}
	return decoded;
}

function deinterleave(data: Uint8Array): Uint8Array {
	const result = new Uint8Array(data.byteLength);
	let evenOffset = 0;
	let oddOffset = (data.byteLength + 1) >> 1;
	let targetOffset = 0;

	while (targetOffset < data.byteLength) {
		result[targetOffset++] = data[evenOffset++];
		if (targetOffset < data.byteLength) {
			result[targetOffset++] = data[oddOffset++];
		}
	}

	return result;
}

function decodePredictor(data: Uint8Array): Uint8Array {
	if (data.byteLength <= 1) {
		return data;
	}
	let previous = data[0];
	for (let i = 1; i < data.byteLength; i++) {
		previous = (previous + data[i] - 128) & 0xff;
		data[i] = previous;
	}
	return data;
}

async function inflateZlib(data: Uint8Array): Promise<Uint8Array> {
	if (typeof DecompressionStream !== "function") {
		throw new Error(
			"EXRLoader: ZIP/ZIPS EXR compression requires DecompressionStream support."
		);
	}

	const stream = new Blob([toArrayBuffer(data)]).stream().pipeThrough(
		new DecompressionStream("deflate")
	);
	const buffer = await new Response(stream).arrayBuffer();
	return new Uint8Array(buffer);
}

function toArrayBuffer(data: Uint8Array): ArrayBuffer {
	if (data.buffer instanceof ArrayBuffer) {
		return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
	}
	const copy = new Uint8Array(data.byteLength);
	copy.set(data);
	return copy.buffer;
}
