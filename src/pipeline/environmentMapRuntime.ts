import { Texture } from "../core/Texture";
import { CubeTexture } from "../core/CubeTexture";
import { clamp, sRGBToLinear } from "../maths/Common";
import type { RGBA } from "../foundation/Color";
import type { IVector3 } from "../maths/types";

interface LinearRGB {
	r: number;
	g: number;
	b: number;
}

interface CachedEquirectMapEntry {
	version: number;
	width: number;
	height: number;
	texture: Texture;
}

const _cubeToEquirectCache = new WeakMap<CubeTexture, CachedEquirectMapEntry>();
const ENVIRONMENT_DIRECTION_EPSILON = 1e-6;

export function isCubeTexture(texture: Texture | null | undefined): texture is CubeTexture {
	return texture instanceof CubeTexture;
}

export function isTextureReadyForEnvironment(
	texture: Texture | null | undefined
): texture is Texture {
	if (!texture) return false;
	if (texture.isLoadErrorFallback) return false;
	if (
		!isFinitePositiveNumber(texture.width) ||
		!isFinitePositiveNumber(texture.height)
	) {
		return false;
	}
	if (texture instanceof CubeTexture) {
		const faces = texture.getFaces(0);
		if (faces.length !== 6) return false;
		for (const face of faces) {
			if (!face || face.length <= 0) {
				return false;
			}
		}
		return true;
	}
	return !!texture.data || texture.mipmaps.length > 0;
}

export function getEnvironmentMipLevelCount(texture: Texture): number {
	if (texture instanceof CubeTexture) {
		return texture.mipLevelCount;
	}
	return Math.max(1, texture.mipmaps.length || 1);
}

export function sampleEnvironmentTextureSpecular(
	texture: Texture,
	direction: IVector3,
	roughness: number
): { r: number; g: number; b: number } {
	const mipCount = getEnvironmentMipLevelCount(texture);
	const level = clamp(roughness, 0, 1) * (mipCount - 1);
	return sampleEnvironmentTextureLevelLinear(texture, direction, level);
}

/**
 * Samples an environment texture and returns linear radiance without clamping
 * Float32-backed HDR sources to the legacy 0..255 RGBA contract.
 *
 * @param texture - Environment texture to sample.
 * @param direction - World-space direction used for cube or equirect lookup.
 * @param level - Mipmap level to sample. Fractional values are floored.
 * @returns Linear RGB radiance in 0..1 or HDR range.
 */
export function sampleEnvironmentTextureLevelLinear(
	texture: Texture,
	direction: IVector3,
	level = 0
): LinearRGB {
	const normalized = normalizeDirection(direction);
	if (texture instanceof CubeTexture) {
		const resolvedLevel = Math.max(
			0,
			Math.min(texture.mipLevelCount - 1, Math.floor(level))
		);
		const faceData = texture.getFaces(resolvedLevel)[0];
		const sample = texture.sampleDirectionRaw(normalized, resolvedLevel);
		return convertRawSampleToLinear(
			sample,
			texture.colorSpace,
			faceData instanceof Float32Array
		);
	}
	const uv = directionToEquirectUV(normalized);
	return sampleEquirectTextureLevelLinear(texture, uv.u, uv.v, level);
}

export function sampleEnvironmentTextureLevel(
	texture: Texture,
	direction: IVector3,
	level = 0
): RGBA {
	const normalized = normalizeDirection(direction);
	if (texture instanceof CubeTexture) {
		return texture.sampleDirection(normalized, level);
	}
	const uv = directionToEquirectUV(normalized);
	return texture.sampleLevel(uv.u, uv.v, level);
}

export function ensureEnvironmentTextureEquirect(
	texture: Texture | null | undefined
): Texture | null {
	if (!texture) return null;
	if (!(texture instanceof CubeTexture)) return texture;

	const targetWidth = Math.max(4, texture.width * 4);
	const targetHeight = Math.max(2, texture.height * 2);
	const cached = _cubeToEquirectCache.get(texture);
	if (
		cached &&
		cached.version === texture.version &&
		cached.width === targetWidth &&
		cached.height === targetHeight
	) {
		return cached.texture;
	}

	const converted = convertCubeTextureToEquirect(
		texture,
		targetWidth,
		targetHeight
	);
	_cubeToEquirectCache.set(texture, {
		version: texture.version,
		width: targetWidth,
		height: targetHeight,
		texture: converted,
	});
	return converted;
}

export function convertCubeTextureToEquirect(
	texture: CubeTexture,
	targetWidth = Math.max(4, texture.width * 4),
	targetHeight = Math.max(2, texture.height * 2)
): Texture {
	const mipCount = texture.mipLevelCount;
	const mipmaps: (Uint8Array | Uint8ClampedArray | Float32Array)[] = [];

	for (let level = 0; level < mipCount; level++) {
		const levelWidth = Math.max(1, targetWidth >> level);
		const levelHeight = Math.max(1, targetHeight >> level);
		const faceLevel = Math.min(level, texture.mipLevelCount - 1);
		const faceData = texture.getFaces(faceLevel)[0];
		const convertedLevel = createCompatibleMipBuffer(
			faceData,
			levelWidth * levelHeight * 4
		);

		for (let y = 0; y < levelHeight; y++) {
			const v = (y + 0.5) / levelHeight;
			for (let x = 0; x < levelWidth; x++) {
				const u = (x + 0.5) / levelWidth;
				const direction = directionFromEquirectUV(u, v);
				const sample = texture.sampleDirectionRaw(direction, faceLevel);
				const index = (y * levelWidth + x) << 2;
				writeSample(convertedLevel, index, sample);
			}
		}

		mipmaps.push(convertedLevel);
	}

	const converted = new Texture(
		mipmaps[0] as Texture["data"],
		targetWidth,
		targetHeight,
		texture.colorSpace
	);
	converted.wrapS = "Repeat";
	converted.wrapT = "Clamp";
	converted.minFilter = "Linear";
	converted.magFilter = "Linear";
	converted.mipmaps = mipmaps;
	converted.data = mipmaps[0] as Texture["data"];
	if (texture.isLoadErrorFallback) {
		converted.markAsLoadErrorFallback();
	}
	return converted;
}

export function directionToEquirectUV(direction: IVector3): { u: number; v: number } {
	const phi = Math.atan2(direction.x, direction.z);
	const theta = Math.acos(clamp(direction.y, -1, 1));
	return {
		u: (phi + Math.PI) / (2 * Math.PI),
		v: theta / Math.PI,
	};
}

export function directionFromEquirectUV(u: number, v: number): IVector3 {
	const phi = u * (2 * Math.PI) - Math.PI;
	const theta = v * Math.PI;
	const sinTheta = Math.sin(theta);
	return normalizeDirection({
		x: sinTheta * Math.sin(phi),
		y: Math.cos(theta),
		z: sinTheta * Math.cos(phi),
	});
}

function sampleEquirectTextureLevelLinear(
	texture: Texture,
	u: number,
	v: number,
	level = 0
): LinearRGB {
	if (texture.mipmaps.length === 0) {
		return { r: 1, g: 1, b: 1 };
	}

	const maxLevel = texture.mipmaps.length - 1;
	const resolvedLevel = Math.max(0, Math.min(maxLevel, Math.floor(level)));
	const levelWidth = Math.max(1, texture.width >> resolvedLevel);
	const levelHeight = Math.max(1, texture.height >> resolvedLevel);
	const data =
		texture.mipmaps[resolvedLevel] ??
		(resolvedLevel === 0 ? texture.data : null) ??
		texture.mipmaps[0] ??
		null;
	if (!data) {
		return { r: 1, g: 1, b: 1 };
	}

	let uu = u * texture.repeat.x;
	let vv = v * texture.repeat.y;

	if (texture.rotation !== 0) {
		const c = Math.cos(texture.rotation);
		const s = Math.sin(texture.rotation);
		const rotatedU = uu * c - vv * s;
		const rotatedV = uu * s + vv * c;
		uu = rotatedU;
		vv = rotatedV;
	}

	uu += texture.offset.x;
	vv += texture.offset.y;

	if (texture.wrapS === "Repeat") {
		uu = uu - Math.floor(uu);
	} else if (texture.wrapS === "MirroredRepeat") {
		const iter = Math.floor(uu);
		uu = uu - iter;
		if (Math.abs(iter) % 2 === 1) uu = 1.0 - uu;
	} else {
		uu = clamp(uu);
	}

	if (texture.wrapT === "Repeat") {
		vv = vv - Math.floor(vv);
	} else if (texture.wrapT === "MirroredRepeat") {
		const iter = Math.floor(vv);
		vv = vv - iter;
		if (Math.abs(iter) % 2 === 1) vv = 1.0 - vv;
	} else {
		vv = clamp(vv);
	}

	const x = Math.min(levelWidth - 1, Math.floor(uu * levelWidth));
	const y = Math.min(levelHeight - 1, Math.floor(vv * levelHeight));
	const idx = (y * levelWidth + x) << 2;
	return convertRawSampleToLinear(
		{
			r: data[idx] ?? 255,
			g: data[idx + 1] ?? 255,
			b: data[idx + 2] ?? 255,
			a: data[idx + 3] ?? 255,
		},
		texture.colorSpace,
		data instanceof Float32Array
	);
}

function convertRawSampleToLinear(
	sample: RGBA,
	colorSpace: Texture["colorSpace"],
	sourceIsFloat: boolean
): LinearRGB {
	if (sourceIsFloat) {
		if (colorSpace === "sRGB") {
			return {
				r: sRGBToLinear(sample.r),
				g: sRGBToLinear(sample.g),
				b: sRGBToLinear(sample.b),
			};
		}
		return {
			r: sample.r,
			g: sample.g,
			b: sample.b,
		};
	}
	if (colorSpace === "sRGB") {
		return {
			r: sRGBToLinear(sample.r / 255),
			g: sRGBToLinear(sample.g / 255),
			b: sRGBToLinear(sample.b / 255),
		};
	}
	return {
		r: sample.r / 255,
		g: sample.g / 255,
		b: sample.b / 255,
	};
}

function writeSample(
	target: Uint8Array | Uint8ClampedArray | Float32Array,
	index: number,
	sample: RGBA
): void {
	if (target instanceof Float32Array) {
		target[index] = sample.r;
		target[index + 1] = sample.g;
		target[index + 2] = sample.b;
		target[index + 3] = sample.a;
		return;
	}
	target[index] = clamp(Math.round(sample.r), 0, 255);
	target[index + 1] = clamp(Math.round(sample.g), 0, 255);
	target[index + 2] = clamp(Math.round(sample.b), 0, 255);
	target[index + 3] = clamp(Math.round(sample.a), 0, 255);
}

function createCompatibleMipBuffer(
	faceData: Uint8Array | Uint8ClampedArray | Float32Array | undefined,
	pixelCount: number
): Uint8Array | Uint8ClampedArray | Float32Array {
	if (faceData instanceof Float32Array) {
		return new Float32Array(pixelCount);
	}
	if (faceData instanceof Uint8ClampedArray) {
		return new Uint8ClampedArray(pixelCount);
	}
	return new Uint8Array(pixelCount);
}

function normalizeDirection(direction: IVector3): IVector3 {
	const length = Math.hypot(direction.x, direction.y, direction.z);
	if (length <= ENVIRONMENT_DIRECTION_EPSILON) {
		return { x: 0, y: 0, z: 1 };
	}
	const invLength = 1 / length;
	return {
		x: direction.x * invLength,
		y: direction.y * invLength,
		z: direction.z * invLength,
	};
}

function isFinitePositiveNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value) && value > 0;
}
