import { clamp } from "../maths/Common";
import type { RGBA } from "../foundation/Color";
import type { IVector3 } from "../maths/types";
import { Texture, type TextureColorSpace } from "./Texture";

export type CubeTextureFaceData = Uint8Array | Uint8ClampedArray | Float32Array;

export enum CubeTextureFace {
	PositiveX = 0,
	NegativeX = 1,
	PositiveY = 2,
	NegativeY = 3,
	PositiveZ = 4,
	NegativeZ = 5,
}

export interface CubeTextureParams {
	faces: CubeTextureFaceData[];
	faceMipmaps?: CubeTextureFaceData[][];
	size?: number;
	colorSpace?: TextureColorSpace;
}

export interface CubeFaceUV {
	face: CubeTextureFace;
	u: number;
	v: number;
}

const CUBE_TEXTURE_FACE_COUNT = 6;
const CUBE_TEXTURE_MIN_SIZE = 1;
const CUBE_TEXTURE_EPSILON = 1e-6;

/**
 * Cubemap texture container for reflection/environment sampling.
 * Face order follows:
 * +X, -X, +Y, -Y, +Z, -Z.
 */
export class CubeTexture extends Texture {
	private _facesByMipLevel: CubeTextureFaceData[][];

	constructor(params: CubeTextureParams) {
		const inferredSize = inferCubeFaceSize(params.faces);
		const requestedSize =
			typeof params.size === "number" && Number.isFinite(params.size) ?
				Math.max(CUBE_TEXTURE_MIN_SIZE, Math.floor(params.size))
			:	inferredSize;
		super(null, requestedSize, requestedSize, params.colorSpace ?? "sRGB");
		this.wrapS = "Clamp";
		this.wrapT = "Clamp";
		this.magFilter = "Linear";
		this.minFilter = "Linear";
		this._facesByMipLevel = [];
		this.setFaces(params.faces, params.faceMipmaps);
	}

	public get mipLevelCount(): number {
		return Math.max(1, this._facesByMipLevel.length);
	}

	public getFaces(level = 0): ReadonlyArray<CubeTextureFaceData> {
		if (this._facesByMipLevel.length === 0) {
			return [];
		}
		const resolvedLevel = clampCubeMipLevel(level, this._facesByMipLevel.length);
		return this._facesByMipLevel[resolvedLevel];
	}

	public setFaces(
		faces: CubeTextureFaceData[],
		faceMipmaps: CubeTextureFaceData[][] | undefined = undefined
	): void {
		assertFaceCount(faces, "CubeTexture base faces");
		const normalizedBaseSize = Math.max(
			CUBE_TEXTURE_MIN_SIZE,
			this.width | 0
		);
		assertFaceDimensions(faces, normalizedBaseSize, 0);

		const mipLevels: CubeTextureFaceData[][] = [faces.slice()];
		const requestedMipmaps = faceMipmaps ?? [];
		for (let level = 0; level < requestedMipmaps.length; level++) {
			const mipFaces = requestedMipmaps[level];
			assertFaceCount(mipFaces, `CubeTexture mip level ${level + 1}`);
			const faceSize = Math.max(1, normalizedBaseSize >> (level + 1));
			assertFaceDimensions(mipFaces, faceSize, level + 1);
			mipLevels.push(mipFaces.slice());
		}

		this._facesByMipLevel = mipLevels;
		this.minFilter =
			this._facesByMipLevel.length > 1 ? "LinearMipmapLinear" : "Linear";
		this.data = null;
		this.mipmaps = [];
		this.markNeedsUpdate();
	}

	public sampleDirectionRaw(direction: IVector3, level = 0): RGBA {
		if (this._facesByMipLevel.length === 0) {
			return { r: 255, g: 255, b: 255, a: 255 };
		}
		const resolvedLevel = clampCubeMipLevel(level, this._facesByMipLevel.length);
		const faceSize = Math.max(CUBE_TEXTURE_MIN_SIZE, this.width >> resolvedLevel);
		const faces = this._facesByMipLevel[resolvedLevel];
		const { face, u, v } = directionToCubeFaceUV(direction);
		const faceData = faces[face];
		if (!faceData) {
			return { r: 255, g: 255, b: 255, a: 255 };
		}

		const x = Math.max(
			0,
			Math.min(faceSize - 1, Math.floor(clamp(u) * faceSize))
		);
		const y = Math.max(
			0,
			Math.min(faceSize - 1, Math.floor(clamp(v) * faceSize))
		);
		const idx = (y * faceSize + x) << 2;
		return {
			r: faceData[idx] ?? 255,
			g: faceData[idx + 1] ?? 255,
			b: faceData[idx + 2] ?? 255,
			a: faceData[idx + 3] ?? 255,
		};
	}

	public sampleDirection(direction: IVector3, level = 0): RGBA {
		const sample = this.sampleDirectionRaw(direction, level);
		if (this.colorSpace === "HDR") {
			return {
				r: Math.max(0, Math.min(255, sample.r * 255)),
				g: Math.max(0, Math.min(255, sample.g * 255)),
				b: Math.max(0, Math.min(255, sample.b * 255)),
				a: 255,
			};
		}
		return sample;
	}
}

export function directionToCubeFaceUV(direction: IVector3): CubeFaceUV {
	const absX = Math.abs(direction.x);
	const absY = Math.abs(direction.y);
	const absZ = Math.abs(direction.z);
	const safeX = Math.max(absX, CUBE_TEXTURE_EPSILON);
	const safeY = Math.max(absY, CUBE_TEXTURE_EPSILON);
	const safeZ = Math.max(absZ, CUBE_TEXTURE_EPSILON);

	let face = CubeTextureFace.PositiveZ;
	let uc = 0;
	let vc = 0;

	if (absX >= absY && absX >= absZ) {
		if (direction.x >= 0) {
			face = CubeTextureFace.PositiveX;
			uc = -direction.z / safeX;
			vc = direction.y / safeX;
		} else {
			face = CubeTextureFace.NegativeX;
			uc = direction.z / safeX;
			vc = direction.y / safeX;
		}
	} else if (absY >= absX && absY >= absZ) {
		if (direction.y >= 0) {
			face = CubeTextureFace.PositiveY;
			uc = direction.x / safeY;
			vc = -direction.z / safeY;
		} else {
			face = CubeTextureFace.NegativeY;
			uc = direction.x / safeY;
			vc = direction.z / safeY;
		}
	} else if (direction.z >= 0) {
		face = CubeTextureFace.PositiveZ;
		uc = direction.x / safeZ;
		vc = direction.y / safeZ;
	} else {
		face = CubeTextureFace.NegativeZ;
		uc = -direction.x / safeZ;
		vc = direction.y / safeZ;
	}

	return {
		face,
		u: clamp(uc * 0.5 + 0.5),
		v: clamp(1 - (vc * 0.5 + 0.5)),
	};
}

function clampCubeMipLevel(level: number, totalLevels: number): number {
	const safeLevel =
		typeof level === "number" && Number.isFinite(level) ? Math.floor(level) : 0;
	return Math.max(0, Math.min(totalLevels - 1, safeLevel));
}

function inferCubeFaceSize(faces: CubeTextureFaceData[]): number {
	assertFaceCount(faces, "CubeTexture faces");
	const firstFace = faces[0];
	if (!firstFace || firstFace.length < 4) {
		throw new Error("CubeTexture requires non-empty face data.");
	}
	const estimated = Math.floor(Math.sqrt(firstFace.length / 4));
	if (!Number.isFinite(estimated) || estimated < 1) {
		throw new Error("CubeTexture failed to infer a valid face size.");
	}
	return Math.max(CUBE_TEXTURE_MIN_SIZE, estimated);
}

function assertFaceCount(
	faces: CubeTextureFaceData[],
	label: string
): void {
	if (!Array.isArray(faces) || faces.length !== CUBE_TEXTURE_FACE_COUNT) {
		throw new Error(`${label} must contain exactly 6 faces.`);
	}
}

function assertFaceDimensions(
	faces: CubeTextureFaceData[],
	faceSize: number,
	mipLevel: number
): void {
	const expectedLength = faceSize * faceSize * 4;
	for (let face = 0; face < faces.length; face++) {
		const data = faces[face];
		if (!data || data.length < expectedLength) {
			throw new Error(
				`CubeTexture mip level ${mipLevel} face ${face} has insufficient data.`
			);
		}
	}
}
