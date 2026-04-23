import type { Camera } from "../cameras/Camera";
import type { Matrix4 } from "../maths/Matrix4";
import type { Material } from "../materials/Material";
import { getMaterialTransmissionFactor } from "../materials/transparency";
import type { Renderer } from "../renderers/Renderer";
import {
	buildDirtyTileCoverage,
	computePostProcessInflationRadius,
	getDirtyTileCoverageAreaRatio,
	inflateDirtyRects,
	makeFullScreenRect,
	mergeDirtyRects,
	scaleFullFrameFallbackAreaRatioForPostProcess,
	tileCoverageToDirtyRects,
	type DirtyRect,
	type IncrementalRenderingOptions,
} from "./incremental";
import type { DrawPacket, PreparedScene, ResolvedFeatureState } from "./types";
import { PreparedSceneBuilder } from "./PreparedSceneBuilder";
import { PreparedSceneTileSpatialIndex } from "./PreparedSceneSpatialIndex";

interface CachedPacketState {
	signature: string;
	rect: DirtyRect | null;
}

const MATRIX_SIGNATURE_SCRATCH = new DataView(new ArrayBuffer(8));
const MATRIX_SIGNATURE_PRIME = 16777619;
const MATRIX_SIGNATURE_INIT_A = 2166136261;
const MATRIX_SIGNATURE_INIT_B = 2246822519;

export interface PreparedSceneCacheBuildInput {
	renderer: Renderer;
	viewportWidth: number;
	viewportHeight: number;
	features: ResolvedFeatureState;
	incrementalOptions: IncrementalRenderingOptions;
}

export interface PreparedSceneCacheBuildResult {
	frame: PreparedScene;
	dirtyRects: DirtyRect[];
	dirtyTiles: number[];
	dirtyTileSize: number;
	dirtyTileColumns: number;
	dirtyTileRows: number;
	dirtyAreaRatio: number;
	forceFullFrame: boolean;
	packetRects: Map<string, DirtyRect>;
}

export class PreparedSceneCache {
	private _packetStateById = new Map<string, CachedPacketState>();
	private _frameIndex = 0;

	public reset(): void {
		this._packetStateById.clear();
		this._frameIndex = 0;
	}

	public build(input: PreparedSceneCacheBuildInput): PreparedSceneCacheBuildResult {
		const frame = PreparedSceneBuilder.build(input.renderer);
		const width = Math.max(1, Math.floor(input.viewportWidth));
		const height = Math.max(1, Math.floor(input.viewportHeight));
		const fullScreenRect = makeFullScreenRect(width, height);
		const packetRects = new Map<string, DirtyRect>();
		const fullFrameTileCoverage = buildDirtyTileCoverage(
			[fullScreenRect],
			width,
			height,
			input.incrementalOptions.dirtyTileSize
		);

		if (!input.incrementalOptions.enabled) {
			this._syncPacketCacheState(frame, packetRects, width, height);
			frame.spatialIndex = this._buildSpatialIndex(
				frame,
				packetRects,
				width,
				height,
				input.incrementalOptions.dirtyTileSize
			);
			return {
				frame,
				dirtyRects: [fullScreenRect],
				dirtyTiles: fullFrameTileCoverage.dirtyTiles.slice(),
				dirtyTileSize: fullFrameTileCoverage.tileSize,
				dirtyTileColumns: fullFrameTileCoverage.tileColumns,
				dirtyTileRows: fullFrameTileCoverage.tileRows,
				dirtyAreaRatio: 1,
				forceFullFrame: true,
				packetRects,
			};
		}

		const currentPacketStateById = new Map<string, CachedPacketState>();
		const dirtyCandidates: DirtyRect[] = [];
		const visited = new Set<string>();

		this._processPackets(
			frame.opaquePackets,
			frame.camera,
			width,
			height,
			packetRects,
			currentPacketStateById,
			visited,
			dirtyCandidates
		);
		this._processPackets(
			frame.transparentPackets,
			frame.camera,
			width,
			height,
			packetRects,
			currentPacketStateById,
			visited,
			dirtyCandidates
		);

		for (const [packetId, previous] of this._packetStateById.entries()) {
			if (visited.has(packetId)) {
				continue;
			}
			if (previous.rect) {
				dirtyCandidates.push(previous.rect);
			}
		}

		this._packetStateById = currentPacketStateById;
		this._frameIndex++;
		frame.spatialIndex = this._buildSpatialIndex(
			frame,
			packetRects,
			width,
			height,
			input.incrementalOptions.dirtyTileSize
		);

		if (this._frameIndex <= 1) {
			return {
				frame,
				dirtyRects: [fullScreenRect],
				dirtyTiles: fullFrameTileCoverage.dirtyTiles.slice(),
				dirtyTileSize: fullFrameTileCoverage.tileSize,
				dirtyTileColumns: fullFrameTileCoverage.tileColumns,
				dirtyTileRows: fullFrameTileCoverage.tileRows,
				dirtyAreaRatio: 1,
				forceFullFrame: true,
				packetRects,
			};
		}

		const emptyCoverage = buildDirtyTileCoverage(
			[],
			width,
			height,
			input.incrementalOptions.dirtyTileSize
		);

		if (dirtyCandidates.length === 0) {
			return {
				frame,
				dirtyRects: [],
				dirtyTiles: [],
				dirtyTileSize: emptyCoverage.tileSize,
				dirtyTileColumns: emptyCoverage.tileColumns,
				dirtyTileRows: emptyCoverage.tileRows,
				dirtyAreaRatio: 0,
				forceFullFrame: false,
				packetRects,
			};
		}

		let dirtyRects = mergeDirtyRects(
			dirtyCandidates,
			input.incrementalOptions.maxDirtyRects,
			width,
			height
		);
		const inflationRadius = computePostProcessInflationRadius(input.features);
		if (inflationRadius > 0) {
			dirtyRects = mergeDirtyRects(
				inflateDirtyRects(dirtyRects, inflationRadius, width, height),
				input.incrementalOptions.maxDirtyRects,
				width,
				height
			);
		}

		const dirtyTileCoverage = buildDirtyTileCoverage(
			dirtyRects,
			width,
			height,
			input.incrementalOptions.dirtyTileSize
		);
		const tileDirtyRects = tileCoverageToDirtyRects(
			dirtyTileCoverage,
			input.incrementalOptions.maxDirtyRects,
			width,
			height
		);
		const dirtyAreaRatio = getDirtyTileCoverageAreaRatio(
			dirtyTileCoverage,
			width,
			height
		);
		const fallbackAreaRatio = scaleFullFrameFallbackAreaRatioForPostProcess(
			input.incrementalOptions.fullFrameFallbackAreaRatio,
			input.features
		);
		if (dirtyAreaRatio > fallbackAreaRatio) {
			return {
				frame,
				dirtyRects: [fullScreenRect],
				dirtyTiles: fullFrameTileCoverage.dirtyTiles.slice(),
				dirtyTileSize: fullFrameTileCoverage.tileSize,
				dirtyTileColumns: fullFrameTileCoverage.tileColumns,
				dirtyTileRows: fullFrameTileCoverage.tileRows,
				dirtyAreaRatio: 1,
				forceFullFrame: true,
				packetRects,
			};
		}

		return {
			frame,
			dirtyRects: tileDirtyRects,
			dirtyTiles: dirtyTileCoverage.dirtyTiles.slice(),
			dirtyTileSize: dirtyTileCoverage.tileSize,
			dirtyTileColumns: dirtyTileCoverage.tileColumns,
			dirtyTileRows: dirtyTileCoverage.tileRows,
			dirtyAreaRatio,
			forceFullFrame: false,
			packetRects,
		};
	}

	private _processPackets(
		packets: readonly DrawPacket[],
		camera: Camera,
		width: number,
		height: number,
		packetRects: Map<string, DirtyRect>,
		currentPacketStateById: Map<string, CachedPacketState>,
		visited: Set<string>,
		dirtyCandidates: DirtyRect[]
	): void {
		for (let index = 0; index < packets.length; index++) {
			const packet = packets[index];
			const signature = buildPacketSignature(packet);
			const rect = computePacketScreenRect(packet, camera, width, height);
			if (rect) {
				packetRects.set(packet.id, rect);
			}
			const currentState: CachedPacketState = {
				signature,
				rect,
			};
			currentPacketStateById.set(packet.id, currentState);
			visited.add(packet.id);

			const previous = this._packetStateById.get(packet.id);
			if (!previous) {
				if (rect) {
					dirtyCandidates.push(rect);
				}
				continue;
			}

			if (previous.signature !== signature) {
				if (previous.rect) {
					dirtyCandidates.push(previous.rect);
				}
				if (rect) {
					dirtyCandidates.push(rect);
				}
			}
		}
	}

	private _syncPacketCacheState(
		frame: PreparedScene,
		packetRects: Map<string, DirtyRect>,
		width: number,
		height: number
	): void {
		const next = new Map<string, CachedPacketState>();
		for (let index = 0; index < frame.opaquePackets.length; index++) {
			const packet = frame.opaquePackets[index];
			const rect = computePacketScreenRect(packet, frame.camera, width, height);
			if (rect) {
				packetRects.set(packet.id, rect);
			}
			next.set(packet.id, {
				signature: buildPacketSignature(packet),
				rect,
			});
		}
		for (let index = 0; index < frame.transparentPackets.length; index++) {
			const packet = frame.transparentPackets[index];
			const rect = computePacketScreenRect(packet, frame.camera, width, height);
			if (rect) {
				packetRects.set(packet.id, rect);
			}
			next.set(packet.id, {
				signature: buildPacketSignature(packet),
				rect,
			});
		}
		this._packetStateById = next;
		this._frameIndex++;
	}

	private _buildSpatialIndex(
		frame: PreparedScene,
		packetRects: ReadonlyMap<string, DirtyRect>,
		width: number,
		height: number,
		tileSize: number
	) {
		return new PreparedSceneTileSpatialIndex({
			viewportWidth: width,
			viewportHeight: height,
			tileSize,
			packetRects,
			opaquePackets: frame.opaquePackets,
			transparentPackets: frame.transparentPackets,
		});
	}
}

function buildPacketSignature(packet: DrawPacket): string {
	const matrixSignature = matrix4Signature(packet.worldMatrix);
	const materialSignature = materialSignatureOf(packet.material);
	const primitiveVisibility = packet.primitive.visible === false ? 0 : 1;
	const meshVisibility = packet.meshInstance.visible === false ? 0 : 1;
	return [
		packet.id,
		packet.pipelineKey,
		packet.primitive.geometryVersion ?? 0,
		matrixSignature,
		materialSignature,
		primitiveVisibility,
		meshVisibility,
	].join("|");
}

function matrix4Signature(matrix: Matrix4): string {
	const elements = matrix.elements;
	let hashA = MATRIX_SIGNATURE_INIT_A;
	let hashB = MATRIX_SIGNATURE_INIT_B;
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			const value = elements[row][col];
			MATRIX_SIGNATURE_SCRATCH.setFloat64(0, value, true);
			const lo = MATRIX_SIGNATURE_SCRATCH.getUint32(0, true);
			const hi = MATRIX_SIGNATURE_SCRATCH.getUint32(4, true);
			hashA = mixFnv32(hashA, lo);
			hashA = mixFnv32(hashA, hi);
			hashB = mixFnv32(hashB, hi ^ 0x9e3779b9);
			hashB = mixFnv32(hashB, lo ^ 0x85ebca6b);
		}
	}
	return `${toPaddedHex(hashA)}${toPaddedHex(hashB)}`;
}

function mixFnv32(hash: number, value: number): number {
	return Math.imul((hash ^ value) >>> 0, MATRIX_SIGNATURE_PRIME) >>> 0;
}

function toPaddedHex(value: number): string {
	return (value >>> 0).toString(16).padStart(8, "0");
}

function materialSignatureOf(material: Material): string {
	const mat = material as Material & Record<string, any>;
	const textureFields = [
		"map",
		"normalMap",
		"emissiveMap",
		"metallicRoughnessMap",
		"occlusionMap",
		"specularMap",
		"specularColorMap",
		"clearcoatMap",
		"clearcoatRoughnessMap",
		"clearcoatNormalMap",
		"sheenColorMap",
		"sheenRoughnessMap",
		"transmissionMap",
		"thicknessMap",
	];
	const textureSignature = textureFields
		.map((field) => {
			const texture = mat[field];
			const version =
				texture && typeof texture === "object" &&
					typeof texture.version === "number" ?
					texture.version
				:	-1;
			return `${field}:${version}`;
		})
		.join(",");
	const albedo = mat.albedo ?? { r: 255, g: 255, b: 255 };
	const emissive = mat.emissive ?? { r: 0, g: 0, b: 0 };
	return [
		material.type,
		material.shading,
		material.opacity.toFixed(4),
		material.alphaMode,
		material.alphaCutoff.toFixed(4),
		getMaterialTransmissionFactor(material).toFixed(4),
		material.doubleSided ? "1" : "0",
		material.cullMode,
		material.wireframe ? "1" : "0",
		material.reflectivity.toFixed(4),
		`${albedo.r},${albedo.g},${albedo.b}`,
		`${emissive.r},${emissive.g},${emissive.b}`,
		textureSignature,
	].join("|");
}

function computePacketScreenRect(
	packet: DrawPacket,
	camera: Camera,
	viewportWidth: number,
	viewportHeight: number
): DirtyRect | null {
	const worldCenter = packet.worldBounds.center;
	const worldRadius = packet.worldBounds.radius;
	if (
		!Number.isFinite(worldCenter.x) ||
		!Number.isFinite(worldCenter.y) ||
		!Number.isFinite(worldCenter.z) ||
		!Number.isFinite(worldRadius) ||
		worldRadius <= 0
	) {
		return null;
	}

	const center = projectToScreen(
		camera.viewProjectionMatrix,
		worldCenter.x,
		worldCenter.y,
		worldCenter.z,
		viewportWidth,
		viewportHeight
	);
	if (!center) {
		return null;
	}

	const right = camera.getWorldDirection(
		{ x: 1, y: 0, z: 0 },
		{ x: 0, y: 0, z: 0 }
	);
	const up = camera.getWorldDirection(
		{ x: 0, y: 1, z: 0 },
		{ x: 0, y: 0, z: 0 }
	);

	const rightPoint = projectToScreen(
		camera.viewProjectionMatrix,
		worldCenter.x + right.x * worldRadius,
		worldCenter.y + right.y * worldRadius,
		worldCenter.z + right.z * worldRadius,
		viewportWidth,
		viewportHeight
	);
	const upPoint = projectToScreen(
		camera.viewProjectionMatrix,
		worldCenter.x + up.x * worldRadius,
		worldCenter.y + up.y * worldRadius,
		worldCenter.z + up.z * worldRadius,
		viewportWidth,
		viewportHeight
	);
	if (!rightPoint || !upPoint) {
		return null;
	}

	const radiusX = Math.max(1, Math.abs(rightPoint.x - center.x));
	const radiusY = Math.max(1, Math.abs(upPoint.y - center.y));
	const minX = center.x - radiusX - 1;
	const minY = center.y - radiusY - 1;
	const maxX = center.x + radiusX + 1;
	const maxY = center.y + radiusY + 1;

	const x = Math.max(0, Math.floor(minX));
	const y = Math.max(0, Math.floor(minY));
	const width = Math.min(
		viewportWidth,
		Math.ceil(maxX)
	) - x;
	const height = Math.min(
		viewportHeight,
		Math.ceil(maxY)
	) - y;
	if (width <= 0 || height <= 0) {
		return null;
	}
	return {
		x,
		y,
		width,
		height,
	};
}

function projectToScreen(
	viewProjection: Matrix4,
	x: number,
	y: number,
	z: number,
	viewportWidth: number,
	viewportHeight: number
): { x: number; y: number; depth: number } | null {
	const matrix = viewProjection.elements;
	const clipX = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3];
	const clipY = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3];
	const clipZ = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3];
	const clipW = matrix[3][0] * x + matrix[3][1] * y + matrix[3][2] * z + matrix[3][3];

	if (!Number.isFinite(clipW) || Math.abs(clipW) < 1e-8) {
		return null;
	}

	const invW = 1 / clipW;
	const ndcX = clipX * invW;
	const ndcY = clipY * invW;
	const ndcZ = clipZ * invW;
	if (!Number.isFinite(ndcX) || !Number.isFinite(ndcY) || !Number.isFinite(ndcZ)) {
		return null;
	}

	return {
		x: (ndcX * 0.5 + 0.5) * viewportWidth,
		y: (0.5 - ndcY * 0.5) * viewportHeight,
		depth: ndcZ,
	};
}
