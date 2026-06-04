import type { Camera } from "../cameras/Camera";
import type { Matrix4 } from "../maths/Matrix4";
import type { Material } from "../materials/Material";
import { getMaterialTransmissionFactor } from "../materials/transparency";
import { DECAL_CHANNELS, resolveDecalChannelBlendMode } from "../decals";
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
import type {
	DecalPacket,
	DrawPacket,
	PreparedScene,
	ResolvedFeatureState,
} from "./types";
import type { ResolvedPostProcessState } from "../postprocess";
import { PreparedSceneBuilder } from "./PreparedSceneBuilder";
import { PreparedSceneTileSpatialIndex } from "./PreparedSceneSpatialIndex";

interface CachedSignatureState {
	matrixSignatureA: number;
	matrixSignatureB: number;
	materialSignatureA: number;
	materialSignatureB: number;
	rect: DirtyRect | null;
}

interface CachedPacketState extends CachedSignatureState {
	pipelineKey: string;
	geometryVersion: number;
	primitiveVisibility: number;
	meshVisibility: number;
}

interface CachedDecalState extends CachedSignatureState {}

type PacketStateObserver = (
	packetId: string,
	currentState: CachedPacketState,
	previousState: CachedPacketState | undefined
) => void;

type DecalStateObserver = (
	packetId: string,
	currentState: CachedDecalState,
	previousState: CachedDecalState | undefined
) => void;

const SIGNATURE_FLOAT64_SCRATCH = new DataView(new ArrayBuffer(8));
const SIGNATURE_PRIME = 16777619;
const SIGNATURE_INIT_A = 2166136261;
const SIGNATURE_INIT_B = 2246822519;
const SIGNATURE_MIX_B_A = 0x9e3779b9;
const SIGNATURE_MIX_B_B = 0x85ebca6b;
const MATERIAL_TEXTURE_FIELDS = [
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
	"iridescenceMap",
	"iridescenceThicknessMap",
	"thicknessMap",
] as const;
const MATERIAL_TEXTURE_FIELD_HASHES = MATERIAL_TEXTURE_FIELDS.map(hashStaticToken32);
const DECAL_CHANNEL_HASHES = DECAL_CHANNELS.map(hashStaticToken32);

export interface PreparedSceneCacheBuildInput {
	renderer: Renderer;
	viewportWidth: number;
	viewportHeight: number;
	features: ResolvedFeatureState;
	postProcess: ResolvedPostProcessState;
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
	private _decalStateById = new Map<string, CachedDecalState>();
	private _frameIndex = 0;

	public reset(): void {
		this._packetStateById.clear();
		this._decalStateById.clear();
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
			this._syncCacheState(frame, packetRects, width, height);
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
		const currentDecalStateById = new Map<string, CachedDecalState>();
		const dirtyCandidates: DirtyRect[] = [];
		const visited = new Set<string>();
		const visitedDecals = new Set<string>();

		this._processFramePackets(
			frame,
			width,
			height,
			packetRects,
			currentPacketStateById,
			visited,
			(_packetId, currentState, previous) => {
				if (!previous) {
					if (currentState.rect) {
						dirtyCandidates.push(currentState.rect);
					}
					return;
				}

				if (!packetStateEquals(previous, currentState)) {
					if (previous.rect) {
						dirtyCandidates.push(previous.rect);
					}
					if (currentState.rect) {
						dirtyCandidates.push(currentState.rect);
					}
				}
			}
		);

		this._processFrameDecals(
			frame,
			width,
			height,
			currentDecalStateById,
			visitedDecals,
			(_packetId, currentState, previous) => {
				if (!previous) {
					if (currentState.rect) {
						dirtyCandidates.push(currentState.rect);
					}
					return;
				}

				if (!decalStateEquals(previous, currentState)) {
					if (previous.rect) {
						dirtyCandidates.push(previous.rect);
					}
					if (currentState.rect) {
						dirtyCandidates.push(currentState.rect);
					}
				}
			}
		);

		for (const [packetId, previous] of this._packetStateById.entries()) {
			if (visited.has(packetId)) {
				continue;
			}
			if (previous.rect) {
				dirtyCandidates.push(previous.rect);
			}
		}

		for (const [packetId, previous] of this._decalStateById.entries()) {
			if (visitedDecals.has(packetId)) {
				continue;
			}
			if (previous.rect) {
				dirtyCandidates.push(previous.rect);
			}
		}

		this._packetStateById = currentPacketStateById;
		this._decalStateById = currentDecalStateById;
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
		const inflationRadius = computePostProcessInflationRadius(input.postProcess);
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
			input.postProcess
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

	private _processFramePackets(
		frame: PreparedScene,
		width: number,
		height: number,
		packetRects: Map<string, DirtyRect>,
		currentPacketStateById: Map<string, CachedPacketState>,
		visited?: Set<string>,
		observer?: PacketStateObserver
	): void {
		this._processPacketList(
			frame.opaquePackets,
			frame.camera,
			width,
			height,
			packetRects,
			currentPacketStateById,
			visited,
			observer
		);
		this._processPacketList(
			frame.transparentPackets,
			frame.camera,
			width,
			height,
			packetRects,
			currentPacketStateById,
			visited,
			observer
		);
	}

	private _processFrameDecals(
		frame: PreparedScene,
		width: number,
		height: number,
		currentDecalStateById: Map<string, CachedDecalState>,
		visited?: Set<string>,
		observer?: DecalStateObserver
	): void {
		for (const packet of frame.decalPackets) {
			const rect = computePacketScreenRect(packet, frame.camera, width, height);
			const currentState = createDecalState(packet, rect);
			currentDecalStateById.set(packet.id, currentState);
			if (visited) {
				visited.add(packet.id);
			}
			if (observer) {
				observer(packet.id, currentState, this._decalStateById.get(packet.id));
			}
		}
	}

	private _processPacketList(
		packets: readonly DrawPacket[],
		camera: Camera,
		width: number,
		height: number,
		packetRects: Map<string, DirtyRect>,
		currentPacketStateById: Map<string, CachedPacketState>,
		visited?: Set<string>,
		observer?: PacketStateObserver
	): void {
		for (let index = 0; index < packets.length; index++) {
			const packet = packets[index];
			const rect = computePacketScreenRect(packet, camera, width, height);
			if (rect) {
				packetRects.set(packet.id, rect);
			}
			const currentState = createPacketState(packet, rect);
			currentPacketStateById.set(packet.id, currentState);
			if (visited) {
				visited.add(packet.id);
			}
			if (observer) {
				observer(packet.id, currentState, this._packetStateById.get(packet.id));
			}
		}
	}

	private _syncCacheState(
		frame: PreparedScene,
		packetRects: Map<string, DirtyRect>,
		width: number,
		height: number
	): void {
		const next = new Map<string, CachedPacketState>();
		this._processFramePackets(frame, width, height, packetRects, next);
		const nextDecals = new Map<string, CachedDecalState>();
		this._processFrameDecals(frame, width, height, nextDecals);
		this._packetStateById = next;
		this._decalStateById = nextDecals;
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

function createPacketState(
	packet: DrawPacket,
	rect: DirtyRect | null
): CachedPacketState {
	const state: CachedPacketState = {
		pipelineKey: packet.pipelineKey,
		geometryVersion: packet.primitive.geometryVersion ?? 0,
		matrixSignatureA: SIGNATURE_INIT_A,
		matrixSignatureB: SIGNATURE_INIT_B,
		materialSignatureA: SIGNATURE_INIT_A,
		materialSignatureB: SIGNATURE_INIT_B,
		primitiveVisibility: packet.primitive.visible === false ? 0 : 1,
		meshVisibility: packet.meshInstance.visible === false ? 0 : 1,
		rect,
	};
	writeMatrix4Signature(state, packet.worldMatrix);
	writeMaterialSignature(state, packet.material);
	return state;
}

function createDecalState(
	packet: DecalPacket,
	rect: DirtyRect | null
): CachedDecalState {
	const state: CachedDecalState = {
		matrixSignatureA: SIGNATURE_INIT_A,
		matrixSignatureB: SIGNATURE_INIT_B,
		materialSignatureA: SIGNATURE_INIT_A,
		materialSignatureB: SIGNATURE_INIT_B,
		rect,
	};
	writeMatrix4Signature(state, packet.worldMatrix);
	writeMaterialSignature(state, packet.material);
	mixMaterialUint32(state, packet.receiverLayerMask >>> 0);
	mixMaterialFloat(state, packet.priority);
	mixMaterialFloat(state, packet.opacity);
	mixMaterialFloat(state, packet.edgeFade);
	mixMaterialUint32(state, packet.sceneOrder >>> 0);
	writeDecalChannelBlendModeSignature(state, packet);
	return state;
}

function packetStateEquals(
	left: CachedPacketState,
	right: CachedPacketState
): boolean {
	return (
		left.pipelineKey === right.pipelineKey &&
		left.geometryVersion === right.geometryVersion &&
		left.matrixSignatureA === right.matrixSignatureA &&
		left.matrixSignatureB === right.matrixSignatureB &&
		left.materialSignatureA === right.materialSignatureA &&
		left.materialSignatureB === right.materialSignatureB &&
		left.primitiveVisibility === right.primitiveVisibility &&
		left.meshVisibility === right.meshVisibility
	);
}

function decalStateEquals(
	left: CachedDecalState,
	right: CachedDecalState
): boolean {
	return (
		left.matrixSignatureA === right.matrixSignatureA &&
		left.matrixSignatureB === right.matrixSignatureB &&
		left.materialSignatureA === right.materialSignatureA &&
		left.materialSignatureB === right.materialSignatureB
	);
}

function writeMatrix4Signature(
	state: CachedSignatureState,
	matrix: Matrix4
): void {
	const elements = matrix.elements;
	let hashA = SIGNATURE_INIT_A;
	let hashB = SIGNATURE_INIT_B;
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			const value = elements[row][col];
			SIGNATURE_FLOAT64_SCRATCH.setFloat64(0, value, true);
			const lo = SIGNATURE_FLOAT64_SCRATCH.getUint32(0, true);
			const hi = SIGNATURE_FLOAT64_SCRATCH.getUint32(4, true);
			hashA = mixFnv32(hashA, lo);
			hashA = mixFnv32(hashA, hi);
			hashB = mixFnv32(hashB, hi ^ SIGNATURE_MIX_B_A);
			hashB = mixFnv32(hashB, lo ^ SIGNATURE_MIX_B_B);
		}
	}
	state.matrixSignatureA = hashA;
	state.matrixSignatureB = hashB;
}

function writeMaterialSignature(
	state: CachedSignatureState,
	material: Material
): void {
	const mat = material as Material & Record<string, unknown>;
	state.materialSignatureA = SIGNATURE_INIT_A;
	state.materialSignatureB = SIGNATURE_INIT_B;

	mixMaterialString(state, material.type);
	mixMaterialString(state, material.shading);
	mixMaterialFloat(state, material.opacity);
	mixMaterialString(state, material.alphaMode);
	mixMaterialFloat(state, material.alphaCutoff);
	mixMaterialFloat(state, getMaterialTransmissionFactor(material));
	mixMaterialUint32(state, material.depthWrite ? 1 : 0);
	mixMaterialFloat(state, resolveMaterialNumber(mat.iridescenceFactor, 0));
	mixMaterialFloat(state, resolveMaterialNumber(mat.iridescenceIor, 1.3));
	mixMaterialFloat(
		state,
		resolveMaterialNumber(mat.iridescenceThicknessMinimum, 100)
	);
	mixMaterialFloat(
		state,
		resolveMaterialNumber(mat.iridescenceThicknessMaximum, 400)
	);
	mixMaterialUint32(state, material.doubleSided ? 1 : 0);
	mixMaterialString(state, material.cullMode);
	mixMaterialUint32(state, material.wireframe ? 1 : 0);
	mixMaterialFloat(state, material.reflectivity);

	const albedo = (mat.albedo ?? { r: 255, g: 255, b: 255 }) as {
		r?: unknown;
		g?: unknown;
		b?: unknown;
	};
	const emissive = (mat.emissive ?? { r: 0, g: 0, b: 0 }) as {
		r?: unknown;
		g?: unknown;
		b?: unknown;
	};
	mixMaterialFloat(state, resolveColorChannel(albedo.r, 255));
	mixMaterialFloat(state, resolveColorChannel(albedo.g, 255));
	mixMaterialFloat(state, resolveColorChannel(albedo.b, 255));
	mixMaterialFloat(state, resolveColorChannel(emissive.r, 0));
	mixMaterialFloat(state, resolveColorChannel(emissive.g, 0));
	mixMaterialFloat(state, resolveColorChannel(emissive.b, 0));

	for (let index = 0; index < MATERIAL_TEXTURE_FIELDS.length; index++) {
		const field = MATERIAL_TEXTURE_FIELDS[index];
		mixMaterialUint32(state, MATERIAL_TEXTURE_FIELD_HASHES[index]);
		mixMaterialFloat(state, resolveTextureVersion(mat[field]));
	}
}

function writeDecalChannelBlendModeSignature(
	state: CachedSignatureState,
	packet: DecalPacket
): void {
	for (let index = 0; index < DECAL_CHANNELS.length; index++) {
		const channel = DECAL_CHANNELS[index];
		mixMaterialUint32(state, DECAL_CHANNEL_HASHES[index]);
		mixMaterialString(
			state,
			resolveDecalChannelBlendMode(packet.channelBlendModes, channel)
		);
	}
}

function resolveColorChannel(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return value;
}

function resolveMaterialNumber(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return value;
}

function resolveTextureVersion(value: unknown): number {
	if (!value || typeof value !== "object") {
		return -1;
	}
	const version = (value as { version?: unknown }).version;
	if (typeof version !== "number" || !Number.isFinite(version)) {
		return -1;
	}
	return version;
}

function mixMaterialString(state: CachedSignatureState, value: unknown): void {
	const normalized = typeof value === "string" ? value : String(value ?? "");
	let hashA = mixFnv32(state.materialSignatureA, normalized.length);
	let hashB = mixFnv32(
		state.materialSignatureB,
		normalized.length ^ SIGNATURE_MIX_B_A
	);
	for (let index = 0; index < normalized.length; index++) {
		const charCode = normalized.charCodeAt(index);
		hashA = mixFnv32(hashA, charCode);
		hashB = mixFnv32(hashB, charCode ^ SIGNATURE_MIX_B_A);
	}
	state.materialSignatureA = hashA;
	state.materialSignatureB = hashB;
}

function mixMaterialUint32(state: CachedSignatureState, value: number): void {
	const normalized = value >>> 0;
	state.materialSignatureA = mixFnv32(state.materialSignatureA, normalized);
	state.materialSignatureB = mixFnv32(
		state.materialSignatureB,
		normalized ^ SIGNATURE_MIX_B_A
	);
}

function mixMaterialFloat(state: CachedSignatureState, value: number): void {
	SIGNATURE_FLOAT64_SCRATCH.setFloat64(0, value, true);
	const lo = SIGNATURE_FLOAT64_SCRATCH.getUint32(0, true);
	const hi = SIGNATURE_FLOAT64_SCRATCH.getUint32(4, true);
	state.materialSignatureA = mixFnv32(state.materialSignatureA, lo);
	state.materialSignatureA = mixFnv32(state.materialSignatureA, hi);
	state.materialSignatureB = mixFnv32(
		state.materialSignatureB,
		hi ^ SIGNATURE_MIX_B_A
	);
	state.materialSignatureB = mixFnv32(
		state.materialSignatureB,
		lo ^ SIGNATURE_MIX_B_B
	);
}

function mixFnv32(hash: number, value: number): number {
	return Math.imul((hash ^ value) >>> 0, SIGNATURE_PRIME) >>> 0;
}

function hashStaticToken32(value: string): number {
	let hash = SIGNATURE_INIT_A;
	hash = mixFnv32(hash, value.length);
	for (let index = 0; index < value.length; index++) {
		hash = mixFnv32(hash, value.charCodeAt(index));
	}
	return hash;
}

function computePacketScreenRect(
	packet: { worldBounds: DrawPacket["worldBounds"] },
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
