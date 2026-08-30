import type { Camera } from "../cameras/Camera";
import type { Matrix4 } from "../maths/Matrix4";
import type { Material } from "../materials/Material";
import { getMaterialTransmissionFactor } from "../materials/transparency";
import {
	DECAL_CHANNELS,
	resolveDecalChannelBlendMode,
} from "../decals";
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
	DrawSubmission,
	OcclusionCullingOptions,
	PreparedScene,
	ResolvedFeatureState,
} from "./types";
import type { OcclusionVisibilityProvider } from "./OcclusionCulling";
import { normalizeOcclusionCullingOptions } from "./OcclusionCulling";
import type { ResolvedPostProcessState } from "../postprocess";
import {
	PreparedSceneBuilder,
	PreparedScenePacketCache,
} from "./PreparedSceneBuilder";
import type { PreparedSceneBuildSource } from "./PreparedSceneBuilder";
import { PreparedSceneTileSpatialIndex } from "./PreparedSceneSpatialIndex";
import { computePacketScreenRect } from "./screenBounds";

interface MatrixSignatureState {
	matrixSignatureA: number;
	matrixSignatureB: number;
}

interface CachedSignatureState extends MatrixSignatureState {
	materialSignatureA: number;
	materialSignatureB: number;
	rect: DirtyRect | null;
}

interface CachedPacketState extends CachedSignatureState {
	pipelineKey: string;
	geometryVersion: number;
	deformationRevision: number;
	boundsCenterX: number;
	boundsCenterY: number;
	boundsCenterZ: number;
	boundsRadius: number;
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
	source: PreparedSceneBuildSource;
	viewportWidth: number;
	viewportHeight: number;
	features: ResolvedFeatureState;
	postProcess: ResolvedPostProcessState;
	incrementalOptions: IncrementalRenderingOptions;
	occlusionVisibilityProvider?: OcclusionVisibilityProvider | null;
	occlusionCullingOptions?: OcclusionCullingOptions;
	/** Renderer-accumulated backend invalidation regions for this frame. */
	additionalDirtyRects?: readonly DirtyRect[];
	/** Forces conservative full coverage for an unbounded backend invalidation. */
	forceFullFrame?: boolean;
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
	private readonly _preparedPackets = new PreparedScenePacketCache();
	private _packetStateById = new Map<string, CachedPacketState>();
	private _decalStateById = new Map<string, CachedDecalState>();
	private _reusableFrame: PreparedScene | null = null;
	private _reusableScene: PreparedSceneBuildSource["scene"] | null = null;
	private _reusableSubmissionsById = new Map<string, DrawSubmission>();
	private _frameIndex = 0;
	private _cameraSignatureA: number | null = null;
	private _cameraSignatureB: number | null = null;

	public reset(): void {
		this._preparedPackets.clear();
		this._packetStateById.clear();
		this._decalStateById.clear();
		this._reusableFrame = null;
		this._reusableScene = null;
		this._reusableSubmissionsById.clear();
		this._frameIndex = 0;
		this._cameraSignatureA = null;
		this._cameraSignatureB = null;
	}

	public build(input: PreparedSceneCacheBuildInput): PreparedSceneCacheBuildResult {
		this._preparedPackets.beginFrame();
		const buildOptions = {
			viewportWidth: input.viewportWidth,
			viewportHeight: input.viewportHeight,
			occlusionVisibilityProvider: input.occlusionVisibilityProvider ?? null,
			occlusionCullingOptions:
				input.occlusionVisibilityProvider ?
					normalizeOcclusionCullingOptions(input.occlusionCullingOptions)
				:	undefined,
			packetCache: this._preparedPackets,
		};
		let frame: PreparedScene;
		try {
			const reusable = this._resolveReusableFrame(input.source);
			if (reusable) {
				frame = PreparedSceneBuilder.buildView(
					reusable,
					input.source.camera,
					{
						...buildOptions,
						visibilityScene: input.source.scene,
						decals: input.source.scene.getDecals(),
					},
				);
				frame = {
					...frame,
					hasActiveAnimations: input.source.hasActiveAnimations,
					deformationStates: input.source.deformationStates ?? null,
				};
			} else {
				frame = PreparedSceneBuilder.build(input.source, buildOptions);
				if (input.source?.scene && input.source.camera) {
					this._reusableFrame = frame;
					this._reusableScene = input.source.scene;
					this._reusableSubmissionsById = new Map(
						frame.submissions.map((submission) => [submission.id, submission]),
					);
				} else {
					this._reusableFrame = null;
					this._reusableScene = null;
					this._reusableSubmissionsById.clear();
				}
			}
		} finally {
			this._preparedPackets.endFrame();
		}
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
		const cameraSignature = createMatrixSignature(frame.camera.viewProjectionMatrix);
		const cameraChanged =
			this._cameraSignatureA !== null &&
			this._cameraSignatureB !== null &&
			(this._cameraSignatureA !== cameraSignature.matrixSignatureA ||
				this._cameraSignatureB !== cameraSignature.matrixSignatureB);
		this._cameraSignatureA = cameraSignature.matrixSignatureA;
		this._cameraSignatureB = cameraSignature.matrixSignatureB;

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
		const dirtyCandidates: DirtyRect[] = (input.additionalDirtyRects ?? []).map(
			(rect) => ({
				x: rect.x,
				y: rect.y,
				width: rect.width,
				height: rect.height,
			}),
		);
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

		if (this._frameIndex <= 1 || cameraChanged || input.forceFullFrame === true) {
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

	private _resolveReusableFrame(
		source: PreparedSceneBuildSource | undefined,
	): PreparedScene | null {
		const reusable = this._reusableFrame;
		if (!source?.scene || !source.camera || !reusable) return null;
		if (source.scene !== this._reusableScene) return null;

		const meshInstances = source.scene.getMeshInstances();
		if (!sameReferenceList(meshInstances, reusable.meshInstances)) return null;
		if (!this._preparedPackets.canReuseSubmissions(
			meshInstances,
			this._reusableSubmissionsById,
			source.deformationStates ?? null,
		)) return null;

		const lights = source.scene.ecs.findLights();
		if (!sameReferenceList(lights, reusable.lights)) return null;
		const particleSystems = source.scene.ecs.findParticleSystems();
		if (!sameReferenceList(particleSystems, reusable.particleSystems)) return null;
		if (!preparedEnvironmentMatchesScene(reusable, source.scene.environment)) {
			return null;
		}
		return reusable;
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
			const submission = packet.submission;
			const rect = computePacketScreenRect(submission, camera, width, height);
			if (rect) {
				packetRects.set(submission.id, rect);
			}
			const currentState = createPacketState(packet, rect);
			currentPacketStateById.set(submission.id, currentState);
			if (visited) {
				visited.add(submission.id);
			}
			if (observer) {
				observer(
					submission.id,
					currentState,
					this._packetStateById.get(submission.id),
				);
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

function sameReferenceList<T>(left: readonly T[], right: readonly T[]): boolean {
	if (left.length !== right.length) return false;
	for (let index = 0; index < left.length; index++) {
		if (left[index] !== right[index]) return false;
	}
	return true;
}

function preparedEnvironmentMatchesScene(
	frame: PreparedScene,
	environment: PreparedSceneBuildSource["scene"]["environment"],
): boolean {
	const prepared = frame.environment;
	const tint = environment.backgroundTintLinear;
	return (
		prepared.backgroundEnabled === environment.backgroundEnabled &&
		prepared.lightingEnabled === environment.lightingEnabled &&
		prepared.backgroundTexture === environment.backgroundTexture &&
		prepared.iblTexture === environment.iblTexture &&
		prepared.backgroundStrength === environment.backgroundStrength &&
		prepared.diffuseStrength === environment.diffuseStrength &&
		prepared.specularStrength === environment.specularStrength &&
		prepared.backgroundExposure === environment.backgroundExposure &&
		prepared.backgroundTintLinear.r === tint.r &&
		prepared.backgroundTintLinear.g === tint.g &&
		prepared.backgroundTintLinear.b === tint.b
	);
}

function createPacketState(
	packet: DrawPacket,
	rect: DirtyRect | null
): CachedPacketState {
	const state: CachedPacketState = {
		pipelineKey: packet.submission.material.pipelineKey,
		geometryVersion: packet.submission.geometry.version,
		deformationRevision: packet.submission.deformation.revision,
		boundsCenterX: packet.submission.worldBounds.center.x,
		boundsCenterY: packet.submission.worldBounds.center.y,
		boundsCenterZ: packet.submission.worldBounds.center.z,
		boundsRadius: packet.submission.worldBounds.radius,
		matrixSignatureA: SIGNATURE_INIT_A,
		matrixSignatureB: SIGNATURE_INIT_B,
		materialSignatureA: SIGNATURE_INIT_A,
		materialSignatureB: SIGNATURE_INIT_B,
		rect,
	};
	writeMatrix4Signature(state, packet.submission.instance.worldMatrix);
	writeMaterialSignature(state, packet.submission.material.effective);
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
		left.deformationRevision === right.deformationRevision &&
		left.boundsCenterX === right.boundsCenterX &&
		left.boundsCenterY === right.boundsCenterY &&
		left.boundsCenterZ === right.boundsCenterZ &&
		left.boundsRadius === right.boundsRadius &&
		left.matrixSignatureA === right.matrixSignatureA &&
		left.matrixSignatureB === right.matrixSignatureB &&
		left.materialSignatureA === right.materialSignatureA &&
		left.materialSignatureB === right.materialSignatureB
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
	state: MatrixSignatureState,
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

function createMatrixSignature(matrix: Matrix4): MatrixSignatureState {
	const signature: MatrixSignatureState = {
		matrixSignatureA: SIGNATURE_INIT_A,
		matrixSignatureB: SIGNATURE_INIT_B,
	};
	writeMatrix4Signature(signature, matrix);
	return signature;
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
