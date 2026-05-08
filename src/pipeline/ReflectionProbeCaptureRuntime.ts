import type { Scene } from "../core/Scene";
import { Texture } from "../core/Texture";
import { Logger } from "../foundation/Logger";
import {
	LightType,
	type AmbientLight,
	type AreaLight,
	type DirectionalLight,
	type PointLight,
	type ReflectionProbe,
	type SceneLight,
	type SpotLight,
} from "../lights";
import { sRGBToLinear, clamp } from "../maths/Common";
import type { IVector3 } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import type { FrameContext } from "./types";
import {
	bakeEnvironmentIBLFromEnvironmentMap,
	type BakedEnvironmentIBL,
	type EnvironmentIBLBakeOptions,
} from "./EnvironmentIBLBaker";
import {
	directionFromEquirectUV,
	sampleEnvironmentTextureLevel,
} from "./environmentMapRuntime";
import { RENDER_DIRTY_REASON_MASK } from "./incremental";
import type { WebGPUComputeFacadeSource } from "../renderers/webgpu/ComputeFacade";

const DIRECTIONAL_LOBE_EXPONENT = 96;
const LOCAL_LIGHT_LOBE_EXPONENT = 64;
const AREA_LIGHT_LOBE_EXPONENT = 48;
const DEFAULT_MAX_BAKES_PER_FRAME = 1;
const DEFAULT_CAPTURE_BUDGET_MS = 4;
const CAPTURE_RESOLUTION_SCALE_STEPS = [1, 0.75, 0.5] as const;
const CAPTURE_RESOLUTION_DOWNSCALE_OVERBUDGET_RATIO = 4;
const CAPTURE_PREFILTER_MAX_MIP_LEVELS = 9;
const MIN_LIGHT_DISTANCE = 1e-4;
const CAPTURE_RELEVANT_SCENE_DIRTY_MASK =
	RENDER_DIRTY_REASON_MASK.unknown |
	RENDER_DIRTY_REASON_MASK.transform |
	RENDER_DIRTY_REASON_MASK.material |
	RENDER_DIRTY_REASON_MASK.texture |
	RENDER_DIRTY_REASON_MASK.lighting |
	RENDER_DIRTY_REASON_MASK.shadow |
	RENDER_DIRTY_REASON_MASK.physics |
	RENDER_DIRTY_REASON_MASK.particles;

interface CaptureTaskState {
	taskId: number;
	probeId: string;
	probeSignature: string;
	captureRequestToken: number;
	sceneDirtyStamp: number;
	startedAtSeconds: number;
	scene: Scene;
	baseCaptureWidth: number;
	baseCaptureHeight: number;
	scaleIndex: number;
	captureWidth: number;
	captureHeight: number;
	faceSize: number;
	pendingFaces: number[];
	capturedFaces: Array<Float32Array | null>;
	useMeshCapture: boolean;
	lightingState: CaptureLightingState | null;
}

interface RGBLinear {
	r: number;
	g: number;
	b: number;
}

interface CapturedDirectionalLight {
	direction: IVector3;
	color: RGBLinear;
}

interface CapturedLocalLight {
	direction: IVector3;
	color: RGBLinear;
}

interface CaptureLightingState {
	environmentBackgroundTexture: Texture | null;
	includeEnvironmentBackground: boolean;
	ambient: RGBLinear;
	directionalLights: CapturedDirectionalLight[];
	pointLights: CapturedLocalLight[];
	spotLights: CapturedLocalLight[];
	areaLights: CapturedLocalLight[];
}

export interface ReflectionProbeWebGPUCaptureFaceRequest {
	frameContext: FrameContext;
	probe: ReflectionProbe;
	faceIndex: number;
	faceSize: number;
	includeEnvironment: boolean;
	includeTransparent: boolean;
	includeParticles: boolean;
	includeShadows: boolean;
}

export interface ReflectionProbeWebGPUCaptureSource {
	captureReflectionProbeFace(
		request: ReflectionProbeWebGPUCaptureFaceRequest
	): Promise<Float32Array | null>;
}

export interface ReflectionProbeCaptureRuntimeExecuteContext {
	scene: Scene;
	nowMs: number;
	frameDirtyReasonMask?: number | null;
	frameContext?: FrameContext | null;
	cameraWorldPosition?: IVector3 | null;
	webgpuSource?: WebGPUComputeFacadeSource | null;
	webgpuCaptureSource?: ReflectionProbeWebGPUCaptureSource | null;
}

export interface ReflectionProbeCaptureRuntimeOptions {
	maxBakesPerFrame?: number;
	captureBudgetMs?: number;
	bakeEnvironmentIBL?: (
		envMap: Texture,
		options: EnvironmentIBLBakeOptions
	) => Promise<BakedEnvironmentIBL>;
}

export class ReflectionProbeCaptureRuntime {
	private _activeTask: CaptureTaskState | null = null;
	private _inFlightQuantum: Promise<void> | null = null;
	private _nextTaskId = 0;
	private _maxBakesPerFrame: number;
	private _captureBudgetMs: number;
	private _bakeEnvironmentIBL: (
		envMap: Texture,
		options: EnvironmentIBLBakeOptions
	) => Promise<BakedEnvironmentIBL>;
	private _lastCaptureSecondsByProbeId = new Map<string, number>();
	private _lastCaptureSceneDirtyStampByProbeId = new Map<string, number>();
	private _lastHandledRequestTokenByProbeId = new Map<string, number>();
	private _sceneDirtyStampByScene = new WeakMap<Scene, number>();
	private _lastRelevantSceneVersionByScene = new WeakMap<Scene, number>();

	constructor(options: ReflectionProbeCaptureRuntimeOptions = {}) {
		this._maxBakesPerFrame = Math.max(
			1,
			Math.floor(options.maxBakesPerFrame ?? DEFAULT_MAX_BAKES_PER_FRAME)
		);
		this._captureBudgetMs = Math.max(
			0.1,
			Number.isFinite(options.captureBudgetMs) ?
				Number(options.captureBudgetMs)
			:	DEFAULT_CAPTURE_BUDGET_MS
		);
		this._bakeEnvironmentIBL =
			options.bakeEnvironmentIBL ?? bakeEnvironmentIBLFromEnvironmentMap;
	}

	public execute(
		context: ReflectionProbeCaptureRuntimeExecuteContext
	): Promise<void> {
		const frameDirtyReasonMask =
			context.frameDirtyReasonMask ?? context.scene.consumeDirtyReasonMask();
		const sceneDirtyStamp = this._resolveSceneDirtyStamp(
			context.scene,
			frameDirtyReasonMask
		);
		const probes = collectCapturedSceneProbes(
			context.scene.getLights(),
			context.cameraWorldPosition ?? null
		);
		this._pruneProbeState(probes);
		if (probes.length <= 0) {
			return Promise.resolve();
		}
		if (this._inFlightQuantum) {
			return this._inFlightQuantum;
		}

		if (!this._activeTask) {
			const nowSeconds = Math.max(0, context.nowMs) / 1000;
			const candidates = probes.filter((probe) =>
				this._shouldCaptureProbe(probe, sceneDirtyStamp, nowSeconds)
			);
			if (candidates.length <= 0) {
				return Promise.resolve();
			}

			let started = 0;
				for (const probe of candidates) {
					if (started >= this._maxBakesPerFrame) {
						break;
					}
					const task = this._createTask(probe, context, sceneDirtyStamp);
					if (!task) continue;
				this._activeTask = task;
				started++;
				break;
			}
			if (!this._activeTask) {
				return Promise.resolve();
			}
		}

		const quantum = this._runQuantum(context)
			.catch(() => {
				this._activeTask = null;
			})
			.finally(() => {
				if (this._inFlightQuantum === quantum) {
					this._inFlightQuantum = null;
				}
			});
		this._inFlightQuantum = quantum;
		return quantum;
	}

	private _shouldCaptureProbe(
		probe: ReflectionProbe,
		sceneDirtyStamp: number,
		nowSeconds: number
	): boolean {
		const lastHandledToken =
			this._lastHandledRequestTokenByProbeId.get(probe.id) ?? 0;
		if (probe.captureRequestToken > lastHandledToken) {
			return true;
		}

		if (probe.captureUpdateMode === "manual") {
			return false;
		}

		if (probe.captureUpdateMode === "onSceneDirty") {
			const lastCapturedSceneDirtyStamp =
				this._lastCaptureSceneDirtyStampByProbeId.get(probe.id);
			return lastCapturedSceneDirtyStamp !== sceneDirtyStamp;
		}

		const lastCaptureSeconds = this._lastCaptureSecondsByProbeId.get(probe.id);
		if (lastCaptureSeconds === undefined) {
			return true;
		}
		return nowSeconds - lastCaptureSeconds >= probe.captureIntervalSeconds;
	}

	private _createTask(
		probe: ReflectionProbe,
		context: ReflectionProbeCaptureRuntimeExecuteContext,
		sceneDirtyStamp: number
	): CaptureTaskState | null {
		const baseCaptureWidth = Math.max(
			8,
			Math.floor(probe.captureResolution.width)
		);
		const baseCaptureHeight = Math.max(
			4,
			Math.floor(probe.captureResolution.height)
		);
		const useMeshCapture =
			probe.includeMeshes &&
			!!context.frameContext &&
			!!context.webgpuCaptureSource;
		if (probe.includeMeshes && !useMeshCapture) {
			Logger.warn(
				"[reflection-probe-mesh-capture-unsupported] Reflection probe scene mesh capture requested without a compatible GPU face capture source; falling back to environment background and analytic lights only.",
				{
					scope: "ReflectionProbeCaptureRuntime",
					onceKey: "reflection-probe-mesh-capture-unsupported",
				}
			);
		}
		const task: CaptureTaskState = {
			taskId: ++this._nextTaskId,
			probeId: probe.id,
			probeSignature: buildProbeCaptureSignature(probe),
			captureRequestToken: probe.captureRequestToken,
			sceneDirtyStamp,
			startedAtSeconds: Math.max(0, context.nowMs) / 1000,
			scene: context.scene,
			baseCaptureWidth,
			baseCaptureHeight,
			scaleIndex: 0,
			captureWidth: baseCaptureWidth,
			captureHeight: baseCaptureHeight,
			faceSize: Math.max(
				4,
				Math.floor(Math.min(baseCaptureWidth / 4, baseCaptureHeight / 2))
			),
			pendingFaces: [0, 1, 2, 3, 4, 5],
			capturedFaces: [null, null, null, null, null, null],
			useMeshCapture,
			lightingState:
				useMeshCapture ? null : buildCaptureLightingState(context.scene, probe),
		};
		applyTaskScale(task, 0);
		return task;
	}

	private async _runQuantum(
		context: ReflectionProbeCaptureRuntimeExecuteContext
	): Promise<void> {
		const task = this._activeTask;
		if (!task) {
			return;
		}

		const probe = findCapturedSceneProbe(task.scene, task.probeId);
		if (!probe || !this._isTaskFresh(task, probe)) {
			this._activeTask = null;
			return;
		}

		const quantumStart = resolveNowMs();
		while (task.pendingFaces.length > 0) {
			const faceIndex = task.pendingFaces.shift()!;
			const faceStart = resolveNowMs();
			const faceData = await this._captureFace(
				task,
				probe,
				faceIndex,
				context
			);
			if (!faceData) {
				this._activeTask = null;
				return;
			}
			task.capturedFaces[faceIndex] = faceData;

			const faceDurationMs = resolveNowMs() - faceStart;
			if (
				faceDurationMs >
					this._captureBudgetMs *
						CAPTURE_RESOLUTION_DOWNSCALE_OVERBUDGET_RATIO &&
				this._downgradeTaskResolution(task)
			) {
				return;
			}

			if (resolveNowMs() - quantumStart >= this._captureBudgetMs) {
				break;
			}
		}

		if (task.pendingFaces.length > 0) {
			return;
		}

		if (!this._isTaskFresh(task, probe)) {
			this._activeTask = null;
			return;
		}

		const environmentMap = buildCapturedEnvironmentMap(task);
		if (!environmentMap) {
			this._activeTask = null;
			return;
		}

		await this._runCaptureBake(task, environmentMap, context.webgpuSource ?? null);
		if (this._activeTask?.taskId === task.taskId) {
			this._activeTask = null;
		}
	}

	private _downgradeTaskResolution(task: CaptureTaskState): boolean {
		if (task.scaleIndex + 1 >= CAPTURE_RESOLUTION_SCALE_STEPS.length) {
			return false;
		}
		applyTaskScale(task, task.scaleIndex + 1);
		return true;
	}

	private async _captureFace(
		task: CaptureTaskState,
		probe: ReflectionProbe,
		faceIndex: number,
		context: ReflectionProbeCaptureRuntimeExecuteContext
	): Promise<Float32Array | null> {
		if (
			task.useMeshCapture &&
			context.frameContext &&
			context.webgpuCaptureSource &&
			probe.includeMeshes
		) {
			try {
				const captured =
					await context.webgpuCaptureSource.captureReflectionProbeFace({
						frameContext: context.frameContext,
						probe,
						faceIndex,
						faceSize: task.faceSize,
						includeEnvironment: probe.includeEnvironment,
						includeTransparent: probe.includeTransparent,
						includeParticles: probe.includeParticles,
						includeShadows: probe.includeShadows,
					});
				if (captured && captured.length >= task.faceSize * task.faceSize * 4) {
					return captured;
				}
			} catch {
				// fall through to CPU fallback
			}
		}

		if (!task.lightingState) {
			task.lightingState = buildCaptureLightingState(task.scene, probe);
		}
		return captureCubeFace(task.faceSize, faceIndex, task.lightingState);
	}

	private async _runCaptureBake(
		task: CaptureTaskState,
		environmentMap: Texture,
		webgpuSource: WebGPUComputeFacadeSource | null
	): Promise<void> {
		const prefilterMaxSampleWidth = Math.max(1, Math.floor(task.captureWidth));
		const prefilterMaxSampleHeight = Math.max(1, Math.floor(task.captureHeight));
		const bakeOptions: EnvironmentIBLBakeOptions = {
			acceleration: "auto",
			prefilterMaxSampleWidth,
			prefilterMaxSampleHeight,
			prefilterMaxMipLevels: resolveCapturePrefilterMipLevels(
				prefilterMaxSampleWidth,
				prefilterMaxSampleHeight
			),
		};
		if (webgpuSource) {
			bakeOptions.webgpuSource = webgpuSource;
		}

		const baked = await this._bakeEnvironmentIBL(environmentMap, bakeOptions);
		const probe = findCapturedSceneProbe(task.scene, task.probeId);
		if (!probe || !this._isTaskFresh(task, probe)) {
			return;
		}

		probe.prefilteredMap = baked.prefilteredMap;
		probe.markCaptureUpdated();
		probe.markRuntimeDirty();
		task.scene.invalidate("reflection-probe");

		this._lastCaptureSecondsByProbeId.set(task.probeId, task.startedAtSeconds);
		this._lastCaptureSceneDirtyStampByProbeId.set(
			task.probeId,
			task.sceneDirtyStamp
		);
		this._lastHandledRequestTokenByProbeId.set(
			task.probeId,
			task.captureRequestToken
		);
	}

	private _isTaskFresh(task: CaptureTaskState, probe: ReflectionProbe): boolean {
		if (this._activeTask?.taskId !== task.taskId) {
			return false;
		}
		if (probe.source !== "capturedScene") {
			return false;
		}
		if (probe.captureRequestToken !== task.captureRequestToken) {
			return false;
		}
		if (this._getSceneDirtyStamp(task.scene) !== task.sceneDirtyStamp) {
			return false;
		}
		return buildProbeCaptureSignature(probe) === task.probeSignature;
	}

	private _pruneProbeState(probes: ReflectionProbe[]): void {
		const activeProbeIds = new Set(probes.map((probe) => probe.id));
		pruneProbeMap(this._lastCaptureSecondsByProbeId, activeProbeIds);
		pruneProbeMap(this._lastCaptureSceneDirtyStampByProbeId, activeProbeIds);
		pruneProbeMap(this._lastHandledRequestTokenByProbeId, activeProbeIds);
		if (
			this._activeTask &&
			activeProbeIds.has(this._activeTask.probeId) === false
		) {
			this._activeTask = null;
		}
	}

	private _resolveSceneDirtyStamp(scene: Scene, dirtyReasonMask: number): number {
		const currentStamp = this._getSceneDirtyStamp(scene);
		const relevantDirtyReasonMask =
			(dirtyReasonMask >>> 0) & CAPTURE_RELEVANT_SCENE_DIRTY_MASK;
		if (relevantDirtyReasonMask === 0) {
			return currentStamp;
		}
		const lastRelevantSceneVersion =
			this._lastRelevantSceneVersionByScene.get(scene);
		if (lastRelevantSceneVersion === scene.version) {
			return currentStamp;
		}
		const nextStamp = currentStamp + 1;
		this._sceneDirtyStampByScene.set(scene, nextStamp);
		this._lastRelevantSceneVersionByScene.set(scene, scene.version);
		return nextStamp;
	}

	private _getSceneDirtyStamp(scene: Scene): number {
		return this._sceneDirtyStampByScene.get(scene) ?? 0;
	}
}

function collectCapturedSceneProbes(
	lights: SceneLight[],
	cameraWorldPosition: IVector3 | null
): ReflectionProbe[] {
	const probes: ReflectionProbe[] = [];
	for (const light of lights) {
		if (light.type !== LightType.ReflectionProbe) continue;
		const probe = light as ReflectionProbe;
		if (probe.source !== "capturedScene") continue;
		probes.push(probe);
	}
	probes.sort((left, right) => {
		if (cameraWorldPosition) {
			const leftDistance = squaredDistanceToProbe(
				cameraWorldPosition,
				left.getWorldPosition({ x: 0, y: 0, z: 0 })
			);
			const rightDistance = squaredDistanceToProbe(
				cameraWorldPosition,
				right.getWorldPosition({ x: 0, y: 0, z: 0 })
			);
			if (leftDistance !== rightDistance) {
				return leftDistance - rightDistance;
			}
		}
		return left.id.localeCompare(right.id);
	});
	return probes;
}

function findCapturedSceneProbe(
	scene: Scene,
	probeId: string
): ReflectionProbe | null {
	for (const light of scene.getLights()) {
		if (light.type !== LightType.ReflectionProbe) continue;
		const probe = light as ReflectionProbe;
		if (probe.source !== "capturedScene") continue;
		if (probe.id === probeId) {
			return probe;
		}
	}
	return null;
}

function pruneProbeMap<TValue>(
	map: Map<string, TValue>,
	activeProbeIds: Set<string>
): void {
	for (const probeId of map.keys()) {
		if (activeProbeIds.has(probeId)) continue;
		map.delete(probeId);
	}
}

function buildProbeCaptureSignature(probe: ReflectionProbe): string {
	const elements = probe.worldMatrix.elements;
	const matrix = new Array<string>(16);
	let cursor = 0;
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			matrix[cursor++] = elements[row][col].toFixed(6);
		}
	}
	return [
		probe.source,
		probe.captureUpdateMode,
		probe.captureResolution.width,
		probe.captureResolution.height,
		probe.captureFar.toFixed(6),
		probe.includeEnvironment ? 1 : 0,
		probe.includeMeshes ? 1 : 0,
		probe.includeTransparent ? 1 : 0,
		probe.includeParticles ? 1 : 0,
		probe.includeShadows ? 1 : 0,
		probe.captureRequestToken,
		...matrix,
	].join("|");
}

function applyTaskScale(task: CaptureTaskState, scaleIndex: number): void {
	const scale =
		CAPTURE_RESOLUTION_SCALE_STEPS[
			Math.max(0, Math.min(CAPTURE_RESOLUTION_SCALE_STEPS.length - 1, scaleIndex))
		];
	const captureWidth = Math.max(
		8,
		Math.floor(task.baseCaptureWidth * scale)
	);
	const captureHeight = Math.max(
		4,
		Math.floor(task.baseCaptureHeight * scale)
	);
	const faceSize = Math.max(
		4,
		Math.floor(Math.min(captureWidth / 4, captureHeight / 2))
	);
	task.scaleIndex = scaleIndex;
	task.captureWidth = captureWidth;
	task.captureHeight = captureHeight;
	task.faceSize = faceSize;
	task.pendingFaces = [0, 1, 2, 3, 4, 5];
	task.capturedFaces = [null, null, null, null, null, null];
}

function buildCapturedEnvironmentMap(task: CaptureTaskState): Texture | null {
	const cubeFaces: Float32Array[] = [];
	for (let faceIndex = 0; faceIndex < 6; faceIndex++) {
		const faceData = task.capturedFaces[faceIndex];
		if (!faceData) {
			return null;
		}
		cubeFaces.push(faceData);
	}
	const equirectData = convertCubeFacesToEquirect(
		cubeFaces,
		task.faceSize,
		task.captureWidth,
		task.captureHeight
	);
	const texture = new Texture(
		equirectData,
		task.captureWidth,
		task.captureHeight,
		"HDR"
	);
	texture.wrapS = "Repeat";
	texture.wrapT = "Clamp";
	texture.minFilter = "Linear";
	texture.magFilter = "Linear";
	return texture;
}

function squaredDistanceToProbe(from: IVector3, to: IVector3): number {
	const dx = to.x - from.x;
	const dy = to.y - from.y;
	const dz = to.z - from.z;
	return dx * dx + dy * dy + dz * dz;
}

function resolveNowMs(): number {
	if (typeof performance !== "undefined" && typeof performance.now === "function") {
		return performance.now();
	}
	return Date.now();
}

function buildCaptureLightingState(
	scene: Scene,
	probe: ReflectionProbe
): CaptureLightingState {
	const lights = scene.getLights();
	const capturePosition = probe.getRuntimeCache().captureWorldPosition;
	const captureRange = Math.max(1, probe.captureFar);
	const ambient: RGBLinear = { r: 0, g: 0, b: 0 };
	const directionalLights: CapturedDirectionalLight[] = [];
	const pointLights: CapturedLocalLight[] = [];
	const spotLights: CapturedLocalLight[] = [];
	const areaLights: CapturedLocalLight[] = [];

	for (const light of lights) {
		switch (light.type) {
			case LightType.Ambient: {
				const ambientLight = light as AmbientLight;
				accumulateAmbient(ambient, ambientLight);
				break;
			}
			case LightType.Directional: {
				const directional = light as DirectionalLight;
				const worldDirection = directional.getWorldLightDirection();
				const incoming = normalizeDirection({
					x: -worldDirection.x,
					y: -worldDirection.y,
					z: -worldDirection.z,
				});
				directionalLights.push({
					direction: incoming,
					color: toLinearColor(directional.color, directional.intensity ?? 1),
				});
				break;
			}
			case LightType.Point: {
				const point = light as PointLight;
				const worldPosition = point.getWorldLightPosition();
				const captured = buildCapturedLocalLight(
					capturePosition,
					worldPosition,
					point.range,
					captureRange,
					toLinearColor(point.color, point.intensity ?? 1)
				);
				if (captured) {
					pointLights.push(captured);
				}
				break;
			}
			case LightType.Spot: {
				const spot = light as SpotLight;
				const worldPosition = spot.getWorldLightPosition();
				const directionToProbe = normalizeDirection({
					x: capturePosition.x - worldPosition.x,
					y: capturePosition.y - worldPosition.y,
					z: capturePosition.z - worldPosition.z,
				});
				const worldDirection = spot.getWorldLightDirection();
				const coneWeight = computeSpotConeWeight(
					worldDirection,
					directionToProbe,
					spot.outerAngle,
					spot.getInnerAngle()
				);
				if (coneWeight <= 0) {
					break;
				}
				const color = toLinearColor(spot.color, (spot.intensity ?? 1) * coneWeight);
				const captured = buildCapturedLocalLight(
					capturePosition,
					worldPosition,
					spot.range,
					captureRange,
					color
				);
				if (captured) {
					spotLights.push(captured);
				}
				break;
			}
			case LightType.RectArea: {
				const area = light as AreaLight;
				const worldPosition = area.getWorldPosition({ x: 0, y: 0, z: 0 });
				const areaScale = Math.max(1, Math.sqrt(area.width * area.height)) * 0.01;
				const color = toLinearColor(
					area.color,
					(area.intensity ?? 1) * areaScale
				);
				const captured = buildCapturedLocalLight(
					capturePosition,
					worldPosition,
					area.range,
					captureRange,
					color
				);
				if (captured) {
					areaLights.push(captured);
				}
				break;
			}
			default:
				break;
		}
	}

	return {
		environmentBackgroundTexture:
			probe.includeEnvironment && scene.environment.backgroundEnabled ?
				scene.environment.backgroundTexture
			:	null,
		includeEnvironmentBackground:
			probe.includeEnvironment && scene.environment.backgroundEnabled,
		ambient,
		directionalLights,
		pointLights,
		spotLights,
		areaLights,
	};
}

function accumulateAmbient(
	target: RGBLinear,
	light: AmbientLight
): void {
	const linear = toLinearColor(light.color, light.intensity ?? 1);
	target.r += linear.r;
	target.g += linear.g;
	target.b += linear.b;
}

function buildCapturedLocalLight(
	capturePosition: IVector3,
	lightPosition: IVector3,
	lightRange: number,
	captureFar: number,
	color: RGBLinear
): CapturedLocalLight | null {
	const toLight = {
		x: lightPosition.x - capturePosition.x,
		y: lightPosition.y - capturePosition.y,
		z: lightPosition.z - capturePosition.z,
	};
	const distance = Math.hypot(toLight.x, toLight.y, toLight.z);
	const range = Math.max(
		MIN_LIGHT_DISTANCE,
		Math.min(
			Number.isFinite(lightRange) ? Math.max(lightRange, MIN_LIGHT_DISTANCE) : Infinity,
			captureFar
		)
	);
	if (distance > range) {
		return null;
	}
	const attenuation = computeDistanceAttenuation(distance, range);
	const direction = normalizeDirection(toLight);
	return {
		direction,
		color: {
			r: color.r * attenuation,
			g: color.g * attenuation,
			b: color.b * attenuation,
		},
	};
}

function computeSpotConeWeight(
	lightDirection: IVector3,
	directionToProbe: IVector3,
	outerAngle: number,
	innerAngle: number
): number {
	const cosTheta = Vector3.dot(lightDirection, directionToProbe);
	const outerCos = Math.cos(Math.max(outerAngle, 0));
	const innerCos = Math.cos(Math.max(0, Math.min(innerAngle, outerAngle)));
	if (cosTheta <= outerCos) {
		return 0;
	}
	if (innerCos <= outerCos) {
		return 1;
	}
	return smoothstep(outerCos, innerCos, cosTheta);
}

function captureCubeFace(
	faceSize: number,
	faceIndex: number,
	lightingState: CaptureLightingState
): Float32Array {
	const data = new Float32Array(faceSize * faceSize * 4);
	for (let y = 0; y < faceSize; y++) {
		for (let x = 0; x < faceSize; x++) {
			const direction = directionFromCubeFace(faceIndex, x, y, faceSize);
			const radiance = sampleCapturedRadiance(direction, lightingState);
			const index = (y * faceSize + x) * 4;
			data[index] = radiance.r;
			data[index + 1] = radiance.g;
			data[index + 2] = radiance.b;
			data[index + 3] = 1;
		}
	}
	return data;
}

function convertCubeFacesToEquirect(
	cubeFaces: Float32Array[],
	faceSize: number,
	width: number,
	height: number
): Float32Array {
	const data = new Float32Array(width * height * 4);
	for (let y = 0; y < height; y++) {
		const v = (y + 0.5) / height;
		for (let x = 0; x < width; x++) {
			const u = (x + 0.5) / width;
			const direction = directionFromEquirectUV(u, v);
			const sample = sampleCubeFaces(cubeFaces, faceSize, direction);
			const index = (y * width + x) * 4;
			data[index] = sample.r;
			data[index + 1] = sample.g;
			data[index + 2] = sample.b;
			data[index + 3] = 1;
		}
	}
	return data;
}

function sampleCapturedRadiance(
	direction: IVector3,
	lightingState: CaptureLightingState
): RGBLinear {
	const result: RGBLinear = {
		r: lightingState.ambient.r,
		g: lightingState.ambient.g,
		b: lightingState.ambient.b,
	};

	if (
		lightingState.includeEnvironmentBackground &&
		lightingState.environmentBackgroundTexture
	) {
		const sky = sampleEnvironmentBackgroundLinear(
			lightingState.environmentBackgroundTexture,
			direction
		);
		result.r += sky.r;
		result.g += sky.g;
		result.b += sky.b;
	}

	for (const directional of lightingState.directionalLights) {
		const lobe = Math.pow(
			Math.max(0, Vector3.dot(direction, directional.direction)),
			DIRECTIONAL_LOBE_EXPONENT,
		);
		result.r += directional.color.r * lobe;
		result.g += directional.color.g * lobe;
		result.b += directional.color.b * lobe;
	}

	for (const point of lightingState.pointLights) {
		const lobe = Math.pow(
			Math.max(0, Vector3.dot(direction, point.direction)),
			LOCAL_LIGHT_LOBE_EXPONENT,
		);
		result.r += point.color.r * lobe;
		result.g += point.color.g * lobe;
		result.b += point.color.b * lobe;
	}

	for (const spot of lightingState.spotLights) {
		const lobe = Math.pow(
			Math.max(0, Vector3.dot(direction, spot.direction)),
			LOCAL_LIGHT_LOBE_EXPONENT,
		);
		result.r += spot.color.r * lobe;
		result.g += spot.color.g * lobe;
		result.b += spot.color.b * lobe;
	}

	for (const area of lightingState.areaLights) {
		const lobe = Math.pow(
			Math.max(0, Vector3.dot(direction, area.direction)),
			AREA_LIGHT_LOBE_EXPONENT,
		);
		result.r += area.color.r * lobe;
		result.g += area.color.g * lobe;
		result.b += area.color.b * lobe;
	}

	return result;
}

function directionFromCubeFace(
	face: number,
	x: number,
	y: number,
	faceSize: number
): IVector3 {
	const u = (2 * (x + 0.5)) / faceSize - 1;
	const v = 1 - (2 * (y + 0.5)) / faceSize;
	switch (face) {
		case 0:
			return normalizeDirection({ x: 1, y: v, z: -u });
		case 1:
			return normalizeDirection({ x: -1, y: v, z: u });
		case 2:
			return normalizeDirection({ x: u, y: 1, z: -v });
		case 3:
			return normalizeDirection({ x: u, y: -1, z: v });
		case 4:
			return normalizeDirection({ x: u, y: v, z: 1 });
		default:
			return normalizeDirection({ x: -u, y: v, z: -1 });
	}
}

function sampleCubeFaces(
	cubeFaces: Float32Array[],
	faceSize: number,
	direction: IVector3
): RGBLinear {
	const { face, uc, vc } = directionToCubeFaceUV(direction);
	const x = Math.max(0, Math.min(faceSize - 1, Math.floor((uc + 1) * 0.5 * faceSize)));
	const y = Math.max(0, Math.min(faceSize - 1, Math.floor((1 - (vc + 1) * 0.5) * faceSize)));
	const data = cubeFaces[face];
	const index = (y * faceSize + x) * 4;
	return {
		r: data[index],
		g: data[index + 1],
		b: data[index + 2],
	};
}

function directionToCubeFaceUV(direction: IVector3): {
	face: number;
	uc: number;
	vc: number;
} {
	const ax = Math.abs(direction.x);
	const ay = Math.abs(direction.y);
	const az = Math.abs(direction.z);
	if (ax >= ay && ax >= az) {
		if (direction.x > 0) {
			return {
				face: 0,
				uc: -direction.z / ax,
				vc: direction.y / ax,
			};
		}
		return {
			face: 1,
			uc: direction.z / ax,
			vc: direction.y / ax,
		};
	}
	if (ay >= ax && ay >= az) {
		if (direction.y > 0) {
			return {
				face: 2,
				uc: direction.x / ay,
				vc: -direction.z / ay,
			};
		}
		return {
			face: 3,
			uc: direction.x / ay,
			vc: direction.z / ay,
		};
	}
	if (direction.z > 0) {
		return {
			face: 4,
			uc: direction.x / az,
			vc: direction.y / az,
		};
	}
	return {
		face: 5,
		uc: -direction.x / az,
		vc: direction.y / az,
	};
}

function sampleEnvironmentBackgroundLinear(
	environmentBackgroundTexture: Texture,
	direction: IVector3
): RGBLinear {
	const sample = sampleEnvironmentTextureLevel(
		environmentBackgroundTexture,
		direction,
		0
	);
	if (environmentBackgroundTexture.colorSpace === "sRGB") {
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

function toLinearColor(
	color: { r: number; g: number; b: number },
	intensity: number
): RGBLinear {
	const safeIntensity = Number.isFinite(intensity) ? Math.max(0, intensity) : 0;
	return {
		r: sRGBToLinear(clamp(color.r / 255, 0, 1)) * safeIntensity,
		g: sRGBToLinear(clamp(color.g / 255, 0, 1)) * safeIntensity,
		b: sRGBToLinear(clamp(color.b / 255, 0, 1)) * safeIntensity,
	};
}

function computeDistanceAttenuation(distance: number, range: number): number {
	if (distance <= MIN_LIGHT_DISTANCE) return 1;
	const normalized = clamp(distance / Math.max(range, MIN_LIGHT_DISTANCE), 0, 1);
	const inverseSquare = 1 / Math.max(distance * distance, MIN_LIGHT_DISTANCE);
	const rangeFade = 1 - normalized * normalized;
	return inverseSquare * rangeFade * rangeFade * 128;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
	if (edge1 <= edge0) {
		return x >= edge1 ? 1 : 0;
	}
	const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
	return t * t * (3 - 2 * t);
}


function normalizeDirection(direction: IVector3): IVector3 {
	const length = Math.hypot(direction.x, direction.y, direction.z);
	if (length <= MIN_LIGHT_DISTANCE) {
		return { x: 0, y: 0, z: 1 };
	}
	const inv = 1 / length;
	return {
		x: direction.x * inv,
		y: direction.y * inv,
		z: direction.z * inv,
	};
}

function resolveCapturePrefilterMipLevels(width: number, height: number): number {
	const longestSide = Math.max(1, Math.floor(Math.max(width, height)));
	const fullMipLevels = Math.floor(Math.log2(longestSide)) + 1;
	return Math.max(
		1,
		Math.min(CAPTURE_PREFILTER_MAX_MIP_LEVELS, fullMipLevels)
	);
}
