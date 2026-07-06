import type { Scene } from "../../core/Scene";
import type { CubeTextureFaceData } from "../../core/CubeTexture";
import { Texture } from "../../core/Texture";
import type { TextureMipLevel } from "../../core/Texture";
import { Logger } from "../../foundation/Logger";
import {
	LightType,
	type AmbientLight,
	type AreaLight,
	type DirectionalLight,
	type IrradianceProbeGrid,
	type LightProbe,
	type PointLight,
	type ReflectionProbe,
	type SceneLight,
	type SpotLight,
} from "../../lights";
import { sRGBToLinear, clamp } from "../../maths/Common";
import type { IVector3, SHCoefficients } from "../../maths/types";
import { Vector3 } from "../../maths/Vector3";
import type { FrameContext } from "../../pipeline/types";
import { projectEnvironmentTextureToSH } from "../ibl/EnvironmentSH";
import {
	prefilterEnvironmentIBL,
	type IBLPrefilterOptions,
} from "../ibl/IBLPrefilter";
import {
	directionFromEquirectUV,
	sampleEnvironmentTextureLevel,
} from "./environmentMapRuntime";
import { RENDER_DIRTY_REASON_MASK } from "../../pipeline/incremental";

const DIRECTIONAL_LOBE_EXPONENT = 96;
const LOCAL_LIGHT_LOBE_EXPONENT = 64;
const AREA_LIGHT_LOBE_EXPONENT = 48;
const DEFAULT_MAX_CAPTURES_PER_FRAME = 1;
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

export type ProbeCaptureTargetKind = "reflection" | "light" | "grid";

interface ProbeCaptureTarget {
	kind: ProbeCaptureTargetKind;
	id: string;
	probe: ReflectionProbe | LightProbe | IrradianceProbeGrid;
	cellIndex: number;
	signature: string;
	groupKey: string;
	captureRequestToken: number;
	captureUpdateMode: "manual" | "onSceneDirty" | "interval";
	captureIntervalSeconds: number;
	captureWidth: number;
	captureHeight: number;
	captureFar: number;
	captureWorldPosition: IVector3;
	includeEnvironment: boolean;
	includeMeshes: boolean;
	includeTransparent: boolean;
	includeParticles: boolean;
	includeShadows: boolean;
}

interface CaptureTaskTarget {
	kind: ProbeCaptureTargetKind;
	id: string;
	cellIndex: number;
	signature: string;
	captureRequestToken: number;
}

interface CaptureTaskState {
	taskId: number;
	groupKey: string;
	targets: CaptureTaskTarget[];
	sceneDirtyStamp: number;
	startedAtSeconds: number;
	scene: Scene;
	captureWorldPosition: IVector3;
	captureFar: number;
	includeEnvironment: boolean;
	includeMeshes: boolean;
	includeTransparent: boolean;
	includeParticles: boolean;
	includeShadows: boolean;
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

export interface ProbeWebGPUCaptureFaceRequest {
	frameContext: FrameContext;
	targetId: string;
	targetKind: ProbeCaptureTargetKind;
	captureWorldPosition: IVector3;
	captureFar: number;
	faceIndex: number;
	faceSize: number;
	includeEnvironment: boolean;
	includeMeshes: boolean;
	includeTransparent: boolean;
	includeParticles: boolean;
	includeShadows: boolean;
}

export interface ProbeWebGPUCaptureSource {
	captureProbeFace(
		request: ProbeWebGPUCaptureFaceRequest
	): Promise<Float32Array | null>;
}

export interface ProbeCaptureRuntimeExecuteContext {
	scene: Scene;
	nowMs: number;
	frameDirtyReasonMask?: number | null;
	frameContext?: FrameContext | null;
	cameraWorldPosition?: IVector3 | null;
	webgpuCaptureSource?: ProbeWebGPUCaptureSource | null;
}

export interface ProbeCaptureRuntimeOptions {
	maxCapturesPerFrame?: number;
	captureBudgetMs?: number;
}

export class ProbeCaptureRuntime {
	private _activeTask: CaptureTaskState | null = null;
	private _inFlightQuantum: Promise<void> | null = null;
	private _nextTaskId = 0;
	private _maxCapturesPerFrame: number;
	private _captureBudgetMs: number;
	private _lastCaptureSecondsByProbeId = new Map<string, number>();
	private _lastCaptureSceneDirtyStampByProbeId = new Map<string, number>();
	private _lastHandledRequestTokenByProbeId = new Map<string, number>();
	private _sceneDirtyStampByScene = new WeakMap<Scene, number>();
	private _lastRelevantSceneVersionByScene = new WeakMap<Scene, number>();

	constructor(options: ProbeCaptureRuntimeOptions = {}) {
		this._maxCapturesPerFrame = Math.max(
			1,
			Math.floor(
				options.maxCapturesPerFrame ?? DEFAULT_MAX_CAPTURES_PER_FRAME
			)
		);
		this._captureBudgetMs = Math.max(
			0.1,
			Number.isFinite(options.captureBudgetMs) ?
				Number(options.captureBudgetMs)
			:	DEFAULT_CAPTURE_BUDGET_MS
		);
	}

	public execute(
		context: ProbeCaptureRuntimeExecuteContext
	): Promise<void> {
		const frameDirtyReasonMask =
			context.frameDirtyReasonMask ?? context.scene.consumeDirtyReasonMask();
		const sceneDirtyStamp = this._resolveSceneDirtyStamp(
			context.scene,
			frameDirtyReasonMask
		);
		const targets = collectCapturedSceneProbeTargets(
			context.scene.getLights(),
			context.cameraWorldPosition ?? null
		);
		this._pruneProbeState(targets);
		if (targets.length <= 0) {
			return Promise.resolve();
		}
		if (this._inFlightQuantum) {
			return this._inFlightQuantum;
		}

		if (!this._activeTask) {
			const nowSeconds = Math.max(0, context.nowMs) / 1000;
			const candidates = targets.filter((target) =>
				this._shouldCaptureTarget(target, sceneDirtyStamp, nowSeconds)
			);
			if (candidates.length <= 0) {
				return Promise.resolve();
			}

			let inspected = 0;
			for (const target of candidates) {
				if (inspected >= this._maxCapturesPerFrame) {
					break;
				}
				inspected++;
				const task = this._createTask(
					target,
					candidates,
					context,
					sceneDirtyStamp
				);
				if (!task) continue;
				this._activeTask = task;
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

	private _shouldCaptureTarget(
		target: ProbeCaptureTarget,
		sceneDirtyStamp: number,
		nowSeconds: number
	): boolean {
		const key = createProbeCaptureTargetKey(
			target.kind,
			target.id,
			target.cellIndex
		);
		const lastHandledToken =
			this._lastHandledRequestTokenByProbeId.get(key) ?? 0;
		if (target.captureRequestToken > lastHandledToken) {
			return true;
		}

		if (target.captureUpdateMode === "manual") {
			return false;
		}

		if (target.captureUpdateMode === "onSceneDirty") {
			const lastCapturedSceneDirtyStamp =
				this._lastCaptureSceneDirtyStampByProbeId.get(key);
			return lastCapturedSceneDirtyStamp !== sceneDirtyStamp;
		}

		const lastCaptureSeconds = this._lastCaptureSecondsByProbeId.get(key);
		if (lastCaptureSeconds === undefined) {
			return true;
		}
		return nowSeconds - lastCaptureSeconds >= target.captureIntervalSeconds;
	}

	private _createTask(
		primaryTarget: ProbeCaptureTarget,
		candidates: ProbeCaptureTarget[],
		context: ProbeCaptureRuntimeExecuteContext,
		sceneDirtyStamp: number
	): CaptureTaskState | null {
		const groupedTargets = candidates.filter(
			(target) => target.groupKey === primaryTarget.groupKey
		);
		if (groupedTargets.length <= 0) {
			return null;
		}
		const baseCaptureWidth = Math.max(
			8,
			Math.floor(
				Math.max(...groupedTargets.map((target) => target.captureWidth))
			)
		);
		const baseCaptureHeight = Math.max(
			4,
			Math.floor(
				Math.max(...groupedTargets.map((target) => target.captureHeight))
			)
		);
		const useMeshCapture =
			primaryTarget.includeMeshes &&
			!!context.frameContext &&
			!!context.webgpuCaptureSource;
		if (primaryTarget.includeMeshes && !useMeshCapture) {
			Logger.warn(
				"[probe-mesh-capture-unsupported] Probe scene mesh capture requested without a compatible GPU face capture source; falling back to environment background and analytic lights only.",
				{
					scope: "ProbeCaptureRuntime",
					onceKey: "probe-mesh-capture-unsupported",
				}
			);
		}
		const task: CaptureTaskState = {
			taskId: ++this._nextTaskId,
			groupKey: primaryTarget.groupKey,
			targets: groupedTargets.map((target) => ({
				kind: target.kind,
				id: target.id,
				cellIndex: target.cellIndex,
				signature: target.signature,
				captureRequestToken: target.captureRequestToken,
			})),
			sceneDirtyStamp,
			startedAtSeconds: Math.max(0, context.nowMs) / 1000,
			scene: context.scene,
			captureWorldPosition: {
				x: primaryTarget.captureWorldPosition.x,
				y: primaryTarget.captureWorldPosition.y,
				z: primaryTarget.captureWorldPosition.z,
			},
			captureFar: primaryTarget.captureFar,
			includeEnvironment: primaryTarget.includeEnvironment,
			includeMeshes: primaryTarget.includeMeshes,
			includeTransparent: primaryTarget.includeTransparent,
			includeParticles: primaryTarget.includeParticles,
			includeShadows: primaryTarget.includeShadows,
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
			lightingState: null,
		};
		if (!useMeshCapture) {
			task.lightingState = buildCaptureLightingState(context.scene, task);
		}
		applyTaskScale(task, 0);
		return task;
	}

	private async _runQuantum(
		context: ProbeCaptureRuntimeExecuteContext
	): Promise<void> {
		const task = this._activeTask;
		if (!task) {
			return;
		}

		if (!this._hasFreshTargets(task)) {
			this._activeTask = null;
			return;
		}

		const quantumStart = resolveNowMs();
		while (task.pendingFaces.length > 0) {
			const faceIndex = task.pendingFaces.shift()!;
			const faceStart = resolveNowMs();
			const faceData = await this._captureFace(
				task,
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

		if (!this._hasFreshTargets(task)) {
			this._activeTask = null;
			return;
		}

		const environmentMap = buildCapturedEnvironmentMap(task);
		if (!environmentMap) {
			this._activeTask = null;
			return;
		}

		await this._runCaptureResolve(task, environmentMap);
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
		faceIndex: number,
		context: ProbeCaptureRuntimeExecuteContext
	): Promise<Float32Array | null> {
		if (
			task.useMeshCapture &&
			context.frameContext &&
			context.webgpuCaptureSource &&
			task.includeMeshes
		) {
			try {
				const captured =
					await context.webgpuCaptureSource.captureProbeFace({
						frameContext: context.frameContext,
						targetId: task.targets[0]?.id ?? "unknown",
						targetKind: task.targets[0]?.kind ?? "reflection",
						captureWorldPosition: task.captureWorldPosition,
						captureFar: task.captureFar,
						faceIndex,
						faceSize: task.faceSize,
						includeEnvironment: task.includeEnvironment,
						includeMeshes: task.includeMeshes,
						includeTransparent: task.includeTransparent,
						includeParticles: task.includeParticles,
						includeShadows: task.includeShadows,
					});
				if (captured && captured.length >= task.faceSize * task.faceSize * 4) {
					return captured;
				}
			} catch {
				// fall through to CPU fallback
			}
		}

		if (!task.lightingState) {
			task.lightingState = buildCaptureLightingState(task.scene, task);
		}
		return captureCubeFace(task.faceSize, faceIndex, task.lightingState);
	}

	private async _runCaptureResolve(
		task: CaptureTaskState,
		environmentMap: Texture
	): Promise<void> {
		const needsPrefilter = task.targets.some(
			(target) => target.kind === "reflection"
		);
		let sh: SHCoefficients;
		let prefilteredMap: Texture | null = null;

		const maxSampleWidth = Math.max(1, Math.floor(task.captureWidth));
		const maxSampleHeight = Math.max(1, Math.floor(task.captureHeight));
		sh = projectEnvironmentTextureToSH(environmentMap, {
			maxSampleWidth,
			maxSampleHeight,
		});
		if (needsPrefilter) {
			const prefilterOptions: IBLPrefilterOptions = {
				acceleration: "auto",
				maxSampleWidth,
				maxSampleHeight,
				maxMipLevels: resolveCapturePrefilterMipLevels(
					maxSampleWidth,
					maxSampleHeight
				),
			};
			prefilteredMap = await prefilterEnvironmentIBL(
				environmentMap,
				prefilterOptions
			);
		}

		for (const target of task.targets) {
			const probe = findCapturedSceneTarget(
				task.scene,
				target.kind,
				target.id
			);
			if (!probe || !this._isTargetFresh(task, target, probe)) {
				continue;
			}
			writeProbeCaptureTextureOutputs(probe, task, environmentMap);
			if (target.kind === "reflection") {
				if (!prefilteredMap) continue;
				const reflectionProbe = probe as ReflectionProbe;
				const boundPrefiltered =
					reflectionProbe.capture.prefilteredTexture;
				if (boundPrefiltered) {
					copyTexturePayload(boundPrefiltered, prefilteredMap);
					reflectionProbe.prefilteredMap = boundPrefiltered;
				} else {
					reflectionProbe.prefilteredMap = prefilteredMap;
				}
				reflectionProbe.markCaptureUpdated();
				reflectionProbe.markRuntimeDirty();
				task.scene.invalidate("reflection-probe");
			} else {
				if (target.kind === "grid") {
					const grid = probe as IrradianceProbeGrid;
					grid.writeCapturedCellSH(target.cellIndex, sh);
				} else {
					const lightProbe = probe as LightProbe;
					copySHCoefficients(lightProbe.sh, sh);
					lightProbe.markCaptureUpdated();
					task.scene.invalidate("probe-capture");
				}
			}

			const key = createProbeCaptureTargetKey(
				target.kind,
				target.id,
				target.cellIndex
			);
			this._lastCaptureSecondsByProbeId.set(key, task.startedAtSeconds);
			this._lastCaptureSceneDirtyStampByProbeId.set(
				key,
				task.sceneDirtyStamp
			);
			this._lastHandledRequestTokenByProbeId.set(
				key,
				target.captureRequestToken
			);
		}
	}

	private _hasFreshTargets(task: CaptureTaskState): boolean {
		for (const target of task.targets) {
			const probe = findCapturedSceneTarget(task.scene, target.kind, target.id);
			if (probe && this._isTargetFresh(task, target, probe)) {
				return true;
			}
		}
		return false;
	}

	private _isTargetFresh(
		task: CaptureTaskState,
		target: CaptureTaskTarget,
		probe: ReflectionProbe | LightProbe | IrradianceProbeGrid
	): boolean {
		if (this._activeTask?.taskId !== task.taskId) {
			return false;
		}
		if (probe.source !== "capturedScene") return false;
		const requestToken =
			target.kind === "grid" ?
				(probe as IrradianceProbeGrid).getCellCaptureRequestToken(
					target.cellIndex
				)
			:	probe.captureRequestToken;
		if (requestToken !== target.captureRequestToken) return false;
		if (this._getSceneDirtyStamp(task.scene) !== task.sceneDirtyStamp) {
			return false;
		}
		return buildProbeCaptureSignature(probe, target.cellIndex) === target.signature;
	}

	private _pruneProbeState(targets: ProbeCaptureTarget[]): void {
		const activeTargetKeys = new Set(
			targets.map((target) =>
				createProbeCaptureTargetKey(
					target.kind,
					target.id,
					target.cellIndex
				)
			)
		);
		pruneProbeMap(this._lastCaptureSecondsByProbeId, activeTargetKeys);
		pruneProbeMap(this._lastCaptureSceneDirtyStampByProbeId, activeTargetKeys);
		pruneProbeMap(this._lastHandledRequestTokenByProbeId, activeTargetKeys);
		if (
			this._activeTask &&
			this._activeTask.targets.some((target) =>
				activeTargetKeys.has(
					createProbeCaptureTargetKey(
						target.kind,
						target.id,
						target.cellIndex
					)
				)
			) === false
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

function collectCapturedSceneProbeTargets(
	lights: SceneLight[],
	cameraWorldPosition: IVector3 | null
): ProbeCaptureTarget[] {
	const targets: ProbeCaptureTarget[] = [];
	for (const light of lights) {
		if (light.type === LightType.ReflectionProbe) {
			const probe = light as ReflectionProbe;
			if (probe.source !== "capturedScene") continue;
			targets.push(createReflectionProbeCaptureTarget(probe));
			continue;
		}
		if (light.type === LightType.LightProbe) {
			const probe = light as LightProbe;
			if (probe.source !== "capturedScene") continue;
			targets.push(createLightProbeCaptureTarget(probe));
			continue;
		}
		if (light.type === LightType.IrradianceProbeGrid) {
			const grid = light as IrradianceProbeGrid;
			if (grid.source !== "capturedScene") continue;
			targets.push(...createIrradianceProbeGridCaptureTargets(grid));
		}
	}
	targets.sort((left, right) => {
		if (cameraWorldPosition) {
			const leftDistance = squaredDistanceToProbe(
				cameraWorldPosition,
				left.captureWorldPosition
			);
			const rightDistance = squaredDistanceToProbe(
				cameraWorldPosition,
				right.captureWorldPosition
			);
			if (leftDistance !== rightDistance) {
				return leftDistance - rightDistance;
			}
		}
		if (left.groupKey !== right.groupKey) {
			return left.groupKey.localeCompare(right.groupKey);
		}
		if (left.kind !== right.kind) {
			return left.kind.localeCompare(right.kind);
		}
		return left.id.localeCompare(right.id);
	});
	return targets;
}

function createReflectionProbeCaptureTarget(
	probe: ReflectionProbe
): ProbeCaptureTarget {
	const cache = probe.getRuntimeCache();
	const captureWorldPosition = {
		x: cache.captureWorldPosition.x,
		y: cache.captureWorldPosition.y,
		z: cache.captureWorldPosition.z,
	};
	return {
		kind: "reflection",
		id: probe.id,
		probe,
		cellIndex: -1,
		signature: buildProbeCaptureSignature(probe, -1),
		groupKey: buildProbeCaptureGroupKey(
			captureWorldPosition,
			probe.captureFar,
			probe.includeEnvironment,
			probe.includeMeshes,
			probe.includeTransparent,
			probe.includeParticles,
			probe.includeShadows
		),
		captureRequestToken: probe.captureRequestToken,
		captureUpdateMode: probe.captureUpdateMode,
		captureIntervalSeconds: probe.captureIntervalSeconds,
		captureWidth: probe.captureResolution.width,
		captureHeight: probe.captureResolution.height,
		captureFar: probe.captureFar,
		captureWorldPosition,
		includeEnvironment: probe.includeEnvironment,
		includeMeshes: probe.includeMeshes,
		includeTransparent: probe.includeTransparent,
		includeParticles: probe.includeParticles,
		includeShadows: probe.includeShadows,
	};
}

function createLightProbeCaptureTarget(probe: LightProbe): ProbeCaptureTarget {
	const cache = probe.getRuntimeCache();
	const captureWorldPosition = {
		x: cache.probeWorldPosition.x,
		y: cache.probeWorldPosition.y,
		z: cache.probeWorldPosition.z,
	};
	return {
		kind: "light",
		id: probe.id,
		probe,
		cellIndex: -1,
		signature: buildProbeCaptureSignature(probe, -1),
		groupKey: buildProbeCaptureGroupKey(
			captureWorldPosition,
			probe.captureFar,
			probe.includeEnvironment,
			probe.includeMeshes,
			probe.includeTransparent,
			probe.includeParticles,
			probe.includeShadows
		),
		captureRequestToken: probe.captureRequestToken,
		captureUpdateMode: probe.captureUpdateMode,
		captureIntervalSeconds: probe.captureIntervalSeconds,
		captureWidth: probe.captureResolution.width,
		captureHeight: probe.captureResolution.height,
		captureFar: probe.captureFar,
		captureWorldPosition,
		includeEnvironment: probe.includeEnvironment,
		includeMeshes: probe.includeMeshes,
		includeTransparent: probe.includeTransparent,
		includeParticles: probe.includeParticles,
		includeShadows: probe.includeShadows,
	};
}

function createIrradianceProbeGridCaptureTargets(
	grid: IrradianceProbeGrid
): ProbeCaptureTarget[] {
	const cache = grid.getRuntimeCache();
	const targets: ProbeCaptureTarget[] = [];
	for (let cellIndex = 0; cellIndex < cache.cellCount; cellIndex++) {
		const captureWorldPosition = cache.cellWorldPositions[cellIndex];
		targets.push({
			kind: "grid",
			id: grid.id,
			probe: grid,
			cellIndex,
			signature: buildProbeCaptureSignature(grid, cellIndex),
			groupKey: buildProbeCaptureGroupKey(
				captureWorldPosition,
				grid.captureFar,
				grid.includeEnvironment,
				grid.includeMeshes,
				grid.includeTransparent,
				grid.includeParticles,
				grid.includeShadows
			),
			captureRequestToken: grid.getCellCaptureRequestToken(cellIndex),
			captureUpdateMode: grid.captureUpdateMode,
			captureIntervalSeconds: grid.captureIntervalSeconds,
			captureWidth: grid.captureResolution.width,
			captureHeight: grid.captureResolution.height,
			captureFar: grid.captureFar,
			captureWorldPosition,
			includeEnvironment: grid.includeEnvironment,
			includeMeshes: grid.includeMeshes,
			includeTransparent: grid.includeTransparent,
			includeParticles: grid.includeParticles,
			includeShadows: grid.includeShadows,
		});
	}
	return targets;
}

function findCapturedSceneTarget(
	scene: Scene,
	kind: ProbeCaptureTargetKind,
	probeId: string
): ReflectionProbe | LightProbe | IrradianceProbeGrid | null {
	for (const light of scene.getLights()) {
		if (kind === "reflection") {
			if (light.type !== LightType.ReflectionProbe) continue;
			const probe = light as ReflectionProbe;
			if (probe.source !== "capturedScene") continue;
			if (probe.id === probeId) {
				return probe;
			}
			continue;
		}
		if (kind === "grid") {
			if (light.type !== LightType.IrradianceProbeGrid) continue;
			const grid = light as IrradianceProbeGrid;
			if (grid.source !== "capturedScene") continue;
			if (grid.id === probeId) {
				return grid;
			}
			continue;
		}
		if (light.type !== LightType.LightProbe) continue;
		const probe = light as LightProbe;
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

function buildProbeCaptureSignature(
	probe: ReflectionProbe | LightProbe | IrradianceProbeGrid,
	cellIndex: number
): string {
	const elements = probe.worldMatrix.elements;
	const matrix = new Array<string>(16);
	let cursor = 0;
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			matrix[cursor++] = elements[row][col].toFixed(6);
		}
	}
	const kind =
		probe.type === LightType.ReflectionProbe ? "reflection"
		: probe.type === LightType.IrradianceProbeGrid ? "grid"
		: "light";
	const grid =
		probe.type === LightType.IrradianceProbeGrid ?
			(probe as IrradianceProbeGrid)
		:	null;
	return [
		kind,
		cellIndex,
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
		grid ? grid.getCellCaptureRequestToken(cellIndex) : probe.captureRequestToken,
		grid ? grid.dimensions.x : 0,
		grid ? grid.dimensions.y : 0,
		grid ? grid.dimensions.z : 0,
		grid ? grid.halfExtents.x.toFixed(6) : 0,
		grid ? grid.halfExtents.y.toFixed(6) : 0,
		grid ? grid.halfExtents.z.toFixed(6) : 0,
		...matrix,
	].join("|");
}

function buildProbeCaptureGroupKey(
	captureWorldPosition: IVector3,
	captureFar: number,
	includeEnvironment: boolean,
	includeMeshes: boolean,
	includeTransparent: boolean,
	includeParticles: boolean,
	includeShadows: boolean
): string {
	return [
		captureWorldPosition.x.toFixed(6),
		captureWorldPosition.y.toFixed(6),
		captureWorldPosition.z.toFixed(6),
		captureFar.toFixed(6),
		includeEnvironment ? 1 : 0,
		includeMeshes ? 1 : 0,
		includeTransparent ? 1 : 0,
		includeParticles ? 1 : 0,
		includeShadows ? 1 : 0,
	].join("|");
}

function createProbeCaptureTargetKey(
	kind: ProbeCaptureTargetKind,
	id: string,
	cellIndex = -1
): string {
	return `${kind}:${id}:${cellIndex}`;
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

function writeProbeCaptureTextureOutputs(
	probe: ReflectionProbe | LightProbe | IrradianceProbeGrid,
	task: CaptureTaskState,
	environmentMap: Texture
): void {
	if (
		probe.type !== LightType.ReflectionProbe &&
		probe.type !== LightType.LightProbe
	) {
		return;
	}
	const output = probe.capture;
	const cubeTexture = output.cubeTexture;
	if (cubeTexture) {
		const faces = collectCapturedCubeFaces(task);
		if (faces) {
			cubeTexture.replaceFaces({
				faces,
				size: task.faceSize,
				colorSpace: "HDR",
			});
		}
	}
	const rawTexture = output.rawTexture;
	if (rawTexture) {
		copyTexturePayload(rawTexture, environmentMap);
	}
}

function collectCapturedCubeFaces(
	task: CaptureTaskState
): CubeTextureFaceData[] | null {
	const faces: CubeTextureFaceData[] = [];
	for (let faceIndex = 0; faceIndex < 6; faceIndex++) {
		const faceData = task.capturedFaces[faceIndex];
		if (!faceData) {
			return null;
		}
		faces.push(faceData);
	}
	return faces;
}

function copyTexturePayload(target: Texture, source: Texture): void {
	target.colorSpace = source.colorSpace;
	target.format = source.format;
	target.formatExplicit = source.formatExplicit;
	target.wrapS = source.wrapS;
	target.wrapT = source.wrapT;
	target.minFilter = source.minFilter;
	target.magFilter = source.magFilter;
	target.setMipLevels(cloneTextureLevels(source.levels));
}

function cloneTextureLevels(levels: TextureMipLevel[]): TextureMipLevel[] {
	return levels.map((level) => ({
		data: level.data,
		width: level.width,
		height: level.height,
		depthOrArrayLayers: level.depthOrArrayLayers,
		bytesPerRow: level.bytesPerRow,
		rowsPerImage: level.rowsPerImage,
	}));
}

function copySHCoefficients(
	target: SHCoefficients,
	source: SHCoefficients
): void {
	const coeffCount = Math.min(target.length, source.length);
	for (let i = 0; i < coeffCount; i++) {
		target[i].r = source[i].r;
		target[i].g = source[i].g;
		target[i].b = source[i].b;
	}
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
	task: CaptureTaskState
): CaptureLightingState {
	const lights = scene.getLights();
	const capturePosition = task.captureWorldPosition;
	const captureRange = Math.max(1, task.captureFar);
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
			task.includeEnvironment && scene.environment.backgroundEnabled ?
				scene.environment.backgroundTexture
			:	null,
		includeEnvironmentBackground:
			task.includeEnvironment && scene.environment.backgroundEnabled,
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
