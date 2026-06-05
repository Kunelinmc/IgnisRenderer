import type { Scene } from "../core/Scene";
import { Texture } from "../core/Texture";
import { Logger } from "../foundation/Logger";
import { LightProbe, LightType, ReflectionProbe } from "../lights";
import type { SHCoefficients } from "../maths/types";
import type { WebGPUComputeFacadeSource } from "../renderers/webgpu/ComputeFacade";
import {
	bakeEnvironmentIBLFromEnvironmentMap,
	type BakedEnvironmentIBL,
	type EnvironmentIBLBakeAcceleration,
	type EnvironmentIBLBakeOptions,
} from "./EnvironmentIBLBaker";
import {
	ensureEnvironmentTextureEquirect,
	getEnvironmentMipLevelCount,
	isTextureReadyForEnvironment,
} from "./environmentMapRuntime";
import type { RenderDirtyReason } from "./incremental";

const DEFAULT_MIPS_PER_FRAME = 1;
const DEFAULT_TEMPORAL_BLEND_FACTOR = 0.2;
const DEFAULT_TEMPORAL_BLEND_EPSILON = 1e-3;
const ENVIRONMENT_IBL_MIN_DIMENSION = 1;

interface ActiveBakeTask {
	taskId: number;
	signature: string;
	scene: Scene;
	controller: AbortController;
	status: "pending" | "ready" | "failed";
	result: BakedEnvironmentIBL | null;
}

interface PendingApplyState {
	signature: string;
	scene: Scene;
	baked: BakedEnvironmentIBL;
	blendedSpecularMap: Texture | null;
	mipConverged: boolean[];
	mipCursor: number;
}

interface EnvironmentIBLSignature {
	signature: string;
	texture: Texture;
}

export interface EnvironmentIBLUpdateOptions {
	enabled: boolean;
	autoUpdate: boolean;
	mipsPerFrame: number;
	temporalBlendFactor: number;
	temporalBlendEpsilon: number;
	acceleration: EnvironmentIBLBakeAcceleration;
	prefilterMaxSampleWidth?: number;
	prefilterMaxSampleHeight?: number;
	prefilterMaxMipLevels?: number;
	resetTemporalHistoryOnComplete: boolean;
}

export interface EnvironmentIBLUpdateRuntimeExecuteContext {
	scene: Scene;
	requestToken: number;
	options: EnvironmentIBLUpdateOptions;
	webgpuSource?: WebGPUComputeFacadeSource | null;
}

export interface EnvironmentIBLUpdateRuntimeExecuteResult {
	dirtyReason: RenderDirtyReason | null;
	inProgress: boolean;
}

interface EnvironmentIBLUpdateRuntimeOptions {
	bakeEnvironmentIBL?: (
		envMap: Texture,
		options: EnvironmentIBLBakeOptions
	) => Promise<BakedEnvironmentIBL>;
}

export const DEFAULT_ENVIRONMENT_IBL_UPDATE_OPTIONS: EnvironmentIBLUpdateOptions = {
	enabled: false,
	autoUpdate: true,
	mipsPerFrame: DEFAULT_MIPS_PER_FRAME,
	temporalBlendFactor: DEFAULT_TEMPORAL_BLEND_FACTOR,
	temporalBlendEpsilon: DEFAULT_TEMPORAL_BLEND_EPSILON,
	acceleration: "auto",
	resetTemporalHistoryOnComplete: true,
};

export function normalizeEnvironmentIBLUpdateOptions(
	options?: Partial<EnvironmentIBLUpdateOptions> | null
): EnvironmentIBLUpdateOptions {
	const source = options ?? {};
	return {
		enabled: source.enabled ?? DEFAULT_ENVIRONMENT_IBL_UPDATE_OPTIONS.enabled,
		autoUpdate:
			source.autoUpdate ?? DEFAULT_ENVIRONMENT_IBL_UPDATE_OPTIONS.autoUpdate,
		mipsPerFrame: clampInteger(
			source.mipsPerFrame,
			1,
			64,
			DEFAULT_ENVIRONMENT_IBL_UPDATE_OPTIONS.mipsPerFrame
		),
		temporalBlendFactor: clampNumber(
			source.temporalBlendFactor,
			0.01,
			1,
			DEFAULT_ENVIRONMENT_IBL_UPDATE_OPTIONS.temporalBlendFactor
		),
		temporalBlendEpsilon: clampNumber(
			source.temporalBlendEpsilon,
			1e-6,
			1,
			DEFAULT_ENVIRONMENT_IBL_UPDATE_OPTIONS.temporalBlendEpsilon
		),
		acceleration: sanitizeAcceleration(
			source.acceleration ?? DEFAULT_ENVIRONMENT_IBL_UPDATE_OPTIONS.acceleration
		),
		prefilterMaxSampleWidth: sanitizeOptionalDimension(
			source.prefilterMaxSampleWidth
		),
		prefilterMaxSampleHeight: sanitizeOptionalDimension(
			source.prefilterMaxSampleHeight
		),
		prefilterMaxMipLevels: sanitizeOptionalDimension(source.prefilterMaxMipLevels),
		resetTemporalHistoryOnComplete:
			source.resetTemporalHistoryOnComplete ??
			DEFAULT_ENVIRONMENT_IBL_UPDATE_OPTIONS.resetTemporalHistoryOnComplete,
	};
}

export class EnvironmentIBLUpdateRuntime {
	private _bakeEnvironmentIBL: (
		envMap: Texture,
		options: EnvironmentIBLBakeOptions
	) => Promise<BakedEnvironmentIBL>;
	private _activeBakeTask: ActiveBakeTask | null = null;
	private _pendingApplyState: PendingApplyState | null = null;
	private _nextTaskId = 0;
	private _lastHandledRequestToken = 0;
	private _lastRequestedSignature: string | null = null;
	private _lastEnvironmentSignatureByScene = new WeakMap<Scene, string | null>();
	private _textureIdentityByTexture = new WeakMap<Texture, number>();
	private _nextTextureIdentity = 0;

	constructor(options: EnvironmentIBLUpdateRuntimeOptions = {}) {
		this._bakeEnvironmentIBL =
			options.bakeEnvironmentIBL ?? bakeEnvironmentIBLFromEnvironmentMap;
	}

	public execute(
		context: EnvironmentIBLUpdateRuntimeExecuteContext
	): EnvironmentIBLUpdateRuntimeExecuteResult {
		const resolvedOptions = normalizeEnvironmentIBLUpdateOptions(context.options);
		if (!resolvedOptions.enabled) {
			this._cancelActiveTask();
			this._pendingApplyState = null;
			return {
				dirtyReason: null,
				inProgress: false,
			};
		}

		const signature = this._resolveEnvironmentSignature(context.scene);
		this._consumeSettledBakeTask();

		const hasManualRequest = context.requestToken > this._lastHandledRequestToken;
		if (hasManualRequest) {
			this._lastHandledRequestToken = context.requestToken;
			if (signature) {
				this._requestBakeForSignature(
					signature,
					context,
					resolvedOptions,
					true
				);
			}
		} else if (
			resolvedOptions.autoUpdate &&
			signature &&
			this._shouldAutoUpdate(context.scene, signature.signature)
		) {
			this._requestBakeForSignature(
				signature,
				context,
				resolvedOptions,
				false
			);
		}

		let completedThisFrame = false;
		if (this._pendingApplyState) {
			completedThisFrame = this._applyPendingBake(
				this._pendingApplyState,
				resolvedOptions
			);
			if (completedThisFrame) {
				this._pendingApplyState = null;
			}
		}

		const inProgress =
			this._activeBakeTask !== null || this._pendingApplyState !== null;
		if (completedThisFrame && resolvedOptions.resetTemporalHistoryOnComplete) {
			return {
				dirtyReason: "environment-ibl-complete",
				inProgress,
			};
		}
		if (inProgress) {
			return {
				dirtyReason: "environment-ibl",
				inProgress,
			};
		}
		return {
			dirtyReason: null,
			inProgress: false,
		};
	}

	public reset(): void {
		this._cancelActiveTask();
		this._pendingApplyState = null;
		this._lastRequestedSignature = null;
		this._lastHandledRequestToken = 0;
	}

	private _resolveEnvironmentSignature(scene: Scene): EnvironmentIBLSignature | null {
		const environment = scene.environment;
		if (!environment.lightingEnabled) {
			this._lastEnvironmentSignatureByScene.set(scene, null);
			return null;
		}
		const normalizedEnvironment = ensureEnvironmentTextureEquirect(
			environment.iblTexture
		);
		if (!isTextureReadyForEnvironment(normalizedEnvironment)) {
			this._lastEnvironmentSignatureByScene.set(scene, null);
			return null;
		}
		const textureIdentity = this._resolveTextureIdentity(normalizedEnvironment);
		const signature =
			`${textureIdentity}:${normalizedEnvironment.version}:` +
			`${normalizedEnvironment.width}x${normalizedEnvironment.height}:` +
			`${getEnvironmentMipLevelCount(normalizedEnvironment)}:${normalizedEnvironment.colorSpace}`;
		return {
			signature,
			texture: normalizedEnvironment,
		};
	}

	private _shouldAutoUpdate(scene: Scene, signature: string): boolean {
		const previous = this._lastEnvironmentSignatureByScene.get(scene);
		this._lastEnvironmentSignatureByScene.set(scene, signature);
		if (previous === undefined) {
			return true;
		}
		return previous !== signature;
	}

	private _requestBakeForSignature(
		signature: EnvironmentIBLSignature,
		context: EnvironmentIBLUpdateRuntimeExecuteContext,
		options: EnvironmentIBLUpdateOptions,
		force: boolean
	): void {
		if (
			this._activeBakeTask &&
			this._activeBakeTask.signature === signature.signature &&
			this._activeBakeTask.status === "pending"
		) {
			return;
		}
		if (
			!force &&
			this._lastRequestedSignature === signature.signature &&
			!this._activeBakeTask
		) {
			return;
		}
		this._lastRequestedSignature = signature.signature;
		this._startBakeTask(signature, context, options);
	}

	private _startBakeTask(
		signature: EnvironmentIBLSignature,
		context: EnvironmentIBLUpdateRuntimeExecuteContext,
		options: EnvironmentIBLUpdateOptions
	): void {
		this._cancelActiveTask();
		const taskId = ++this._nextTaskId;
		const controller = new AbortController();
		const bakeOptions: EnvironmentIBLBakeOptions = {
			acceleration: this._resolveBakeAcceleration(
				options.acceleration,
				context.webgpuSource ?? null
			),
			signal: controller.signal,
			prefilterMaxSampleWidth: options.prefilterMaxSampleWidth,
			prefilterMaxSampleHeight: options.prefilterMaxSampleHeight,
			prefilterMaxMipLevels: options.prefilterMaxMipLevels,
		};
		if (context.webgpuSource) {
			bakeOptions.webgpuSource = context.webgpuSource;
		}

		const activeTask: ActiveBakeTask = {
			taskId,
			signature: signature.signature,
			scene: context.scene,
			controller,
			status: "pending",
			result: null,
		};
		this._activeBakeTask = activeTask;

		void this._bakeEnvironmentIBL(signature.texture, bakeOptions)
			.then((result) => {
				if (!this._activeBakeTask || this._activeBakeTask.taskId !== taskId) {
					return;
				}
				this._activeBakeTask.status = "ready";
				this._activeBakeTask.result = result;
			})
			.catch((error) => {
				if (!this._activeBakeTask || this._activeBakeTask.taskId !== taskId) {
					return;
				}
				if (controller.signal.aborted) {
					this._activeBakeTask = null;
					return;
				}
				this._activeBakeTask.status = "failed";
				const detail = error instanceof Error ? error.message : String(error);
				Logger.warn(
					`[environment-ibl-update-bake-failed] Environment IBL update bake failed: ${detail}`,
					{
						scope: "EnvironmentIBLUpdateRuntime",
						onceKey: "environment-ibl-update-bake-failed",
					}
				);
			});
	}

	private _consumeSettledBakeTask(): void {
		const activeTask = this._activeBakeTask;
		if (!activeTask) return;

		if (activeTask.status === "failed") {
			this._activeBakeTask = null;
			return;
		}
		if (activeTask.status !== "ready" || !activeTask.result) {
			return;
		}
		const mipCount = Math.max(1, getEnvironmentMipLevelCount(activeTask.result.prefilteredMap));
		this._pendingApplyState = {
			signature: activeTask.signature,
			scene: activeTask.scene,
			baked: activeTask.result,
			blendedSpecularMap: null,
			mipConverged: new Array<boolean>(mipCount).fill(false),
			mipCursor: 0,
		};
		this._activeBakeTask = null;
	}

	private _applyPendingBake(
		pending: PendingApplyState,
		options: EnvironmentIBLUpdateOptions
	): boolean {
		const shConverged = this._blendLightProbeSH(
			pending.scene,
			pending.baked.sh,
			options.temporalBlendFactor,
			options.temporalBlendEpsilon
		);
		const specularConverged = this._blendEnvironmentReflectionSpecular(
			pending,
			options
		);
		return shConverged && specularConverged;
	}

	private _blendLightProbeSH(
		scene: Scene,
		target: SHCoefficients,
		blendFactor: number,
		epsilon: number
	): boolean {
		const probes: LightProbe[] = scene
			.getLights()
			.filter(
				(light): light is LightProbe =>
					light.type === LightType.LightProbe &&
					(light as LightProbe).source === "environment"
			);
		if (probes.length === 0) {
			probes.push(scene.add(new LightProbe({})));
		}

		let converged = true;
		for (const probe of probes) {
			const coeffCount = Math.min(probe.sh.length, target.length);
			for (let i = 0; i < coeffCount; i++) {
				const current = probe.sh[i];
				const goal = target[i];
				current.r = blendFloat(current.r, goal.r, blendFactor);
				current.g = blendFloat(current.g, goal.g, blendFactor);
				current.b = blendFloat(current.b, goal.b, blendFactor);
				if (
					Math.abs(goal.r - current.r) > epsilon ||
					Math.abs(goal.g - current.g) > epsilon ||
					Math.abs(goal.b - current.b) > epsilon
				) {
					converged = false;
				}
			}
		}
		return converged;
	}

	private _blendEnvironmentReflectionSpecular(
		pending: PendingApplyState,
		options: EnvironmentIBLUpdateOptions
	): boolean {
		const environmentProbes = pending.scene
			.getLights()
			.filter(
				(light): light is ReflectionProbe =>
					light.type === LightType.ReflectionProbe &&
					light.source === "environment"
			);
		if (environmentProbes.length <= 0) {
			return true;
		}

		const targetMap = pending.baked.prefilteredMap;
		if (!isTextureReadyForEnvironment(targetMap)) {
			return true;
		}

		if (
			!pending.blendedSpecularMap ||
			!isBlendTextureCompatible(pending.blendedSpecularMap, targetMap)
		) {
			const initialSource = this._resolveInitialSpecularSource(
				environmentProbes,
				targetMap
			);
			pending.blendedSpecularMap = cloneTextureWithMipmaps(initialSource);
		}
		const blendedMap = pending.blendedSpecularMap;
		const mipCount = Math.max(1, getEnvironmentMipLevelCount(targetMap));
		if (pending.mipConverged.length !== mipCount) {
			pending.mipConverged = new Array<boolean>(mipCount).fill(false);
			pending.mipCursor = 0;
		}

		const budget = Math.max(1, options.mipsPerFrame);
		let processed = 0;
		let scanned = 0;
		let cursor = pending.mipCursor;
		while (processed < budget && scanned < mipCount) {
			const mipLevel = (cursor + scanned) % mipCount;
			scanned++;
			if (pending.mipConverged[mipLevel]) {
				continue;
			}
			pending.mipConverged[mipLevel] = blendTextureMipTowardsTarget(
				blendedMap,
				targetMap,
				mipLevel,
				options.temporalBlendFactor,
				options.temporalBlendEpsilon
			);
			processed++;
		}
		pending.mipCursor = (cursor + scanned) % Math.max(1, mipCount);

		if (processed > 0) {
			blendedMap.markNeedsUpdate();
		}
		for (const probe of environmentProbes) {
			probe.prefilteredMap = blendedMap;
			probe.markRuntimeDirty();
		}
		for (const probe of environmentProbes) {
			probe.markCaptureUpdated();
		}

		return pending.mipConverged.every((value) => value === true);
	}

	private _resolveInitialSpecularSource(
		probes: ReflectionProbe[],
		targetMap: Texture
	): Texture {
		for (const probe of probes) {
			if (!probe.prefilteredMap) continue;
			if (!isTextureReadyForEnvironment(probe.prefilteredMap)) continue;
			if (!isBlendTextureCompatible(probe.prefilteredMap, targetMap)) continue;
			return probe.prefilteredMap;
		}
		return targetMap;
	}

	private _resolveBakeAcceleration(
		acceleration: EnvironmentIBLBakeAcceleration,
		webgpuSource: WebGPUComputeFacadeSource | null
	): EnvironmentIBLBakeAcceleration {
		if (acceleration === "cpu" || acceleration === "worker") {
			return "cpu";
		}
		if (acceleration === "webgpu") {
			return webgpuSource ? "webgpu" : "cpu";
		}
		return webgpuSource ? "webgpu" : "cpu";
	}

	private _resolveTextureIdentity(texture: Texture): number {
		const cached = this._textureIdentityByTexture.get(texture);
		if (cached !== undefined) {
			return cached;
		}
		const identity = ++this._nextTextureIdentity;
		this._textureIdentityByTexture.set(texture, identity);
		return identity;
	}

	private _cancelActiveTask(): void {
		if (!this._activeBakeTask) return;
		this._activeBakeTask.controller.abort();
		this._activeBakeTask = null;
	}
}

function sanitizeAcceleration(
	value: EnvironmentIBLBakeAcceleration
): EnvironmentIBLBakeAcceleration {
	if (
		value === "auto" ||
		value === "cpu" ||
		value === "worker" ||
		value === "webgpu"
	) {
		return value;
	}
	return "auto";
}

function sanitizeOptionalDimension(value: number | undefined): number | undefined {
	if (!Number.isFinite(value)) return undefined;
	return Math.max(ENVIRONMENT_IBL_MIN_DIMENSION, Math.floor(value as number));
}

function clampInteger(
	value: number | undefined,
	min: number,
	max: number,
	fallback: number
): number {
	if (!Number.isFinite(value)) {
		return fallback;
	}
	return Math.max(min, Math.min(max, Math.floor(value as number)));
}

function clampNumber(
	value: number | undefined,
	min: number,
	max: number,
	fallback: number
): number {
	if (!Number.isFinite(value)) {
		return fallback;
	}
	return Math.max(min, Math.min(max, value as number));
}

function isBlendTextureCompatible(left: Texture, right: Texture): boolean {
	return (
		left.width === right.width &&
		left.height === right.height &&
		left.colorSpace === right.colorSpace &&
		getEnvironmentMipLevelCount(left) === getEnvironmentMipLevelCount(right)
	);
}

function cloneTextureWithMipmaps(source: Texture): Texture {
	const mipmaps = source.mipmaps.map(cloneArrayBufferLike);
	const cloned = new Texture(
		mipmaps[0] ?? cloneArrayBufferLike(source.data),
		source.width,
		source.height,
		source.colorSpace
	);
	cloned.wrapS = source.wrapS;
	cloned.wrapT = source.wrapT;
	cloned.minFilter = source.minFilter;
	cloned.magFilter = source.magFilter;
	cloned.offset = { ...source.offset };
	cloned.repeat = { ...source.repeat };
	cloned.rotation = source.rotation;
	cloned.mipmaps = mipmaps;
	cloned.data = (mipmaps[0] ?? cloneArrayBufferLike(source.data)) as Texture["data"];
	return cloned;
}

function cloneArrayBufferLike(
	value: Uint8ClampedArray | Uint8Array | Float32Array | null | undefined
): Uint8ClampedArray | Uint8Array | Float32Array | null {
	if (!value) return null;
	if (value instanceof Float32Array) {
		return new Float32Array(value);
	}
	if (value instanceof Uint8ClampedArray) {
		return new Uint8ClampedArray(value);
	}
	return new Uint8Array(value);
}

function blendTextureMipTowardsTarget(
	targetTexture: Texture,
	referenceTexture: Texture,
	mipLevel: number,
	blendFactor: number,
	epsilon: number
): boolean {
	const targetMip = resolveMipData(referenceTexture, mipLevel);
	if (!targetMip) return true;
	const blendedMip = ensureCompatibleMipStorage(targetTexture, mipLevel, targetMip);
	if (!blendedMip) return true;
	if (blendedMip.length !== targetMip.length) {
		return true;
	}

	let converged = true;
	if (blendedMip instanceof Float32Array && targetMip instanceof Float32Array) {
		for (let i = 0; i < targetMip.length; i++) {
			const next = blendFloat(blendedMip[i], targetMip[i], blendFactor);
			blendedMip[i] = next;
			if (Math.abs(targetMip[i] - next) > epsilon) {
				converged = false;
			}
		}
		return converged;
	}

	const epsilon8 = Math.max(0.5, epsilon * 255);
	for (let i = 0; i < targetMip.length; i++) {
		const sourceValue = blendedMip[i];
		const goal = targetMip[i];
		const next = clampToByte(blendFloat(sourceValue, goal, blendFactor));
		blendedMip[i] = next;
		if (Math.abs(goal - next) > epsilon8) {
			converged = false;
		}
	}
	return converged;
}

function resolveMipData(
	texture: Texture,
	mipLevel: number
): Uint8ClampedArray | Uint8Array | Float32Array | null {
	const mip = texture.mipmaps[mipLevel];
	if (mip) return mip;
	if (mipLevel === 0) {
		return texture.data;
	}
	return null;
}

function ensureCompatibleMipStorage(
	texture: Texture,
	mipLevel: number,
	source: Uint8ClampedArray | Uint8Array | Float32Array
): Uint8ClampedArray | Uint8Array | Float32Array | null {
	let existing = resolveMipData(texture, mipLevel);
	if (
		existing &&
		existing.length === source.length &&
		((existing instanceof Float32Array && source instanceof Float32Array) ||
			(existing instanceof Uint8ClampedArray && source instanceof Uint8ClampedArray) ||
			(existing instanceof Uint8Array && source instanceof Uint8Array))
	) {
		return existing;
	}

	const replacement = cloneArrayBufferLike(source);
	if (!replacement) {
		return null;
	}
	texture.mipmaps[mipLevel] = replacement;
	if (mipLevel === 0) {
		texture.data = replacement;
	}
	return replacement;
}

function blendFloat(current: number, target: number, factor: number): number {
	return current + (target - current) * factor;
}

function clampToByte(value: number): number {
	if (!Number.isFinite(value)) return 0;
	return Math.max(0, Math.min(255, Math.round(value)));
}
