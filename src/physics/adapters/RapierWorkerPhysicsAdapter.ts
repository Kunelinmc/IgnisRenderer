import type { IVector3 } from "../../maths/types";
import { Platform } from "../../foundation/Platform";
import type {
	WorkerLike,
	WorkerPoolStats,
	WorkerRuntimeCapabilities,
	WorkerTransportPlugin,
} from "../../workers/types";
import { WorkerScheduler } from "../../workers/WorkerScheduler";
import { DEFAULT_WORKER_TRANSPORT_PLUGINS } from "../../workers/transports";
import type {
	IPhysicsEngineAdapter,
	PhysicsAdapterCapabilities,
	PhysicsAdapterStepResult,
} from "../IPhysicsEngineAdapter";
import type {
	CharacterControllerDescriptor,
	CharacterMoveResult,
	ColliderDescriptor,
	ColliderShape,
	JointDescriptor,
	PhysicsBoxCastQuery,
	PhysicsOverlapBoxQuery,
	PhysicsOverlapHit,
	PhysicsOverlapSphereQuery,
	PhysicsQueryHit,
	PhysicsRaycastQuery,
	PhysicsSphereCastQuery,
	PhysicsTransform,
	PhysicsWorldConfig,
	RigidBodyDescriptor,
} from "../types";
import { RapierPhysicsAdapter } from "./RapierPhysicsAdapter";
import type {
	RapierWorkerCommand,
	RapierWorkerTaskPayload,
} from "../workers/rapierWorkerProtocol";

const DEFAULT_WORKER_POOL_ID = "physics-rapier-worker";

function createUnsupportedSyncWorkerCallError(methodName: string): Error {
	return new Error(
		`RapierWorkerPhysicsAdapter.${methodName} is not available in synchronous mode. Use stepAsync() via PhysicsSystem or switch to RapierPhysicsAdapter.`
	);
}

function createDefaultWorker(
	workerIndex: number,
	poolId: string
): WorkerLike {
	const workerCtor =
		(globalThis as typeof globalThis & { Worker?: new (...args: any[]) => Worker })
			.Worker;
	if (typeof workerCtor !== "function") {
		throw new Error(
			`Worker constructor is unavailable for pool "${poolId}" (worker #${workerIndex})`
		);
	}

	return new workerCtor(
		new URL("../workers/rapierPhysics.worker.ts", import.meta.url),
		{ type: "module" }
	) as unknown as WorkerLike;
}

export interface RapierWorkerPhysicsAdapterOptions {
	enabled?: boolean;
	createWorker?: (workerIndex: number, poolId: string) => WorkerLike;
	scheduler?: WorkerScheduler;
	poolId?: string;
	strict?: boolean;
	fallbackOnWorkerFailure?: boolean;
	fallbackAdapter?: IPhysicsEngineAdapter;
	runtimeCapabilities?: Partial<WorkerRuntimeCapabilities>;
	transportPlugins?: WorkerTransportPlugin[];
	defaultTimeoutMs?: number;
}

export class RapierWorkerPhysicsAdapter implements IPhysicsEngineAdapter {
	public readonly id = "rapier-worker";
	public readonly capabilities: PhysicsAdapterCapabilities;

	private _enabled: boolean;
	private _strict: boolean;
	private _fallbackOnWorkerFailure: boolean;
	private _fallbackAdapter: IPhysicsEngineAdapter;
	private _runtimeCapabilities: Partial<WorkerRuntimeCapabilities>;
	private _transportPlugins: WorkerTransportPlugin[];
	private _createWorker: (workerIndex: number, poolId: string) => WorkerLike;
	private _scheduler: WorkerScheduler;
	private _poolId: string;
	private _defaultTimeoutMs?: number;
	private _initialized = false;
	private _usingFallbackAdapter = false;
	private _pendingCommands: RapierWorkerCommand[] = [];
	private _worldIds = new Set<string>();
	private _stepChain: Promise<void> = Promise.resolve();

	constructor(options: RapierWorkerPhysicsAdapterOptions = {}) {
		this._enabled = options.enabled ?? true;
		this._strict = options.strict ?? true;
		this._fallbackOnWorkerFailure = options.fallbackOnWorkerFailure ?? true;
		this._fallbackAdapter =
			options.fallbackAdapter ??
			new RapierPhysicsAdapter({
				strict: this._strict,
			});
		this._runtimeCapabilities = options.runtimeCapabilities ?? {
			sharedArrayBuffer: Platform.supportsSharedArrayBufferTransport(),
			crossOriginIsolated: Platform.isCrossOriginIsolated(globalThis, false),
		};
		this._transportPlugins =
			options.transportPlugins ?? DEFAULT_WORKER_TRANSPORT_PLUGINS;
		this._createWorker = options.createWorker ?? createDefaultWorker;
		this._scheduler = options.scheduler ?? new WorkerScheduler();
		this._poolId = options.poolId ?? DEFAULT_WORKER_POOL_ID;
		this._defaultTimeoutMs = options.defaultTimeoutMs;
		this.capabilities = {
			joints: true,
			characterController: false,
			shapeCast: false,
			query: false,
			syncInit: false,
		};
	}

	public async init(): Promise<void> {
		if (!this._enabled) {
			await this._initializeFallback();
			return;
		}

		try {
			this._ensureWorkerPool();
			await this._dispatchToWorker({
				type: "init",
				strict: this._strict,
			});
			this._initialized = true;
		} catch (error) {
			if (!this._fallbackOnWorkerFailure) {
				throw error;
			}
			await this._initializeFallback();
		}
	}

	public initSync(): void {
		throw new Error(
			"RapierWorkerPhysicsAdapter.initSync is not supported because worker initialization is asynchronous. Use init()."
		);
	}

	public hasWorld(worldId: string): boolean {
		if (this._usingFallbackAdapter) return this._fallbackAdapter.hasWorld(worldId);
		return this._worldIds.has(worldId);
	}

	public createWorld(config: PhysicsWorldConfig): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.createWorld(config);
			return;
		}
		this._assertInitialized();
		if (this._worldIds.has(config.worldId)) {
			throw new Error(`Physics world "${config.worldId}" already exists`);
		}
		this._worldIds.add(config.worldId);
		this._enqueueCommand({
			type: "createWorld",
			config,
		});
	}

	public destroyWorld(worldId: string): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.destroyWorld(worldId);
			return;
		}
		this._assertInitialized();
		this._worldIds.delete(worldId);
		this._enqueueCommand({
			type: "destroyWorld",
			worldId,
		});
	}

	public createBody(
		worldId: string,
		bodyId: string,
		descriptor: RigidBodyDescriptor,
		initialTransform: PhysicsTransform
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.createBody(
				worldId,
				bodyId,
				descriptor,
				initialTransform
			);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "createBody",
			worldId,
			bodyId,
			descriptor,
			initialTransform,
		});
	}

	public destroyBody(worldId: string, bodyId: string): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.destroyBody(worldId, bodyId);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "destroyBody",
			worldId,
			bodyId,
		});
	}

	public setBodyTransform(
		worldId: string,
		bodyId: string,
		transform: PhysicsTransform
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.setBodyTransform(worldId, bodyId, transform);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "setBodyTransform",
			worldId,
			bodyId,
			transform,
		});
	}

	public setBodyLinearVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.setBodyLinearVelocity(worldId, bodyId, velocity);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "setBodyLinearVelocity",
			worldId,
			bodyId,
			velocity,
		});
	}

	public setAngularVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.setAngularVelocity(worldId, bodyId, velocity);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "setAngularVelocity",
			worldId,
			bodyId,
			velocity,
		});
	}

	public applyForce(worldId: string, bodyId: string, force: IVector3): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.applyForce(worldId, bodyId, force);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "applyForce",
			worldId,
			bodyId,
			force,
		});
	}

	public applyTorque(worldId: string, bodyId: string, torque: IVector3): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.applyTorque(worldId, bodyId, torque);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "applyTorque",
			worldId,
			bodyId,
			torque,
		});
	}

	public applyImpulse(
		worldId: string,
		bodyId: string,
		impulse: IVector3
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.applyImpulse(worldId, bodyId, impulse);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "applyImpulse",
			worldId,
			bodyId,
			impulse,
		});
	}

	public addCollider(
		worldId: string,
		bodyId: string,
		colliderId: string,
		descriptor: ColliderDescriptor,
		shape: ColliderShape
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.addCollider(
				worldId,
				bodyId,
				colliderId,
				descriptor,
				shape
			);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "addCollider",
			worldId,
			bodyId,
			colliderId,
			descriptor,
			shape,
		});
	}

	public destroyCollider(worldId: string, colliderId: string): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.destroyCollider(worldId, colliderId);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "destroyCollider",
			worldId,
			colliderId,
		});
	}

	public setColliderSensor(
		worldId: string,
		colliderId: string,
		isSensor: boolean
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.setColliderSensor(worldId, colliderId, isSensor);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "setColliderSensor",
			worldId,
			colliderId,
			isSensor,
		});
	}

	public setCollisionMask(
		worldId: string,
		colliderId: string,
		mask: number
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.setCollisionMask(worldId, colliderId, mask);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "setCollisionMask",
			worldId,
			colliderId,
			mask,
		});
	}

	public createJoint(
		worldId: string,
		jointId: string,
		descriptor: JointDescriptor
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.createJoint(worldId, jointId, descriptor);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "createJoint",
			worldId,
			jointId,
			descriptor,
		});
	}

	public destroyJoint(worldId: string, jointId: string): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.destroyJoint(worldId, jointId);
			return;
		}
		this._assertInitialized();
		this._enqueueCommand({
			type: "destroyJoint",
			worldId,
			jointId,
		});
	}

	public createCharacterController(
		worldId: string,
		controllerId: string,
		descriptor: CharacterControllerDescriptor
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.createCharacterController(
				worldId,
				controllerId,
				descriptor
			);
			return;
		}
		throw createUnsupportedSyncWorkerCallError("createCharacterController");
	}

	public destroyCharacterController(
		worldId: string,
		controllerId: string
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.destroyCharacterController(worldId, controllerId);
			return;
		}
		throw createUnsupportedSyncWorkerCallError("destroyCharacterController");
	}

	public moveCharacterController(
		worldId: string,
		controllerId: string,
		direction: IVector3,
		deltaSeconds: number
	): CharacterMoveResult {
		if (this._usingFallbackAdapter) {
			return this._fallbackAdapter.moveCharacterController(
				worldId,
				controllerId,
				direction,
				deltaSeconds
			);
		}
		throw createUnsupportedSyncWorkerCallError("moveCharacterController");
	}

	public jumpCharacterController(
		worldId: string,
		controllerId: string,
		speed: number
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.jumpCharacterController(worldId, controllerId, speed);
			return;
		}
		throw createUnsupportedSyncWorkerCallError("jumpCharacterController");
	}

	public isCharacterControllerGrounded(
		worldId: string,
		controllerId: string
	): boolean {
		if (this._usingFallbackAdapter) {
			return this._fallbackAdapter.isCharacterControllerGrounded(
				worldId,
				controllerId
			);
		}
		throw createUnsupportedSyncWorkerCallError("isCharacterControllerGrounded");
	}

	public setCharacterControllerMaxSlope(
		worldId: string,
		controllerId: string,
		value: number
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.setCharacterControllerMaxSlope(
				worldId,
				controllerId,
				value
			);
			return;
		}
		throw createUnsupportedSyncWorkerCallError("setCharacterControllerMaxSlope");
	}

	public setCharacterControllerStepHeight(
		worldId: string,
		controllerId: string,
		value: number
	): void {
		if (this._usingFallbackAdapter) {
			this._fallbackAdapter.setCharacterControllerStepHeight(
				worldId,
				controllerId,
				value
			);
			return;
		}
		throw createUnsupportedSyncWorkerCallError("setCharacterControllerStepHeight");
	}

	public raycast(worldId: string, query: PhysicsRaycastQuery): PhysicsQueryHit | null {
		if (this._usingFallbackAdapter) {
			return this._fallbackAdapter.raycast(worldId, query);
		}
		throw createUnsupportedSyncWorkerCallError("raycast");
	}

	public async raycastAsync(
		worldId: string,
		query: PhysicsRaycastQuery
	): Promise<PhysicsQueryHit | null> {
		if (this._usingFallbackAdapter) {
			if (this._fallbackAdapter.raycastAsync) {
				return this._fallbackAdapter.raycastAsync(worldId, query);
			}
			return this._fallbackAdapter.raycast(worldId, query);
		}
		this._assertInitialized();
		return this._dispatchToWorker<PhysicsQueryHit | null>({
			type: "dispatch",
			commands: this._drainPendingCommands(),
			request: {
				type: "raycast",
				worldId,
				query,
			},
		});
	}

	public raycastAll(
		worldId: string,
		query: PhysicsRaycastQuery
	): PhysicsQueryHit[] {
		if (this._usingFallbackAdapter) {
			return this._fallbackAdapter.raycastAll(worldId, query);
		}
		throw createUnsupportedSyncWorkerCallError("raycastAll");
	}

	public async raycastAllAsync(
		worldId: string,
		query: PhysicsRaycastQuery
	): Promise<PhysicsQueryHit[]> {
		if (this._usingFallbackAdapter) {
			if (this._fallbackAdapter.raycastAllAsync) {
				return this._fallbackAdapter.raycastAllAsync(worldId, query);
			}
			return this._fallbackAdapter.raycastAll(worldId, query);
		}
		this._assertInitialized();
		return this._dispatchToWorker<PhysicsQueryHit[]>({
			type: "dispatch",
			commands: this._drainPendingCommands(),
			request: {
				type: "raycastAll",
				worldId,
				query,
			},
		});
	}

	public sphereCast(
		worldId: string,
		query: PhysicsSphereCastQuery
	): PhysicsQueryHit | null {
		if (this._usingFallbackAdapter) {
			return this._fallbackAdapter.sphereCast(worldId, query);
		}
		throw createUnsupportedSyncWorkerCallError("sphereCast");
	}

	public async sphereCastAsync(
		worldId: string,
		query: PhysicsSphereCastQuery
	): Promise<PhysicsQueryHit | null> {
		if (this._usingFallbackAdapter) {
			if (this._fallbackAdapter.sphereCastAsync) {
				return this._fallbackAdapter.sphereCastAsync(worldId, query);
			}
			return this._fallbackAdapter.sphereCast(worldId, query);
		}
		this._assertInitialized();
		return this._dispatchToWorker<PhysicsQueryHit | null>({
			type: "dispatch",
			commands: this._drainPendingCommands(),
			request: {
				type: "sphereCast",
				worldId,
				query,
			},
		});
	}

	public boxCast(
		worldId: string,
		query: PhysicsBoxCastQuery
	): PhysicsQueryHit | null {
		if (this._usingFallbackAdapter) {
			return this._fallbackAdapter.boxCast(worldId, query);
		}
		throw createUnsupportedSyncWorkerCallError("boxCast");
	}

	public async boxCastAsync(
		worldId: string,
		query: PhysicsBoxCastQuery
	): Promise<PhysicsQueryHit | null> {
		if (this._usingFallbackAdapter) {
			if (this._fallbackAdapter.boxCastAsync) {
				return this._fallbackAdapter.boxCastAsync(worldId, query);
			}
			return this._fallbackAdapter.boxCast(worldId, query);
		}
		this._assertInitialized();
		return this._dispatchToWorker<PhysicsQueryHit | null>({
			type: "dispatch",
			commands: this._drainPendingCommands(),
			request: {
				type: "boxCast",
				worldId,
				query,
			},
		});
	}

	public overlapSphere(
		worldId: string,
		query: PhysicsOverlapSphereQuery
	): PhysicsOverlapHit[] {
		if (this._usingFallbackAdapter) {
			return this._fallbackAdapter.overlapSphere(worldId, query);
		}
		throw createUnsupportedSyncWorkerCallError("overlapSphere");
	}

	public async overlapSphereAsync(
		worldId: string,
		query: PhysicsOverlapSphereQuery
	): Promise<PhysicsOverlapHit[]> {
		if (this._usingFallbackAdapter) {
			if (this._fallbackAdapter.overlapSphereAsync) {
				return this._fallbackAdapter.overlapSphereAsync(worldId, query);
			}
			return this._fallbackAdapter.overlapSphere(worldId, query);
		}
		this._assertInitialized();
		return this._dispatchToWorker<PhysicsOverlapHit[]>({
			type: "dispatch",
			commands: this._drainPendingCommands(),
			request: {
				type: "overlapSphere",
				worldId,
				query,
			},
		});
	}

	public overlapBox(
		worldId: string,
		query: PhysicsOverlapBoxQuery
	): PhysicsOverlapHit[] {
		if (this._usingFallbackAdapter) {
			return this._fallbackAdapter.overlapBox(worldId, query);
		}
		throw createUnsupportedSyncWorkerCallError("overlapBox");
	}

	public async overlapBoxAsync(
		worldId: string,
		query: PhysicsOverlapBoxQuery
	): Promise<PhysicsOverlapHit[]> {
		if (this._usingFallbackAdapter) {
			if (this._fallbackAdapter.overlapBoxAsync) {
				return this._fallbackAdapter.overlapBoxAsync(worldId, query);
			}
			return this._fallbackAdapter.overlapBox(worldId, query);
		}
		this._assertInitialized();
		return this._dispatchToWorker<PhysicsOverlapHit[]>({
			type: "dispatch",
			commands: this._drainPendingCommands(),
			request: {
				type: "overlapBox",
				worldId,
				query,
			},
		});
	}

	public stepWorld(
		worldId: string,
		deltaSeconds: number
	): PhysicsAdapterStepResult {
		if (this._usingFallbackAdapter) {
			return this._fallbackAdapter.stepWorld(worldId, deltaSeconds);
		}
		throw createUnsupportedSyncWorkerCallError("stepWorld");
	}

	public async stepWorldAsync(
		worldId: string,
		deltaSeconds: number
	): Promise<PhysicsAdapterStepResult> {
		if (this._usingFallbackAdapter) {
			if (this._fallbackAdapter.stepWorldAsync) {
				return this._fallbackAdapter.stepWorldAsync(worldId, deltaSeconds);
			}
			return this._fallbackAdapter.stepWorld(worldId, deltaSeconds);
		}
		this._assertInitialized();
		const execute = async (): Promise<PhysicsAdapterStepResult> => {
			const result = await this._dispatchToWorker<PhysicsAdapterStepResult>({
				type: "dispatch",
				commands: this._drainPendingCommands(),
				request: {
					type: "stepWorld",
					worldId,
					deltaSeconds,
				},
			});
			return result;
		};

		const chained = this._stepChain.then(execute, execute);
		this._stepChain = chained.then(
			() => undefined,
			() => undefined
		);
		return chained;
	}

	public getWorkerPoolStats(): WorkerPoolStats | null {
		if (this._usingFallbackAdapter) return null;
		return this._scheduler.getPoolStats(this._poolId);
	}

	private _ensureWorkerPool(): void {
		if (this._scheduler.hasPool(this._poolId)) return;
		this._scheduler.registerPool({
			id: this._poolId,
			size: 1,
			createWorker: (workerIndex, poolId) =>
				this._createWorker(workerIndex, poolId),
			transportPlugins: this._transportPlugins,
			runtimeCapabilities: this._runtimeCapabilities,
			defaultTimeoutMs: this._defaultTimeoutMs,
		});
	}

	private async _initializeFallback(): Promise<void> {
		await this._fallbackAdapter.init();
		this._syncCapabilitiesFromFallback();
		this._usingFallbackAdapter = true;
		this._initialized = true;
	}

	private _syncCapabilitiesFromFallback(): void {
		const fallbackCapabilities = this._fallbackAdapter.capabilities;
		this.capabilities.joints = fallbackCapabilities.joints;
		this.capabilities.characterController =
			fallbackCapabilities.characterController;
		this.capabilities.shapeCast = fallbackCapabilities.shapeCast;
		this.capabilities.query = fallbackCapabilities.query;
		this.capabilities.syncInit = false;
	}

	private _enqueueCommand(command: RapierWorkerCommand): void {
		this._pendingCommands.push(command);
	}

	private _drainPendingCommands(): RapierWorkerCommand[] {
		if (this._pendingCommands.length === 0) return [];
		const commands = this._pendingCommands;
		this._pendingCommands = [];
		return commands;
	}

	private _assertInitialized(): void {
		if (this._initialized) return;
		throw new Error(
			"RapierWorkerPhysicsAdapter is not initialized. Call init() before using it."
		);
	}

	private async _dispatchToWorker<TResult>(
		payload: RapierWorkerTaskPayload
	): Promise<TResult> {
		return this._scheduler.schedule<TResult, RapierWorkerTaskPayload>(
			this._poolId,
			payload
		);
	}
}
