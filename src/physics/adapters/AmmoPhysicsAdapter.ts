import type {
	IPhysicsEngineAdapter,
	PhysicsAdapterCapabilities,
	PhysicsAdapterStepResult,
} from "../IPhysicsEngineAdapter";
import type { IVector3 } from "../../maths/types";
import type {
	CharacterControllerDescriptor,
	CharacterMoveResult,
	ColliderDescriptor,
	ColliderShape,
	JointDescriptor,
	PhysicsTransform,
	PhysicsWorldConfig,
	RigidBodyDescriptor,
} from "../types";
import { SimplePhysicsAdapter } from "./SimplePhysicsAdapter";

export interface AmmoPhysicsAdapterOptions {
	moduleLoader?: () => Promise<unknown>;
	strict?: boolean;
	fallbackAdapter?: IPhysicsEngineAdapter;
}

export class AmmoPhysicsAdapter implements IPhysicsEngineAdapter {
	public readonly id = "ammo";
	public readonly capabilities: PhysicsAdapterCapabilities;

	private _moduleLoader: () => Promise<unknown>;
	private _strict: boolean;
	private _delegate: IPhysicsEngineAdapter;
	private _loadedModule: unknown = null;

	constructor(options: AmmoPhysicsAdapterOptions = {}) {
		this._moduleLoader =
			options.moduleLoader ?? (async () => loadOptionalModule("ammo.js"));
		this._strict = options.strict ?? true;
		this._delegate =
			options.fallbackAdapter ?? new SimplePhysicsAdapter("ammo-fallback");
		this.capabilities = {
			...this._delegate.capabilities,
			syncInit: false,
		};
	}

	public async init(): Promise<void> {
		try {
			this._loadedModule = await this._moduleLoader();
		} catch (error) {
			if (this._strict) {
				throw new Error(
					`AmmoPhysicsAdapter failed to load "ammo.js". Install it as an optional peer dependency or provide moduleLoader. Inner error: ${String(error)}`
				);
			}
		}
		await this._delegate.init();
	}

	public initSync(): void {
		throw new Error(
			"AmmoPhysicsAdapter.initSync is not supported because Ammo usually requires asynchronous module loading. Use init() instead."
		);
	}

	public hasWorld(worldId: string): boolean {
		return this._delegate.hasWorld(worldId);
	}

	public createWorld(config: PhysicsWorldConfig): void {
		this._delegate.createWorld(config);
	}

	public destroyWorld(worldId: string): void {
		this._delegate.destroyWorld(worldId);
	}

	public createBody(
		worldId: string,
		bodyId: string,
		descriptor: RigidBodyDescriptor,
		initialTransform: PhysicsTransform
	): void {
		this._delegate.createBody(worldId, bodyId, descriptor, initialTransform);
	}

	public destroyBody(worldId: string, bodyId: string): void {
		this._delegate.destroyBody(worldId, bodyId);
	}

	public setBodyTransform(
		worldId: string,
		bodyId: string,
		transform: PhysicsTransform
	): void {
		this._delegate.setBodyTransform(worldId, bodyId, transform);
	}

	public setBodyLinearVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void {
		this._delegate.setBodyLinearVelocity(worldId, bodyId, velocity);
	}

	public addCollider(
		worldId: string,
		bodyId: string,
		colliderId: string,
		descriptor: ColliderDescriptor,
		shape: ColliderShape
	): void {
		this._delegate.addCollider(worldId, bodyId, colliderId, descriptor, shape);
	}

	public destroyCollider(worldId: string, colliderId: string): void {
		this._delegate.destroyCollider(worldId, colliderId);
	}

	public createJoint(
		worldId: string,
		jointId: string,
		descriptor: JointDescriptor
	): void {
		this._delegate.createJoint(worldId, jointId, descriptor);
	}

	public destroyJoint(worldId: string, jointId: string): void {
		this._delegate.destroyJoint(worldId, jointId);
	}

	public createCharacterController(
		worldId: string,
		controllerId: string,
		descriptor: CharacterControllerDescriptor
	): void {
		this._delegate.createCharacterController(worldId, controllerId, descriptor);
	}

	public destroyCharacterController(
		worldId: string,
		controllerId: string
	): void {
		this._delegate.destroyCharacterController(worldId, controllerId);
	}

	public moveCharacterController(
		worldId: string,
		controllerId: string,
		direction: IVector3,
		deltaSeconds: number
	): CharacterMoveResult {
		return this._delegate.moveCharacterController(
			worldId,
			controllerId,
			direction,
			deltaSeconds
		);
	}

	public jumpCharacterController(
		worldId: string,
		controllerId: string,
		speed: number
	): void {
		this._delegate.jumpCharacterController(worldId, controllerId, speed);
	}

	public isCharacterControllerGrounded(
		worldId: string,
		controllerId: string
	): boolean {
		return this._delegate.isCharacterControllerGrounded(worldId, controllerId);
	}

	public setCharacterControllerMaxSlope(
		worldId: string,
		controllerId: string,
		value: number
	): void {
		this._delegate.setCharacterControllerMaxSlope(worldId, controllerId, value);
	}

	public setCharacterControllerStepHeight(
		worldId: string,
		controllerId: string,
		value: number
	): void {
		this._delegate.setCharacterControllerStepHeight(
			worldId,
			controllerId,
			value
		);
	}

	public stepWorld(
		worldId: string,
		deltaSeconds: number
	): PhysicsAdapterStepResult {
		return this._delegate.stepWorld(worldId, deltaSeconds);
	}
}

function loadOptionalModule(moduleName: string): Promise<unknown> {
	const importer = new Function("m", "return import(m)") as (
		modulePath: string
	) => Promise<unknown>;
	return importer(moduleName);
}
