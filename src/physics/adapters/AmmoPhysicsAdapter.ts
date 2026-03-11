import type {
	IPhysicsEngineAdapter,
	PhysicsAdapterBodyState,
	PhysicsAdapterCapabilities,
	PhysicsAdapterStepResult,
} from "../IPhysicsEngineAdapter";
import type { IVector3 } from "../../maths/types";
import { DEFAULT_GRAVITY } from "../constants";
import type {
	CharacterControllerDescriptor,
	CharacterMoveResult,
	PhysicsBoxCastQuery,
	PhysicsOverlapBoxQuery,
	PhysicsOverlapHit,
	PhysicsOverlapSphereQuery,
	PhysicsQueryHit,
	PhysicsRaycastQuery,
	PhysicsSphereCastQuery,
	ColliderDescriptor,
	ColliderShape,
	JointDescriptor,
	PhysicsTransform,
	PhysicsWorldConfig,
	RigidBodyDescriptor,
	RigidBodyType,
} from "../types";
import { SimplePhysicsAdapter } from "./SimplePhysicsAdapter";

interface AmmoModuleLike {
	btDefaultCollisionConfiguration?: new () => unknown;
	btCollisionDispatcher?: new (config: unknown) => unknown;
	btDbvtBroadphase?: new () => unknown;
	btSequentialImpulseConstraintSolver?: new () => unknown;
	btDiscreteDynamicsWorld?: new (
		dispatcher: unknown,
		broadphase: unknown,
		solver: unknown,
		collisionConfiguration: unknown
	) => unknown;
	btVector3?: new (x: number, y: number, z: number) => unknown;
	btQuaternion?: new (x: number, y: number, z: number, w: number) => unknown;
	btTransform?: new () => unknown;
	btDefaultMotionState?: new (transform: unknown) => unknown;
	btRigidBodyConstructionInfo?: new (
		mass: number,
		motionState: unknown,
		shape: unknown,
		localInertia: unknown
	) => unknown;
	btRigidBody?: new (info: unknown) => unknown;
	btSphereShape?: new (radius: number) => unknown;
	btBoxShape?: new (halfExtents: unknown) => unknown;
	destroy?: (target: unknown) => void;
}

interface AmmoBodyState {
	id: string;
	type: RigidBodyType;
	rigidBody: any;
	motionState: any;
	constructionInfo: any;
	shape: any;
	transform: PhysicsTransform;
	ccd: boolean;
}

interface AmmoWorldState {
	world: any;
	collisionConfig: any;
	dispatcher: any;
	broadphase: any;
	solver: any;
	bodies: Map<string, AmmoBodyState>;
}

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
	private _ammo: AmmoModuleLike | null = null;
	private _worlds = new Map<string, AmmoWorldState>();

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
			await this._delegate.init();
			return;
		}

		const ammo = await resolveAmmoModule(this._loadedModule);
		if (!ammo || !isAmmoUsable(ammo)) {
			if (this._strict) {
				throw new Error(
					"AmmoPhysicsAdapter loaded module is not a usable Ammo runtime. Pass strict: false to fallback."
				);
			}
			await this._delegate.init();
			return;
		}

		this._ammo = ammo;
		await this._delegate.init();
	}

	public initSync(): void {
		throw new Error(
			"AmmoPhysicsAdapter.initSync is not supported because Ammo usually requires asynchronous module loading. Use init() instead."
		);
	}

	public hasWorld(worldId: string): boolean {
		if (this._worlds.has(worldId)) return true;
		return this._delegate.hasWorld(worldId);
	}

	public createWorld(config: PhysicsWorldConfig): void {
		this._delegate.createWorld(config);
		if (this._usingFallback()) return;
		if (this._worlds.has(config.worldId)) {
			throw new Error(`Physics world "${config.worldId}" already exists`);
		}

		try {
			const collisionConfig = this._newAmmo("btDefaultCollisionConfiguration");
			const dispatcher = this._newAmmo(
				"btCollisionDispatcher",
				collisionConfig
			);
			const broadphase = this._newAmmo("btDbvtBroadphase");
			const solver = this._newAmmo("btSequentialImpulseConstraintSolver");
			const world = this._newAmmo(
				"btDiscreteDynamicsWorld",
				dispatcher,
				broadphase,
				solver,
				collisionConfig
			);
			this._setWorldGravity(world, config.gravity ?? DEFAULT_GRAVITY);

			this._worlds.set(config.worldId, {
				world,
				collisionConfig,
				dispatcher,
				broadphase,
				solver,
				bodies: new Map(),
			});
		} catch (error) {
			this._delegate.destroyWorld(config.worldId);
			throw new Error(
				`AmmoPhysicsAdapter failed to create world "${config.worldId}". Inner error: ${String(error)}`
			);
		}
	}

	public destroyWorld(worldId: string): void {
		if (!this._usingFallback()) {
			const world = this._worlds.get(worldId);
			if (world) {
				for (const bodyId of Array.from(world.bodies.keys())) {
					this._destroyAmmoBody(world, bodyId);
				}
				this._destroyAmmoObject(world.world);
				this._destroyAmmoObject(world.solver);
				this._destroyAmmoObject(world.broadphase);
				this._destroyAmmoObject(world.dispatcher);
				this._destroyAmmoObject(world.collisionConfig);
				this._worlds.delete(worldId);
			}
		}
		this._delegate.destroyWorld(worldId);
	}

	public createBody(
		worldId: string,
		bodyId: string,
		descriptor: RigidBodyDescriptor,
		initialTransform: PhysicsTransform
	): void {
		this._delegate.createBody(worldId, bodyId, descriptor, initialTransform);
		if (this._usingFallback()) return;

		const world = this._requireWorld(worldId);
		if (world.bodies.has(bodyId)) {
			throw new Error(
				`Physics body "${bodyId}" already exists in "${worldId}"`
			);
		}

		try {
			const type = descriptor.type ?? "dynamic";
			const mass = resolveBodyMass(type, descriptor.mass);
			const shape = this._createFallbackShape();
			const transform = this._createAmmoTransform(initialTransform);
			const motionState = this._newAmmo("btDefaultMotionState", transform);
			const localInertia = this._createAmmoVector3({ x: 0, y: 0, z: 0 });
			if (mass > 0) {
				this._invoke(shape, ["calculateLocalInertia"], [[mass, localInertia]]);
			}

			const constructionInfo = this._newAmmo(
				"btRigidBodyConstructionInfo",
				mass,
				motionState,
				shape,
				localInertia
			);
			const rigidBody = this._newAmmo("btRigidBody", constructionInfo);
			const ccd = descriptor.ccd ?? false;

			this._applyBodyFlags(rigidBody, type);
			if (ccd) this._applyBodyCcd(rigidBody, 0.5);
			if (descriptor.linearVelocity) {
				this._setVector3(
					rigidBody,
					["setLinearVelocity"],
					descriptor.linearVelocity
				);
			}
			if (descriptor.angularVelocity) {
				this._setVector3(
					rigidBody,
					["setAngularVelocity"],
					descriptor.angularVelocity
				);
			}
			this._invoke(world.world, ["addRigidBody"], [[rigidBody]]);
			world.bodies.set(bodyId, {
				id: bodyId,
				type,
				rigidBody,
				motionState,
				constructionInfo,
				shape,
				transform: cloneTransform(initialTransform),
				ccd,
			});

			this._destroyAmmoObject(localInertia);
			this._destroyAmmoObject(transform);
		} catch (error) {
			this._delegate.destroyBody(worldId, bodyId);
			throw new Error(
				`AmmoPhysicsAdapter failed to create body "${bodyId}" in "${worldId}". Inner error: ${String(error)}`
			);
		}
	}

	public destroyBody(worldId: string, bodyId: string): void {
		if (!this._usingFallback()) {
			const world = this._worlds.get(worldId);
			if (world) this._destroyAmmoBody(world, bodyId);
		}
		this._delegate.destroyBody(worldId, bodyId);
	}

	public setBodyTransform(
		worldId: string,
		bodyId: string,
		transform: PhysicsTransform
	): void {
		this._delegate.setBodyTransform(worldId, bodyId, transform);
		if (this._usingFallback()) return;
		const world = this._worlds.get(worldId);
		const body = world?.bodies.get(bodyId);
		if (!body) return;
		this._setBodyTransform(body, transform, true);
	}

	public setBodyLinearVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void {
		this._delegate.setBodyLinearVelocity(worldId, bodyId, velocity);
		if (this._usingFallback()) return;
		const world = this._worlds.get(worldId);
		const body = world?.bodies.get(bodyId);
		if (!body) return;
		this._setVector3(body.rigidBody, ["setLinearVelocity"], velocity);
		this._invoke(body.rigidBody, ["activate"], [[true], [1], []]);
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

	public raycast(
		worldId: string,
		query: PhysicsRaycastQuery
	): PhysicsQueryHit | null {
		return this._delegate.raycast(worldId, query);
	}

	public sphereCast(
		worldId: string,
		query: PhysicsSphereCastQuery
	): PhysicsQueryHit | null {
		return this._delegate.sphereCast(worldId, query);
	}

	public boxCast(
		worldId: string,
		query: PhysicsBoxCastQuery
	): PhysicsQueryHit | null {
		return this._delegate.boxCast(worldId, query);
	}

	public overlapSphere(
		worldId: string,
		query: PhysicsOverlapSphereQuery
	): PhysicsOverlapHit[] {
		return this._delegate.overlapSphere(worldId, query);
	}

	public overlapBox(
		worldId: string,
		query: PhysicsOverlapBoxQuery
	): PhysicsOverlapHit[] {
		return this._delegate.overlapBox(worldId, query);
	}

	public stepWorld(
		worldId: string,
		deltaSeconds: number
	): PhysicsAdapterStepResult {
		if (this._usingFallback()) {
			return this._delegate.stepWorld(worldId, deltaSeconds);
		}
		const world = this._worlds.get(worldId);
		if (!world) return this._delegate.stepWorld(worldId, deltaSeconds);

		const dt = Math.max(0, deltaSeconds);
		if (dt > 0) {
			this._invokeOrThrow(
				world.world,
				["stepSimulation"],
				[[dt, 1, dt], [dt, 1], [dt]],
				`Ammo world "${worldId}" does not expose stepSimulation()`
			);
		}

		const bodyStates: PhysicsAdapterBodyState[] = [];
		for (const body of world.bodies.values()) {
			const transform = this._readBodyTransform(body);
			const sleeping = this._readBodySleeping(body);
			bodyStates.push({
				bodyId: body.id,
				transform,
				sleeping,
				ccd: body.ccd,
			});
			this._delegate.setBodyTransform(worldId, body.id, transform);
		}

		const delegateResult = this._delegate.stepWorld(worldId, 0);
		const seen = new Set(bodyStates.map((item) => item.bodyId));
		for (const state of delegateResult.bodyStates) {
			if (seen.has(state.bodyId)) continue;
			bodyStates.push(state);
		}

		let sleepingBodies = 0;
		let ccdBodies = 0;
		for (const state of bodyStates) {
			if (state.sleeping) sleepingBodies++;
			if (state.ccd) ccdBodies++;
		}

		return {
			bodyStates,
			events: delegateResult.events,
			activeBodies: bodyStates.length - sleepingBodies,
			sleepingBodies,
			ccdBodies,
		};
	}

	private _destroyAmmoBody(world: AmmoWorldState, bodyId: string): void {
		const body = world.bodies.get(bodyId);
		if (!body) return;
		this._invoke(world.world, ["removeRigidBody"], [[body.rigidBody]]);
		this._destroyAmmoObject(body.rigidBody);
		this._destroyAmmoObject(body.motionState);
		this._destroyAmmoObject(body.constructionInfo);
		this._destroyAmmoObject(body.shape);
		world.bodies.delete(bodyId);
	}

	private _setBodyTransform(
		body: AmmoBodyState,
		transform: PhysicsTransform,
		wakeUp: boolean
	): void {
		const ammoTransform = this._createAmmoTransform(transform);
		this._invoke(body.rigidBody, ["setWorldTransform"], [[ammoTransform]]);
		this._invoke(body.motionState, ["setWorldTransform"], [[ammoTransform]]);
		if (wakeUp) this._invoke(body.rigidBody, ["activate"], [[true], [1], []]);
		body.transform = cloneTransform(transform);
		this._destroyAmmoObject(ammoTransform);
	}

	private _readBodyTransform(body: AmmoBodyState): PhysicsTransform {
		let rawTransform: any = null;
		let ownedTemp: any = null;
		const temp = this._newAmmo("btTransform");
		this._invoke(temp, ["setIdentity"], [[]]);
		if (this._invoke(body.motionState, ["getWorldTransform"], [[temp]])) {
			rawTransform = temp;
			ownedTemp = temp;
		} else {
			this._destroyAmmoObject(temp);
		}

		if (!rawTransform) {
			const direct = this._readFromGetter(body.rigidBody, "getWorldTransform");
			if (direct && typeof direct === "object") rawTransform = direct;
		}
		if (!rawTransform) return cloneTransform(body.transform);

		const origin = this._readFromGetter(rawTransform, "getOrigin");
		const rotation = this._readFromGetter(rawTransform, "getRotation");
		const transform: PhysicsTransform = {
			position: readAmmoVector3(origin),
			rotation: toQuaternionTuple(rotation),
		};
		body.transform = cloneTransform(transform);
		if (ownedTemp) this._destroyAmmoObject(ownedTemp);
		return transform;
	}

	private _readBodySleeping(body: AmmoBodyState): boolean {
		if (body.type !== "dynamic") return false;
		const isActive = this._readFromGetter(body.rigidBody, "isActive");
		if (typeof isActive === "boolean") return !isActive;
		const isSleeping = this._readFromGetter(body.rigidBody, "isSleeping");
		if (typeof isSleeping === "boolean") return isSleeping;
		const velocity = readAmmoVector3(
			this._readFromGetter(body.rigidBody, "getLinearVelocity")
		);
		const speed = Math.hypot(velocity.x, velocity.y, velocity.z);
		return speed < 0.001;
	}

	private _applyBodyFlags(rigidBody: any, type: RigidBodyType): void {
		const CF_KINEMATIC_OBJECT = 2;
		const CF_NO_CONTACT_RESPONSE = 4;
		const DISABLE_DEACTIVATION = 4;
		const rawFlags = this._readFromGetter(rigidBody, "getCollisionFlags");
		let flags = Number.isFinite(rawFlags as number) ? Number(rawFlags) : 0;
		flags |= CF_NO_CONTACT_RESPONSE;
		if (type === "kinematic") {
			flags |= CF_KINEMATIC_OBJECT;
			this._invoke(rigidBody, ["setActivationState"], [[DISABLE_DEACTIVATION]]);
		}
		this._invoke(rigidBody, ["setCollisionFlags"], [[flags]]);
	}

	private _applyBodyCcd(rigidBody: any, radius: number): void {
		this._invoke(rigidBody, ["setCcdMotionThreshold"], [[1e-7]]);
		this._invoke(
			rigidBody,
			["setCcdSweptSphereRadius"],
			[[Math.max(0.01, radius)]]
		);
	}

	private _setWorldGravity(world: any, gravity: IVector3): void {
		const g = this._createAmmoVector3(gravity);
		this._invoke(world, ["setGravity"], [[g], [gravity]]);
		this._destroyAmmoObject(g);
	}

	private _createFallbackShape(): any {
		const ammo = this._requireAmmo();
		if (typeof ammo.btSphereShape === "function") {
			return new ammo.btSphereShape(0.5);
		}
		const halfExtents = this._createAmmoVector3({ x: 0.5, y: 0.5, z: 0.5 });
		const shape = this._newAmmo("btBoxShape", halfExtents);
		this._destroyAmmoObject(halfExtents);
		return shape;
	}

	private _createAmmoVector3(value: IVector3): any {
		const Ctor = this._requireAmmoCtor("btVector3");
		return new Ctor(value.x, value.y, value.z);
	}

	private _createAmmoQuaternion(value: [number, number, number, number]): any {
		const ammo = this._requireAmmo();
		if (typeof ammo.btQuaternion === "function") {
			return new ammo.btQuaternion(value[0], value[1], value[2], value[3]);
		}
		return { x: value[0], y: value[1], z: value[2], w: value[3] };
	}

	private _createAmmoTransform(transform: PhysicsTransform): any {
		const ammoTransform = this._newAmmo("btTransform");
		this._invoke(ammoTransform, ["setIdentity"], [[]]);
		const origin = this._createAmmoVector3(transform.position);
		const rotation = this._createAmmoQuaternion(transform.rotation);
		this._invoke(ammoTransform, ["setOrigin"], [[origin]]);
		this._invoke(ammoTransform, ["setRotation"], [[rotation]]);
		this._destroyAmmoObject(origin);
		this._destroyAmmoObject(rotation);
		return ammoTransform;
	}

	private _setVector3(target: any, methods: string[], value: IVector3): void {
		const vector = this._createAmmoVector3(value);
		this._invoke(target, methods, [
			[vector],
			[value.x, value.y, value.z],
			[value],
		]);
		this._destroyAmmoObject(vector);
	}

	private _newAmmo(name: string, ...args: unknown[]): any {
		const Ctor = this._requireAmmoCtor(name);
		return new Ctor(...args);
	}

	private _requireAmmoCtor(name: string): new (...args: unknown[]) => any {
		const ammo = this._requireAmmo();
		const member = (ammo as Record<string, unknown>)[name];
		if (typeof member !== "function") {
			throw new Error(`Loaded Ammo module is missing constructor ${name}`);
		}
		return member as new (...args: unknown[]) => any;
	}

	private _destroyAmmoObject(target: unknown): void {
		if (!target) return;
		const destroy = this._ammo?.destroy;
		if (typeof destroy !== "function") return;
		try {
			destroy(target);
		} catch {}
	}

	private _invoke(
		target: any,
		methodNames: string[],
		argVariants: unknown[][]
	): boolean {
		if (!target || typeof target !== "object") return false;
		for (const methodName of methodNames) {
			const fn = (target as Record<string, unknown>)[methodName];
			if (typeof fn !== "function") continue;
			for (const args of argVariants) {
				try {
					(fn as (...callArgs: unknown[]) => unknown).apply(target, args);
					return true;
				} catch {}
			}
		}
		return false;
	}

	private _invokeOrThrow(
		target: any,
		methodNames: string[],
		argVariants: unknown[][],
		errorMessage: string
	): void {
		if (this._invoke(target, methodNames, argVariants)) return;
		throw new Error(errorMessage);
	}

	private _readFromGetter(target: any, getterName: string): unknown {
		if (!target || typeof target !== "object") return undefined;
		const member = (target as Record<string, unknown>)[getterName];
		if (typeof member === "function") {
			try {
				return (member as () => unknown).call(target);
			} catch {
				return undefined;
			}
		}
		return member;
	}

	private _usingFallback(): boolean {
		return this._ammo === null;
	}

	private _requireAmmo(): AmmoModuleLike {
		if (this._ammo) return this._ammo;
		throw new Error(
			"AmmoPhysicsAdapter is not initialized with a usable Ammo module"
		);
	}

	private _requireWorld(worldId: string): AmmoWorldState {
		const world = this._worlds.get(worldId);
		if (world) return world;
		throw new Error(`Physics world "${worldId}" does not exist`);
	}
}

async function resolveAmmoModule(
	loaded: unknown
): Promise<AmmoModuleLike | null> {
	for (const candidate of collectAmmoCandidates(loaded)) {
		const value = await materializeAmmoCandidate(candidate);
		if (value && isAmmoUsable(value)) return value;
	}
	for (const candidate of collectAmmoCandidates(loaded)) {
		const value = await materializeAmmoCandidate(candidate);
		if (value) return value;
	}
	return null;
}

function collectAmmoCandidates(loaded: unknown): unknown[] {
	const queue: unknown[] = [loaded];
	if (loaded && typeof loaded === "object") {
		const source = loaded as { default?: unknown; Ammo?: unknown };
		queue.push(source.default);
		queue.push(source.Ammo);
	}
	return queue;
}

async function materializeAmmoCandidate(
	candidate: unknown
): Promise<AmmoModuleLike | null> {
	if (!candidate) return null;
	if (typeof candidate === "function") {
		try {
			const resolved = await Promise.resolve(
				(candidate as () => Promise<unknown> | unknown)()
			);
			if (resolved && typeof resolved === "object") {
				return resolved as AmmoModuleLike;
			}
		} catch {}
		return null;
	}
	return typeof candidate === "object" ? (candidate as AmmoModuleLike) : null;
}

function isAmmoUsable(module: AmmoModuleLike): boolean {
	if (!module || typeof module !== "object") return false;
	return (
		typeof module.btDefaultCollisionConfiguration === "function" &&
		typeof module.btCollisionDispatcher === "function" &&
		typeof module.btDbvtBroadphase === "function" &&
		typeof module.btSequentialImpulseConstraintSolver === "function" &&
		typeof module.btDiscreteDynamicsWorld === "function" &&
		typeof module.btVector3 === "function" &&
		typeof module.btTransform === "function" &&
		typeof module.btDefaultMotionState === "function" &&
		typeof module.btRigidBodyConstructionInfo === "function" &&
		typeof module.btRigidBody === "function" &&
		(typeof module.btSphereShape === "function" ||
			typeof module.btBoxShape === "function")
	);
}

function resolveBodyMass(type: RigidBodyType, mass?: number): number {
	if (type !== "dynamic") return 0;
	if (Number.isFinite(mass) && Number(mass) > 0) return Number(mass);
	return 1;
}

function cloneVector(source: IVector3): IVector3 {
	return { x: source.x, y: source.y, z: source.z };
}

function cloneTransform(transform: PhysicsTransform): PhysicsTransform {
	return {
		position: cloneVector(transform.position),
		rotation: [
			transform.rotation[0],
			transform.rotation[1],
			transform.rotation[2],
			transform.rotation[3],
		],
	};
}

function readAmmoComponent(
	value: unknown,
	key: string,
	fallback: number
): number {
	if (!value || typeof value !== "object") return fallback;
	const source = value as Record<string, unknown>;
	const direct = source[key];
	if (typeof direct === "number" && Number.isFinite(direct)) return direct;
	if (typeof direct === "function") {
		try {
			const result = (direct as () => unknown).call(value);
			if (typeof result === "number" && Number.isFinite(result)) return result;
		} catch {}
	}
	const getter = source[`get${key.toUpperCase()}`];
	if (typeof getter === "function") {
		try {
			const result = (getter as () => unknown).call(value);
			if (typeof result === "number" && Number.isFinite(result)) return result;
		} catch {}
	}
	return fallback;
}

function readAmmoVector3(value: unknown): IVector3 {
	return {
		x: readAmmoComponent(value, "x", 0),
		y: readAmmoComponent(value, "y", 0),
		z: readAmmoComponent(value, "z", 0),
	};
}

function toQuaternionTuple(value: unknown): [number, number, number, number] {
	return [
		readAmmoComponent(value, "x", 0),
		readAmmoComponent(value, "y", 0),
		readAmmoComponent(value, "z", 0),
		readAmmoComponent(value, "w", 1),
	];
}

function loadOptionalModule(moduleName: string): Promise<unknown> {
	const importer = new Function("m", "return import(m)") as (
		modulePath: string
	) => Promise<unknown>;
	return importer(moduleName);
}
