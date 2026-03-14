import { Node } from "../core/Node";
import type { IVector3 } from "../maths/types";
import {
	ParticleBlendMode,
	ParticleSpaceMode,
	type ParticleAtlas,
	type ParticleCollider,
	type ParticleEmitterParams,
	type ParticleGradientKey,
	type ParticleLODLevel,
	type ParticleLODSettings,
	type ParticleSubEmitterConfig,
	type ParticleSystemParams,
} from "./types";
import type { Texture } from "../core/Texture";
import type { RGBA } from "../utils/Color";

const DEFAULT_EMITTER: Required<
	Pick<
		ParticleEmitterParams,
		| "rate"
		| "lifetimeRange"
		| "speedRange"
		| "sizeRange"
		| "direction"
		| "spread"
		| "spawnRadius"
		| "startColor"
		| "rotationRange"
		| "angularVelocityRange"
	>
> & { bursts: [] } = {
	rate: 10,
	bursts: [],
	lifetimeRange: [0.5, 1.5],
	speedRange: [2, 5],
	sizeRange: [0.5, 1.0],
	direction: { x: 0, y: 1, z: 0 },
	spread: 0.35,
	spawnRadius: 0,
	startColor: { r: 255, g: 255, b: 255, a: 1 },
	rotationRange: [0, 0],
	angularVelocityRange: [0, 0],
};

const DEFAULT_GRAVITY: IVector3 = Object.freeze({
	x: 0,
	y: -9.8,
	z: 0,
});

const DEFAULT_LOD_LEVELS: ParticleLODLevel[] = [
	{
		distance: 24,
		projectedSize: 64,
		simulationIntervalFrames: 1,
		spawnScale: 1,
		maxParticlesScale: 1,
		renderSortRatio: 1,
	},
	{
		distance: 72,
		projectedSize: 24,
		simulationIntervalFrames: 2,
		spawnScale: 0.65,
		maxParticlesScale: 0.65,
		renderSortRatio: 0.75,
	},
	{
		distance: Number.POSITIVE_INFINITY,
		projectedSize: 0,
		simulationIntervalFrames: 4,
		spawnScale: 0.35,
		maxParticlesScale: 0.35,
		renderSortRatio: 0.5,
	},
];

const DEFAULT_LOD: ParticleLODSettings = {
	enabled: false,
	hysteresisFrames: 6,
	levels: DEFAULT_LOD_LEVELS,
};

export class ParticleSystem extends Node {
	public maxParticles: number;
	public seed: number;
	public space: ParticleSpaceMode;
	public blendMode: ParticleBlendMode;
	public texture: Texture | null;
	public atlas: ParticleAtlas | null;
	public gravity: IVector3;
	public emit: ParticleEmitterParams;
	public sizeOverLifetime: ParticleGradientKey<number>[];
	public colorOverLifetime: ParticleGradientKey<RGBA>[];
	public colliders: ParticleCollider[];
	public subEmitter: ParticleSubEmitterConfig | null;
	public receiveShadows: boolean;
	public lod: ParticleLODSettings;

	constructor(params: ParticleSystemParams = {}) {
		super({
			idPrefix: "particleSystem",
			name: params.name,
			visible: params.visible,
			position: params.position,
		});
		this.maxParticles = Math.max(1, params.maxParticles ?? 2000);
		this.seed = Math.max(1, Math.floor(params.seed ?? 1337));
		this.space = params.space ?? ParticleSpaceMode.Local;
		this.blendMode = params.blendMode ?? ParticleBlendMode.Alpha;
		this.texture = params.texture ?? null;
		this.atlas = params.atlas ?? null;
		this.gravity = cloneVector(params.gravity ?? DEFAULT_GRAVITY);
		this.emit = {
			...DEFAULT_EMITTER,
			...(params.emit ?? {}),
			direction: cloneVector(
				params.emit?.direction ?? DEFAULT_EMITTER.direction
			),
			startColor: cloneColor(
				params.emit?.startColor ?? DEFAULT_EMITTER.startColor
			),
			lifetimeRange: cloneRange(
				params.emit?.lifetimeRange ?? DEFAULT_EMITTER.lifetimeRange
			),
			speedRange: cloneRange(
				params.emit?.speedRange ?? DEFAULT_EMITTER.speedRange
			),
			sizeRange: cloneRange(
				params.emit?.sizeRange ?? DEFAULT_EMITTER.sizeRange
			),
			rotationRange: cloneRange(
				params.emit?.rotationRange ?? DEFAULT_EMITTER.rotationRange
			),
			angularVelocityRange: cloneRange(
				params.emit?.angularVelocityRange ??
					DEFAULT_EMITTER.angularVelocityRange
			),
			bursts: (params.emit?.bursts ?? []).map((burst) => ({ ...burst })),
		};
		this.sizeOverLifetime = (params.sizeOverLifetime ?? []).map((key) => ({
			t: key.t,
			value: key.value,
		}));
		this.colorOverLifetime = (params.colorOverLifetime ?? []).map((key) => ({
			t: key.t,
			value: cloneColor(key.value),
		}));
		this.colliders = (params.colliders ?? []).map((collider) =>
			cloneCollider(collider)
		);
		this.subEmitter = params.subEmitter ? { ...params.subEmitter } : null;
		this.receiveShadows = params.receiveShadows ?? true;
		this.lod = cloneLOD(params.lod ?? DEFAULT_LOD);
	}

	protected override _createCloneInstance(): this {
		return new ParticleSystem() as this;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.maxParticles = this.maxParticles;
		target.seed = this.seed;
		target.space = this.space;
		target.blendMode = this.blendMode;
		target.texture = this.texture;
		target.atlas = this.atlas ? { ...this.atlas } : null;
		target.gravity = cloneVector(this.gravity);
		target.emit = {
			...this.emit,
			direction: cloneVector(this.emit.direction ?? DEFAULT_EMITTER.direction),
			startColor: cloneColor(this.emit.startColor ?? DEFAULT_EMITTER.startColor),
			lifetimeRange: cloneRange(
				this.emit.lifetimeRange ?? DEFAULT_EMITTER.lifetimeRange
			),
			speedRange: cloneRange(
				this.emit.speedRange ?? DEFAULT_EMITTER.speedRange
			),
			sizeRange: cloneRange(this.emit.sizeRange ?? DEFAULT_EMITTER.sizeRange),
			rotationRange: cloneRange(
				this.emit.rotationRange ?? DEFAULT_EMITTER.rotationRange
			),
			angularVelocityRange: cloneRange(
				this.emit.angularVelocityRange ??
					DEFAULT_EMITTER.angularVelocityRange
			),
			bursts: (this.emit.bursts ?? []).map((burst) => ({ ...burst })),
		};
		target.sizeOverLifetime = this.sizeOverLifetime.map((key) => ({
			t: key.t,
			value: key.value,
		}));
		target.colorOverLifetime = this.colorOverLifetime.map((key) => ({
			t: key.t,
			value: cloneColor(key.value),
		}));
		target.colliders = this.colliders.map((collider) => cloneCollider(collider));
		target.subEmitter = this.subEmitter ? { ...this.subEmitter } : null;
		target.receiveShadows = this.receiveShadows;
		target.lod = cloneLOD(this.lod);
	}
}

function cloneVector(source: IVector3): IVector3 {
	return {
		x: source.x,
		y: source.y,
		z: source.z,
	};
}

function cloneColor(source: RGBA): RGBA {
	return {
		r: source.r,
		g: source.g,
		b: source.b,
		a: source.a,
	};
}

function cloneRange(source: [number, number]): [number, number] {
	return [source[0], source[1]];
}

function cloneCollider(collider: ParticleCollider): ParticleCollider {
	if (collider.type === "plane") {
		return {
			...collider,
			normal: cloneVector(collider.normal),
		};
	}

	if (collider.type === "sphere") {
		return {
			...collider,
			center: cloneVector(collider.center),
		};
	}

	return {
		...collider,
		min: cloneVector(collider.min),
		max: cloneVector(collider.max),
	};
}

function cloneLOD(source: ParticleLODSettings): ParticleLODSettings {
	return {
		enabled: source.enabled ?? false,
		hysteresisFrames: source.hysteresisFrames ?? 6,
		levels: (source.levels ?? []).map((level) => ({ ...level })),
	};
}
