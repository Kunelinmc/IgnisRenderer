import { Node } from "../core/Node";
import type { IVector3 } from "../maths/types";
import {
	ParticleBlendMode,
	ParticleSpaceMode,
	type ParticleAtlas,
	type ParticleCollider,
	type ParticleDefinition,
	type ParticleEmitterParams,
	type ParticleGradientKey,
	type ParticleLODLevel,
	type ParticleLODSettings,
	type ParticleSubEmitterConfig,
	type ParticleSystemParams,
} from "./types";
import type { Texture } from "../core/Texture";
import type { RGBA } from "../foundation/Color";

const DEFAULT_EMITTER = {
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
} satisfies {
	rate: number;
	bursts: [];
	lifetimeRange: [number, number];
	speedRange: [number, number];
	sizeRange: [number, number];
	direction: IVector3;
	spread: number;
	spawnRadius: number;
	startColor: RGBA;
	rotationRange: [number, number];
	angularVelocityRange: [number, number];
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
	public gravity: IVector3;
	public emit: ParticleEmitterParams;
	public definitions: ParticleDefinition[];
	public colliders: ParticleCollider[];
	public subEmitter: ParticleSubEmitterConfig | null;
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
		this.gravity = cloneVector(params.gravity ?? DEFAULT_GRAVITY);
		this.definitions = normalizeParticleDefinitions(params);
		this.emit = createEmitterParams(
			params.emit,
			() => this._primaryDefinition
		);
		this.colliders = (params.colliders ?? []).map((collider) =>
			cloneCollider(collider)
		);
		this.subEmitter = params.subEmitter ? { ...params.subEmitter } : null;
		this.lod = cloneLOD(params.lod ?? DEFAULT_LOD);
	}

	public get blendMode(): ParticleBlendMode {
		return resolveDefinitionBlendMode(this._primaryDefinition);
	}

	public set blendMode(value: ParticleBlendMode) {
		const definition = this._primaryDefinition;
		if (definition.shape.kind === "billboard") {
			definition.shape.blendMode = value;
		}
	}

	public get texture(): Texture | null {
		const shape = this._primaryDefinition.shape;
		return shape.kind === "billboard" ? shape.texture ?? null : null;
	}

	public set texture(value: Texture | null) {
		const definition = this._primaryDefinition;
		if (definition.shape.kind === "billboard") {
			definition.shape.texture = value;
		}
	}

	public get atlas(): ParticleAtlas | null {
		const shape = this._primaryDefinition.shape;
		return shape.kind === "billboard" ? shape.atlas ?? null : null;
	}

	public set atlas(value: ParticleAtlas | null) {
		const definition = this._primaryDefinition;
		if (definition.shape.kind === "billboard") {
			definition.shape.atlas = value;
		}
	}

	public get sizeOverLifetime(): ParticleGradientKey<number>[] {
		return this._primaryDefinition.sizeOverLifetime ?? [];
	}

	public set sizeOverLifetime(value: ParticleGradientKey<number>[]) {
		this._primaryDefinition.sizeOverLifetime = cloneNumberGradient(value);
	}

	public get colorOverLifetime(): ParticleGradientKey<RGBA>[] {
		return this._primaryDefinition.colorOverLifetime ?? [];
	}

	public set colorOverLifetime(value: ParticleGradientKey<RGBA>[]) {
		this._primaryDefinition.colorOverLifetime = cloneColorGradient(value);
	}

	public get receiveShadows(): boolean {
		return this._primaryDefinition.receiveShadows ?? true;
	}

	public set receiveShadows(value: boolean) {
		this._primaryDefinition.receiveShadows = value;
	}

	public get castShadows(): boolean {
		return this._primaryDefinition.castShadows ?? true;
	}

	public set castShadows(value: boolean) {
		this._primaryDefinition.castShadows = value;
	}

	public get shadowDensity(): number {
		return this._primaryDefinition.shadowDensity ?? 1;
	}

	public set shadowDensity(value: number) {
		this._primaryDefinition.shadowDensity = resolveNonNegativeFinite(value, 1);
	}

	public get shadowSoftness(): number {
		return this._primaryDefinition.shadowSoftness ?? 1;
	}

	public set shadowSoftness(value: number) {
		this._primaryDefinition.shadowSoftness = resolveNonNegativeFinite(value, 1);
	}

	protected override _createCloneInstance(): this {
		return new ParticleSystem() as this;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.maxParticles = this.maxParticles;
		target.seed = this.seed;
		target.space = this.space;
		target.gravity = cloneVector(this.gravity);
		target.definitions = this.definitions.map((definition) =>
			cloneParticleDefinition(definition)
		);
		target.emit = createEmitterParams(
			this.emit,
			() => target._primaryDefinition
		);
		target.colliders = this.colliders.map((collider) =>
			cloneCollider(collider)
		);
		target.subEmitter = this.subEmitter ? { ...this.subEmitter } : null;
		target.lod = cloneLOD(this.lod);
	}

	private get _primaryDefinition(): ParticleDefinition {
		if (this.definitions.length === 0) {
			this.definitions.push(createDefaultParticleDefinition({}));
		}
		return this.definitions[0];
	}
}

function resolveNonNegativeFinite(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.max(0, value);
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

function createEmitterParams(
	source: ParticleEmitterParams | undefined,
	resolveDefinition: () => ParticleDefinition
): ParticleEmitterParams {
	const emit: ParticleEmitterParams = {
		rate: source?.rate ?? DEFAULT_EMITTER.rate,
		spread: source?.spread ?? DEFAULT_EMITTER.spread,
		spawnRadius: source?.spawnRadius ?? DEFAULT_EMITTER.spawnRadius,
		direction: cloneVector(source?.direction ?? DEFAULT_EMITTER.direction),
		bursts: (source?.bursts ?? []).map((burst) => ({ ...burst })),
	};
	defineRangeAlias(
		emit,
		"lifetimeRange",
		resolveDefinition,
		DEFAULT_EMITTER.lifetimeRange
	);
	defineRangeAlias(
		emit,
		"speedRange",
		resolveDefinition,
		DEFAULT_EMITTER.speedRange
	);
	defineRangeAlias(
		emit,
		"sizeRange",
		resolveDefinition,
		DEFAULT_EMITTER.sizeRange
	);
	defineColorAlias(
		emit,
		"startColor",
		resolveDefinition,
		DEFAULT_EMITTER.startColor
	);
	defineRangeAlias(
		emit,
		"rotationRange",
		resolveDefinition,
		DEFAULT_EMITTER.rotationRange
	);
	defineRangeAlias(
		emit,
		"angularVelocityRange",
		resolveDefinition,
		DEFAULT_EMITTER.angularVelocityRange
	);
	return emit;
}

function defineRangeAlias(
	emit: ParticleEmitterParams,
	key:
		| "lifetimeRange"
		| "speedRange"
		| "sizeRange"
		| "rotationRange"
		| "angularVelocityRange",
	resolveDefinition: () => ParticleDefinition,
	fallback: [number, number]
): void {
	Object.defineProperty(emit, key, {
		configurable: true,
		enumerable: true,
		get: () => {
			const definition = resolveDefinition() as ParticleDefinition &
				Record<typeof key, [number, number] | undefined>;
			const current = definition[key];
			if (current) return current;
			const next = cloneRange(fallback);
			definition[key] = next;
			return next;
		},
		set: (value: [number, number] | undefined) => {
			const definition = resolveDefinition() as ParticleDefinition &
				Record<typeof key, [number, number] | undefined>;
			definition[key] = cloneRange(value ?? fallback);
		},
	});
}

function defineColorAlias(
	emit: ParticleEmitterParams,
	key: "startColor",
	resolveDefinition: () => ParticleDefinition,
	fallback: RGBA
): void {
	Object.defineProperty(emit, key, {
		configurable: true,
		enumerable: true,
		get: () => {
			const definition = resolveDefinition();
			if (!definition.startColor) {
				definition.startColor = cloneColor(fallback);
			}
			return definition.startColor;
		},
		set: (value: RGBA | undefined) => {
			resolveDefinition().startColor = cloneColor(value ?? fallback);
		},
	});
}

function normalizeParticleDefinitions(
	params: ParticleSystemParams
): ParticleDefinition[] {
	const rawDefinitions =
		params.definitions && params.definitions.length > 0 ?
			params.definitions
		:	[createDefaultParticleDefinition(params)];
	if (rawDefinitions.length > 8) {
		throw new Error("ParticleSystem supports at most 8 particle definitions.");
	}

	const definitions = rawDefinitions.map((definition, index) =>
		cloneParticleDefinition({
			...definition,
			id: definition.id ?? `definition-${index}`,
			weight: sanitizeWeight(definition.weight),
		})
	);
	if (definitions.every((definition) => (definition.weight ?? 1) <= 0)) {
		throw new Error("ParticleSystem requires at least one positive definition weight.");
	}
	return definitions;
}

function createDefaultParticleDefinition(
	params: ParticleSystemParams
): ParticleDefinition {
	const emit = params.emit ?? {};
	return {
		id: "definition-0",
		weight: 1,
		lifetimeRange: cloneRange(
			(emit as ParticleEmitterParams & {
				lifetimeRange?: [number, number];
			}).lifetimeRange ?? DEFAULT_EMITTER.lifetimeRange
		),
		speedRange: cloneRange(
			(emit as ParticleEmitterParams & {
				speedRange?: [number, number];
			}).speedRange ?? DEFAULT_EMITTER.speedRange
		),
		sizeRange: cloneRange(
			(emit as ParticleEmitterParams & {
				sizeRange?: [number, number];
			}).sizeRange ?? DEFAULT_EMITTER.sizeRange
		),
		startColor: cloneColor(
			(emit as ParticleEmitterParams & { startColor?: RGBA }).startColor ??
				DEFAULT_EMITTER.startColor
		),
		rotationRange: cloneRange(
			(emit as ParticleEmitterParams & {
				rotationRange?: [number, number];
			}).rotationRange ?? DEFAULT_EMITTER.rotationRange
		),
		angularVelocityRange: cloneRange(
			(emit as ParticleEmitterParams & {
				angularVelocityRange?: [number, number];
			}).angularVelocityRange ?? DEFAULT_EMITTER.angularVelocityRange
		),
		sizeOverLifetime: cloneNumberGradient(params.sizeOverLifetime ?? []),
		colorOverLifetime: cloneColorGradient(params.colorOverLifetime ?? []),
		shape: {
			kind: "billboard",
			texture: params.texture ?? null,
			atlas: params.atlas ? { ...params.atlas } : null,
			blendMode: params.blendMode ?? ParticleBlendMode.Alpha,
		},
		receiveShadows: params.receiveShadows ?? true,
		castShadows: params.castShadows ?? true,
		shadowDensity: resolveNonNegativeFinite(params.shadowDensity, 1),
		shadowSoftness: resolveNonNegativeFinite(params.shadowSoftness, 1),
	};
}

function cloneParticleDefinition(source: ParticleDefinition): ParticleDefinition {
	return {
		...source,
		id: source.id,
		weight: sanitizeWeight(source.weight),
		lifetimeRange: cloneRange(source.lifetimeRange),
		speedRange: cloneRange(source.speedRange),
		sizeRange: cloneRange(source.sizeRange),
		startColor: cloneColor(source.startColor),
		rotationRange: cloneOptionalRange(source.rotationRange),
		angularVelocityRange: cloneOptionalRange(source.angularVelocityRange),
		sizeOverLifetime: cloneNumberGradient(source.sizeOverLifetime ?? []),
		colorOverLifetime: cloneColorGradient(source.colorOverLifetime ?? []),
		shape:
			source.shape.kind === "billboard" ?
				{
					kind: "billboard",
					texture: source.shape.texture ?? null,
					atlas: source.shape.atlas ? { ...source.shape.atlas } : null,
					blendMode:
						source.shape.blendMode ?? ParticleBlendMode.Alpha,
				}
			:	{
					kind: "mesh",
					mesh: source.shape.mesh,
				},
		receiveShadows: source.receiveShadows ?? true,
		castShadows: source.castShadows ?? true,
		shadowDensity: resolveNonNegativeFinite(source.shadowDensity, 1),
		shadowSoftness: resolveNonNegativeFinite(source.shadowSoftness, 1),
	};
}

function resolveDefinitionBlendMode(definition: ParticleDefinition): ParticleBlendMode {
	if (definition.shape.kind === "billboard") {
		return definition.shape.blendMode ?? ParticleBlendMode.Alpha;
	}
	return ParticleBlendMode.Alpha;
}

function cloneOptionalRange(source: [number, number] | undefined): [number, number] | undefined {
	return source ? cloneRange(source) : undefined;
}

function cloneNumberGradient(
	source: ParticleGradientKey<number>[]
): ParticleGradientKey<number>[] {
	return source.map((key) => ({
		t: key.t,
		value: key.value,
	}));
}

function cloneColorGradient(
	source: ParticleGradientKey<RGBA>[]
): ParticleGradientKey<RGBA>[] {
	return source.map((key) => ({
		t: key.t,
		value: cloneColor(key.value),
	}));
}

function sanitizeWeight(value: number | undefined): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return 1;
	}
	return Math.max(0, value);
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
