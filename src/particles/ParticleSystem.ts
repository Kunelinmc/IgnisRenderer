import { Node } from "../core/Node";
import type { IVector3 } from "../maths/types";
import {
	ParticleBlendMode,
	ParticleSpaceMode,
	type ParticleAtlas,
	type ParticleCollider,
	type ParticleTemplate,
	type ParticleRange,
	type ParticleCurve,
	type ParticleColorGradient,
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
	lifetimeRange: [0.5, 1.5] as [number, number],
	speedRange: [2, 5] as [number, number],
	sizeRange: [0.5, 1.0] as [number, number],
	direction: { x: 0, y: 1, z: 0 },
	spread: 0.1,
	spawnRadius: 0.0,
	startColor: { r: 255, g: 255, b: 255, a: 1.0 },
	rotationRange: [0, 0] as [number, number],
	angularVelocityRange: [0, 0] as [number, number],
};

const DEFAULT_GRAVITY = { x: 0, y: -9.8, z: 0 };

const DEFAULT_LOD_LEVELS: ParticleLODLevel[] = [
	{
		distance: 20,
		projectedSize: 50,
		simulationIntervalFrames: 1,
		spawnScale: 1,
		maxParticlesScale: 1,
		renderSortRatio: 1,
	},
	{
		distance: 50,
		projectedSize: 20,
		simulationIntervalFrames: 2,
		spawnScale: 0.7,
		maxParticlesScale: 0.7,
		renderSortRatio: 0.5,
	},
	{
		distance: 100,
		projectedSize: 5,
		simulationIntervalFrames: 4,
		spawnScale: 0.3,
		maxParticlesScale: 0.3,
		renderSortRatio: 0.2,
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
	public templates: ParticleTemplate[];
	public colliders: ParticleCollider[];
	public subEmitter: ParticleSubEmitterConfig | null;
	public lod: ParticleLODSettings;

	/** @deprecated Use templates instead. */
	public get definitions(): ParticleTemplate[] {
		return this.templates;
	}

	/** @deprecated Use templates instead. */
	public set definitions(value: ParticleTemplate[]) {
		this.templates = value;
	}

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
		this.templates = normalizeParticleTemplates(params);
		this.emit = createEmitterParams(
			params.emit,
			() => this._primaryTemplate
		);
		this.colliders = (params.colliders ?? []).map((collider) =>
			cloneCollider(collider)
		);
		this.subEmitter = params.subEmitter ? { ...params.subEmitter } : null;
		this.lod = cloneLOD(params.lod ?? DEFAULT_LOD);
	}

	public get blendMode(): ParticleBlendMode {
		return resolveTemplateBlendMode(this._primaryTemplate);
	}

	public set blendMode(value: ParticleBlendMode) {
		const template = this._primaryTemplate;
		if (template.shape.kind === "billboard") {
			template.shape.blendMode = value;
		}
	}

	public get texture(): Texture | null {
		const shape = this._primaryTemplate.shape;
		return shape.kind === "billboard" ? shape.texture ?? null : null;
	}

	public set texture(value: Texture | null) {
		const template = this._primaryTemplate;
		if (template.shape.kind === "billboard") {
			template.shape.texture = value;
		}
	}

	public get atlas(): ParticleAtlas | null {
		const shape = this._primaryTemplate.shape;
		return shape.kind === "billboard" ? shape.atlas ?? null : null;
	}

	public set atlas(value: ParticleAtlas | null) {
		const template = this._primaryTemplate;
		if (template.shape.kind === "billboard") {
			template.shape.atlas = value;
		}
	}

	public get sizeOverLifetime(): ParticleGradientKey<number>[] {
		return this._primaryTemplate.sizeOverLifetime ?? [];
	}

	public set sizeOverLifetime(value: ParticleGradientKey<number>[]) {
		this._primaryTemplate.sizeOverLifetime = cloneNumberGradient(value);
	}

	public get colorOverLifetime(): ParticleGradientKey<RGBA>[] {
		return this._primaryTemplate.colorOverLifetime ?? [];
	}

	public set colorOverLifetime(value: ParticleGradientKey<RGBA>[]) {
		this._primaryTemplate.colorOverLifetime = cloneColorGradient(value);
	}

	public get receiveShadows(): boolean {
		return this._primaryTemplate.receiveShadows ?? true;
	}

	public set receiveShadows(value: boolean) {
		this._primaryTemplate.receiveShadows = value;
	}

	public get castShadows(): boolean {
		return this._primaryTemplate.castShadows ?? true;
	}

	public set castShadows(value: boolean) {
		this._primaryTemplate.castShadows = value;
	}

	public get shadowDensity(): number {
		return this._primaryTemplate.shadowDensity ?? 1;
	}

	public set shadowDensity(value: number) {
		this._primaryTemplate.shadowDensity = resolveNonNegativeFinite(value, 1);
	}

	public get shadowSoftness(): number {
		return this._primaryTemplate.shadowSoftness ?? 1;
	}

	public set shadowSoftness(value: number) {
		this._primaryTemplate.shadowSoftness = resolveNonNegativeFinite(value, 1);
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
		target.templates = this.templates.map((template) =>
			cloneParticleTemplate(template)
		);
		target.emit = createEmitterParams(
			this.emit,
			() => target._primaryTemplate
		);
		target.colliders = this.colliders.map((collider) =>
			cloneCollider(collider)
		);
		target.subEmitter = this.subEmitter ? { ...this.subEmitter } : null;
		target.lod = cloneLOD(this.lod);
	}

	private get _primaryTemplate(): ParticleTemplate {
		if (this.templates.length === 0) {
			this.templates.push(createDefaultParticleTemplate({}));
		}
		return this.templates[0];
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
	resolveTemplate: () => ParticleTemplate
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
		resolveTemplate,
		DEFAULT_EMITTER.lifetimeRange
	);
	defineRangeAlias(
		emit,
		"speedRange",
		resolveTemplate,
		DEFAULT_EMITTER.speedRange
	);
	defineRangeAlias(
		emit,
		"sizeRange",
		resolveTemplate,
		DEFAULT_EMITTER.sizeRange
	);
	defineColorAlias(
		emit,
		"startColor",
		resolveTemplate,
		DEFAULT_EMITTER.startColor
	);
	defineRangeAlias(
		emit,
		"rotationRange",
		resolveTemplate,
		DEFAULT_EMITTER.rotationRange
	);
	defineRangeAlias(
		emit,
		"angularVelocityRange",
		resolveTemplate,
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
	resolveTemplate: () => ParticleTemplate,
	fallback: [number, number]
): void {
	Object.defineProperty(emit, key, {
		configurable: true,
		enumerable: true,
		get: () => {
			const template = resolveTemplate() as ParticleTemplate &
				Record<typeof key, [number, number] | undefined>;
			const current = template[key];
			if (current) return current;
			const next = cloneRange(fallback);
			template[key] = next;
			return next;
		},
		set: (value: [number, number] | undefined) => {
			const template = resolveTemplate() as ParticleTemplate &
				Record<typeof key, [number, number] | undefined>;
			template[key] = cloneRange(value ?? fallback);
		},
	});
}

function defineColorAlias(
	emit: ParticleEmitterParams,
	key: "startColor",
	resolveTemplate: () => ParticleTemplate,
	fallback: RGBA
): void {
	Object.defineProperty(emit, key, {
		configurable: true,
		enumerable: true,
		get: () => {
			const template = resolveTemplate();
			if (!template.startColor) {
				template.startColor = cloneColor(fallback);
			}
			return template.startColor!;
		},
		set: (value: RGBA | undefined) => {
			resolveTemplate().startColor = cloneColor(value ?? fallback);
		},
	});
}

function normalizeParticleTemplates(
	params: ParticleSystemParams
): ParticleTemplate[] {
	const rawTemplates =
		params.templates && params.templates.length > 0 ? params.templates
		: params.definitions && params.definitions.length > 0 ? params.definitions
		: [createDefaultParticleTemplate(params)];

	if (rawTemplates.length > 8) {
		throw new Error("ParticleSystem supports at most 8 particle templates.");
	}

	const templates = rawTemplates.map((template, index) =>
		cloneParticleTemplate({
			...template,
			id: template.id ?? `template-${index}`,
			weight: sanitizeWeight(template.weight),
		})
	);
	if (templates.every((template) => (template.weight ?? 1) <= 0)) {
		throw new Error("ParticleSystem requires at least one positive template weight.");
	}
	return templates;
}

function createDefaultParticleTemplate(
	params: ParticleSystemParams
): ParticleTemplate {
	const emit = params.emit ?? {};
	return {
		id: "template-0",
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

function cloneParticleTemplate(source: ParticleTemplate): ParticleTemplate {
	return {
		...source,
		id: source.id,
		weight: sanitizeWeight(source.weight),
		lifetimeRange: cloneRange(source.lifetimeRange),
		speedRange: cloneRange(source.speedRange),
		sizeRange: cloneRange(source.sizeRange),
		startColor: source.startColor ? cloneColor(source.startColor) : undefined,
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

function resolveTemplateBlendMode(template: ParticleTemplate): ParticleBlendMode {
	if (template.shape.kind === "billboard") {
		return template.shape.blendMode ?? ParticleBlendMode.Alpha;
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
