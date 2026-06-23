import type { Texture } from "../core/Texture";
import type { MeshAsset } from "../meshes/MeshAsset";
import type { IVector3 } from "../maths/types";
import type { RGBA } from "../foundation/Color";

export enum ParticleBlendMode {
	Alpha = "alpha",
	Additive = "additive",
}

export enum ParticleSpaceMode {
	Local = "local",
	World = "world",
}

export interface ParticleGradientKey<T> {
	t: number;
	value: T;
}

export interface ParticleAtlas {
	rows: number;
	columns: number;
	fps: number;
	loop?: boolean;
}

export interface ParticleBurst {
	time: number;
	count: number;
	cycles?: number;
	interval?: number;
}

export interface ParticleEmitterParams {
	rate?: number;
	bursts?: ParticleBurst[];
	direction?: IVector3;
	spread?: number;
	spawnRadius?: number;
	/**
	 * @deprecated Use `definitions[].lifetimeRange`.
	 */
	lifetimeRange?: [number, number];
	/**
	 * @deprecated Use `definitions[].speedRange`.
	 */
	speedRange?: [number, number];
	/**
	 * @deprecated Use `definitions[].sizeRange`.
	 */
	sizeRange?: [number, number];
	/**
	 * @deprecated Use `definitions[].startColor`.
	 */
	startColor?: RGBA;
	/**
	 * @deprecated Use `definitions[].rotationRange`.
	 */
	rotationRange?: [number, number];
	/**
	 * @deprecated Use `definitions[].angularVelocityRange`.
	 */
	angularVelocityRange?: [number, number];
}

export interface ParticleBillboardShape {
	kind: "billboard";
	texture?: Texture | null;
	atlas?: ParticleAtlas | null;
	blendMode?: ParticleBlendMode;
}

export interface ParticleMeshShape {
	kind: "mesh";
	mesh: MeshAsset;
}

export type ParticleRenderShape = ParticleBillboardShape | ParticleMeshShape;

export type ParticleRange = [number, number];
export type ParticleCurve = ParticleGradientKey<number>[];
export type ParticleColorGradient = ParticleGradientKey<RGBA>[];

export interface ParticleTemplate {
	id?: string;
	weight?: number;
	lifetimeRange: ParticleRange;
	speedRange: ParticleRange;
	sizeRange: ParticleRange;
	startColor?: RGBA;
	rotationRange?: ParticleRange;
	angularVelocityRange?: ParticleRange;
	sizeOverLifetime?: ParticleCurve;
	colorOverLifetime?: ParticleColorGradient;
	shape: ParticleRenderShape;
	receiveShadows?: boolean;
	castShadows?: boolean;
	shadowDensity?: number;
	shadowSoftness?: number;
}

/** @deprecated Use ParticleTemplate instead. */
export type ParticleDefinition = ParticleTemplate;

interface ParticleColliderBase {
	restitution?: number;
	damping?: number;
	enabled?: boolean;
}

export interface ParticlePlaneCollider extends ParticleColliderBase {
	type: "plane";
	normal: IVector3;
	constant: number;
}

export interface ParticleSphereCollider extends ParticleColliderBase {
	type: "sphere";
	center: IVector3;
	radius: number;
}

export interface ParticleAABBCollider extends ParticleColliderBase {
	type: "aabb";
	min: IVector3;
	max: IVector3;
}

export type ParticleCollider =
	| ParticlePlaneCollider
	| ParticleSphereCollider
	| ParticleAABBCollider;

export interface ParticleSubEmitterConfig {
	enabled?: boolean;
	trigger?: "death";
	count?: number;
	inheritVelocityScale?: number;
	speedRange?: [number, number];
	lifetimeRange?: [number, number];
	sizeRange?: [number, number];
}

export interface ParticleLODLevel {
	distance: number;
	projectedSize: number;
	simulationIntervalFrames: number;
	spawnScale: number;
	maxParticlesScale: number;
	renderSortRatio: number;
}

export interface ParticleLODSettings {
	enabled?: boolean;
	hysteresisFrames?: number;
	levels: ParticleLODLevel[];
}

export interface ParticleSystemParams {
	name?: string;
	visible?: boolean;
	maxParticles?: number;
	seed?: number;
	space?: ParticleSpaceMode;
	position?: IVector3;
	gravity?: IVector3;
	emit?: ParticleEmitterParams;
	templates?: ParticleTemplate[];
	/** @deprecated Use templates instead. */
	definitions?: ParticleTemplate[];
	colliders?: ParticleCollider[];
	subEmitter?: ParticleSubEmitterConfig | null;
	lod?: ParticleLODSettings;

	/**
	 * @deprecated Use `templates[].shape.blendMode`.
	 */
	blendMode?: ParticleBlendMode;
	/**
	 * @deprecated Use `templates[].shape.texture`.
	 */
	texture?: Texture | null;
	/**
	 * @deprecated Use `templates[].shape.atlas`.
	 */
	atlas?: ParticleAtlas | null;
	/**
	 * @deprecated Use `templates[].sizeOverLifetime`.
	 */
	sizeOverLifetime?: ParticleGradientKey<number>[];
	/**
	 * @deprecated Use `templates[].colorOverLifetime`.
	 */
	colorOverLifetime?: ParticleGradientKey<RGBA>[];
	/**
	 * @deprecated Use `templates[].receiveShadows`.
	 */
	receiveShadows?: boolean;
	/**
	 * @deprecated Use `templates[].castShadows`.
	 */
	castShadows?: boolean;
	/**
	 * @deprecated Use `templates[].shadowDensity`.
	 */
	shadowDensity?: number;
	/**
	 * @deprecated Use `templates[].shadowSoftness`.
	 */
	shadowSoftness?: number;
}
