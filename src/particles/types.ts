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

export interface ParticleDefinition {
	id?: string;
	weight?: number;
	lifetimeRange: [number, number];
	speedRange: [number, number];
	sizeRange: [number, number];
	startColor: RGBA;
	rotationRange?: [number, number];
	angularVelocityRange?: [number, number];
	sizeOverLifetime?: ParticleGradientKey<number>[];
	colorOverLifetime?: ParticleGradientKey<RGBA>[];
	shape: ParticleRenderShape;
	receiveShadows?: boolean;
	castShadows?: boolean;
	shadowDensity?: number;
	shadowSoftness?: number;
}

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
	definitions?: ParticleDefinition[];
	colliders?: ParticleCollider[];
	subEmitter?: ParticleSubEmitterConfig | null;
	lod?: ParticleLODSettings;

	/**
	 * @deprecated Use `definitions[].shape.blendMode`.
	 */
	blendMode?: ParticleBlendMode;
	/**
	 * @deprecated Use `definitions[].shape.texture`.
	 */
	texture?: Texture | null;
	/**
	 * @deprecated Use `definitions[].shape.atlas`.
	 */
	atlas?: ParticleAtlas | null;
	/**
	 * @deprecated Use `definitions[].sizeOverLifetime`.
	 */
	sizeOverLifetime?: ParticleGradientKey<number>[];
	/**
	 * @deprecated Use `definitions[].colorOverLifetime`.
	 */
	colorOverLifetime?: ParticleGradientKey<RGBA>[];
	/**
	 * @deprecated Use `definitions[].receiveShadows`.
	 */
	receiveShadows?: boolean;
	/**
	 * @deprecated Use `definitions[].castShadows`.
	 */
	castShadows?: boolean;
	/**
	 * @deprecated Use `definitions[].shadowDensity`.
	 */
	shadowDensity?: number;
	/**
	 * @deprecated Use `definitions[].shadowSoftness`.
	 */
	shadowSoftness?: number;
}
