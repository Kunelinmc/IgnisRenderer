import type { Texture } from '../core/Texture'
import type { IVector3 } from '../maths/types'
import type { RGBA } from '../utils/Color'

export enum ParticleBlendMode {
	Alpha = 'alpha',
	Additive = 'additive',
}

export enum ParticleSpaceMode {
	Local = 'local',
	World = 'world',
}

export interface ParticleGradientKey<T> {
	t: number
	value: T
}

export interface ParticleAtlas {
	rows: number
	columns: number
	fps: number
	loop?: boolean
}

export interface ParticleBurst {
	time: number
	count: number
	cycles?: number
	interval?: number
}

export interface ParticleEmitterParams {
	rate?: number
	bursts?: ParticleBurst[]
	lifetimeRange?: [number, number]
	speedRange?: [number, number]
	sizeRange?: [number, number]
	direction?: IVector3
	spread?: number
	spawnRadius?: number
	startColor?: RGBA
	rotationRange?: [number, number]
	angularVelocityRange?: [number, number]
}

interface ParticleColliderBase {
	restitution?: number
	damping?: number
	enabled?: boolean
}

export interface ParticlePlaneCollider extends ParticleColliderBase {
	type: 'plane'
	normal: IVector3
	constant: number
}

export interface ParticleSphereCollider extends ParticleColliderBase {
	type: 'sphere'
	center: IVector3
	radius: number
}

export interface ParticleAABBCollider extends ParticleColliderBase {
	type: 'aabb'
	min: IVector3
	max: IVector3
}

export type ParticleCollider =
	| ParticlePlaneCollider
	| ParticleSphereCollider
	| ParticleAABBCollider

export interface ParticleSubEmitterConfig {
	enabled?: boolean
	trigger?: 'death'
	count?: number
	inheritVelocityScale?: number
	speedRange?: [number, number]
	lifetimeRange?: [number, number]
	sizeRange?: [number, number]
}

export interface ParticleSystemParams {
	name?: string
	visible?: boolean
	maxParticles?: number
	seed?: number
	space?: ParticleSpaceMode
	blendMode?: ParticleBlendMode
	texture?: Texture | null
	atlas?: ParticleAtlas | null
	position?: IVector3
	gravity?: IVector3
	emit?: ParticleEmitterParams
	sizeOverLifetime?: ParticleGradientKey<number>[]
	colorOverLifetime?: ParticleGradientKey<RGBA>[]
	colliders?: ParticleCollider[]
	subEmitter?: ParticleSubEmitterConfig | null
	receiveShadows?: boolean
}
