import type { Texture } from "../core/Texture";
import type { IPrimitive } from "../core/types";
import type { RGBA } from "../foundation/Color";
import type { Material } from "../materials/Material";
import type { IVector3 } from "../maths/types";
import type { MeshAsset } from "../meshes";
import type { ParticleBlendMode } from "./types";

/** @internal Particle subsystem UV data consumed by billboard renderers. */
export interface ParticleUVRect {
	u0: number;
	v0: number;
	u1: number;
	v1: number;
}

/** @internal Particle subsystem item consumed by billboard renderers. */
export interface ParticleRenderItem {
	templateIndex?: number;
	position: IVector3;
	previousPosition?: IVector3;
	size: number;
	color: RGBA;
	rotation: number;
	previousRotation?: number;
	depth: number;
	uvRect: ParticleUVRect;
}

/** @internal Particle subsystem batch consumed by billboard renderers. */
export interface ParticleRenderBatch {
	kind?: "billboard";
	systemId: string;
	templateIndex?: number;
	templateId?: string;
	blendMode: ParticleBlendMode;
	texture: Texture | null;
	receiveShadows: boolean;
	castShadows: boolean;
	shadowDensity: number;
	shadowSoftness: number;
	particles: ParticleRenderItem[];
}

/** @internal Particle subsystem item consumed by mesh renderers. */
export interface ParticleMeshRenderItem {
	templateIndex: number;
	position: IVector3;
	previousPosition: IVector3;
	size: number;
	rotation: number;
	previousRotation: number;
	depth: number;
}

/** @internal Particle subsystem batch consumed by mesh renderers. */
export interface ParticleMeshRenderBatch {
	kind: "mesh";
	systemId: string;
	templateIndex: number;
	templateId?: string;
	mesh: MeshAsset;
	primitive: IPrimitive;
	material: Material;
	receiveShadows: boolean;
	castShadows: boolean;
	shadowDensity: number;
	shadowSoftness: number;
	particles: ParticleMeshRenderItem[];
}
