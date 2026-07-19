import type { Texture } from "../../core/Texture";
import type { IPrimitive } from "../../core/types";
import type { Material } from "../../materials/Material";
import type { MeshAsset } from "../../meshes/MeshAsset";
import { ParticleBlendMode } from "../../particles";
import { defineTransientKey } from "../../pipeline/types";
import type { IRenderBuffer } from "../types";

export interface WebGPUParticleDrawBatch {
	systemId: string;
	templateIndex?: number;
	templateId?: string;
	blendMode: ParticleBlendMode;
	texture: Texture | null;
	receiveShadows: boolean;
	castShadows: boolean;
	shadowDensity: number;
	shadowSoftness: number;
	instanceBuffer: IRenderBuffer;
	instanceCount: number;
	indirectBuffer: IRenderBuffer;
	indirectOffset: number;
}

export interface WebGPUParticleMeshDrawBatch {
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
	instanceBuffer: IRenderBuffer;
	instanceCount: number;
	indirectBuffer: IRenderBuffer;
	indirectOffset: number;
}

export const WEBGPU_PARTICLE_DRAW_BATCHES_KEY =
	defineTransientKey<WebGPUParticleDrawBatch[]>(
		"webgpu:particle-draw-batches"
	);

export const WEBGPU_PARTICLE_MESH_DRAW_BATCHES_KEY =
	defineTransientKey<WebGPUParticleMeshDrawBatch[]>(
		"webgpu:particle-mesh-draw-batches"
	);
