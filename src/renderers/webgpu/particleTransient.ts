import type { Texture } from "../../core/Texture";
import { ParticleBlendMode } from "../../particles";
import { defineTransientKey } from "../../pipeline/types";
import type { IRenderBuffer } from "../types";

export interface WebGPUParticleDrawBatch {
	systemId: string;
	blendMode: ParticleBlendMode;
	texture: Texture | null;
	receiveShadows: boolean;
	instanceBuffer: IRenderBuffer;
	instanceCount: number;
	indirectBuffer: IRenderBuffer;
	indirectOffset: number;
}

export const WEBGPU_PARTICLE_DRAW_BATCHES_KEY =
	defineTransientKey<WebGPUParticleDrawBatch[]>(
		"webgpu:particle-draw-batches"
	);
