import type { ParticleLODLevel } from "../../particles";
import type { IVector3 } from "../../maths/types";
import type { RGBA } from "../../foundation/Color";

export interface RuntimeParticle {
	position: IVector3;
	velocity: IVector3;
	age: number;
	lifetime: number;
	startSize: number;
	startColor: RGBA;
	rotation: number;
	angularVelocity: number;
}

export interface SystemRuntimeState {
	particles: RuntimeParticle[];
	emissionRemainder: number;
	elapsed: number;
	burstCycles: number[];
	randomState: number;
	frameIndex: number;
	pendingSimulationTime: number;
	lodLevelIndex: number;
	lodCandidateLevelIndex: number;
	lodCandidateFrameCount: number;
	activeLODLevel: ParticleLODLevel | null;
}
