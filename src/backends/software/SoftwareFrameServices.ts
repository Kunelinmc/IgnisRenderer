import type {
	SoftwareShadowRuntimeMap,
	SoftwareShadowSampler,
	SoftwareShadowSamplerFactory,
} from "./SoftwareShadowContracts";
import type {
	SoftwareParticleFrameState,
} from "./SoftwareFrameView";
import { createSoftwareFrameView } from "./SoftwareFrameView";
import type { FrameContext } from "../../pipeline/types";
import { SoftwareShadowResources } from "./SoftwareShadowResources";
import { SoftwareReflectionResources } from "./SoftwareReflectionResources";

export interface SoftwareShadowFrameService {
	readonly runtimeMap: SoftwareShadowRuntimeMap;
	readonly resources: SoftwareShadowResources;
	sampler: SoftwareShadowSampler;
	samplerFactory: SoftwareShadowSamplerFactory;
}

/** @internal Explicit backend-owned services shared by Software passes. */
export interface SoftwareFrameServices {
	readonly shadow: SoftwareShadowFrameService;
	readonly reflection: SoftwareReflectionResources;
	readonly particles: SoftwareParticleFrameState;
}

/** @internal Inputs supplied to concrete Software render passes. */
export interface SoftwarePassContext {
	readonly frame: import("./SoftwareFrameView").SoftwareFrameView;
	readonly services: SoftwareFrameServices;
}

const FULLY_LIT_SHADOW: SoftwareShadowSampler = () => ({ r: 1, g: 1, b: 1 });

export function createSoftwareFrameServices(): SoftwareFrameServices {
	const shadowRuntimeMap: SoftwareShadowRuntimeMap = new Map();
	return {
		shadow: {
			runtimeMap: shadowRuntimeMap,
			resources: new SoftwareShadowResources(shadowRuntimeMap),
			sampler: FULLY_LIT_SHADOW,
			samplerFactory: () => FULLY_LIT_SHADOW,
		},
		reflection: new SoftwareReflectionResources(),
		particles: {
			batches: [],
			meshBatches: [],
		},
	};
}

export function resetSoftwareFrameServices(services: SoftwareFrameServices): void {
	services.shadow.sampler = FULLY_LIT_SHADOW;
	services.shadow.samplerFactory = () => FULLY_LIT_SHADOW;
	services.particles.batches = [];
	services.particles.meshBatches = [];
}

/** @internal Constructs an isolated pass context for direct subsystem tests. */
export function createSoftwarePassContextForTesting(
	context: FrameContext,
): SoftwarePassContext {
	const temporal = {
		currentJitter: [0, 0] as [number, number],
		previousJitter: [0, 0] as [number, number],
		previousViewProjection: null,
		currentViewProjection: context.viewCamera.viewProjectionMatrix,
		previousWorldMatrices: new Map(),
		currentWorldMatrices: new Map(),
	};
	return {
		frame: createSoftwareFrameView(
			context,
			temporal,
			new Float32Array(context.attachments.width * context.attachments.height * 4),
		),
		services: createSoftwareFrameServices(),
	};
}
