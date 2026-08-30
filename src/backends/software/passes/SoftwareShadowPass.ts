import { isShadowCastingLight } from "../../../lights";
import type { ShadowCastingLight } from "../../../lights";
import type {
	PreparedShadowSlice,
} from "../../../lights/shadows/ShadowFramePlan";
import {
	hasParticleShadowCastingBatches,
	injectParticleBatchIntoShadowVolume,
} from "../../../pipeline/ParticleShadowVolume";
import type { Rasterizer } from "../Rasterizer";
import type { SoftwarePassLike } from "./types";
import type { SoftwarePassContext } from "../SoftwareFrameServices";
import {
	createSoftwareShadowSampler as createShadowSampler,
	sampleSoftwareShadow,
} from "../SoftwareShadowSampler";
import { SoftwareShadowRasterPass } from "../SoftwareShadowRasterPass";
import type {
	SoftwareShadowRenderTarget,
	SoftwareShadowRuntimeMap,
	SoftwareShadowSamplerCamera,
} from "../SoftwareShadowContracts";

export type {
	SoftwareShadowRenderTarget,
	SoftwareShadowRuntimeMap,
} from "../SoftwareShadowContracts";

export class SoftwareShadowPass implements SoftwarePassLike {
	private readonly _rasterPass: SoftwareShadowRasterPass;
	private readonly _shadowLightsScratch: ShadowCastingLight[] = [];

	constructor(rasterizer: Rasterizer) {
		this._rasterPass = new SoftwareShadowRasterPass(rasterizer);
	}

	public render(context: SoftwarePassContext): void {
		const frame = context.frame;
		const runtimeMap = context.services.shadow.runtimeMap;
		const resources = context.services.shadow.resources;
		const scene = frame.scene;
		const shadowPlan = frame.shadowPlan;
		const shadowLights = this._shadowLightsScratch;
		shadowLights.length = 0;
		for (const prepared of shadowPlan.lights) {
			if (isShadowCastingLight(prepared.light)) {
				shadowLights.push(prepared.light);
			}
		}
		resources.syncLights(shadowLights);
		this.bindSamplers(context);

		if (shadowLights.length === 0) {
			runtimeMap.clear();
			return;
		}

		for (const shadowLight of shadowLights) {
			const prepared = shadowPlan.lights.find((candidate) => candidate.light === shadowLight);
			if (!prepared || prepared.slices.length <= 0) {
				resources.trim(shadowLight, 0);
				continue;
			}
			resources.trim(shadowLight, prepared.slices.length);

			for (let sliceIndex = 0; sliceIndex < prepared.slices.length; sliceIndex++) {
				const shadowSlice = prepared.slices[sliceIndex];
				const shadowMapSize = shadowSlice.resolution;
				const shadowRuntime = resources.ensure(
					shadowLight,
					sliceIndex,
					shadowMapSize,
				);
				resources.clearTarget(shadowRuntime);

				this._rasterPass.renderSlice(
					frame,
					shadowSlice,
					shadowRuntime,
					scene.shadowCasterPackets,
					scene.shadowTransmitterPackets,
				);

				this._injectParticleShadowVolume(context, shadowSlice, shadowRuntime);
			}
		}
	}

	public bindSamplers(context: SoftwarePassContext): void {
		const factory = (camera: SoftwareShadowSamplerCamera) =>
			createShadowSampler(
				context.frame.shadowPlan,
				context.services.shadow.runtimeMap,
				sampleSoftwareShadow,
				{ camera },
			);
		context.services.shadow.samplerFactory = factory;
		context.services.shadow.sampler = factory(context.frame.camera);
	}

	private _injectParticleShadowVolume(
		context: SoftwarePassContext,
		shadowSlice: PreparedShadowSlice,
		shadowRuntime: SoftwareShadowRenderTarget
	): void {
		const batches = context.services.particles.batches;
		if (!hasParticleShadowCastingBatches(batches)) {
			return;
		}
		for (const batch of batches ?? []) {
			injectParticleBatchIntoShadowVolume(
				shadowRuntime.particleVolume,
				shadowSlice.viewProjection,
				batch
			);
		}
	}
}
