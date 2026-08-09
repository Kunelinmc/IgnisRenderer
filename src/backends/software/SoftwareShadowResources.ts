import type { ShadowCastingLight } from "../../lights";
import {
	clearParticleShadowVolumeGrid,
	createParticleShadowVolumeGrid,
} from "../../pipeline/ParticleShadowVolume";
import type {
	SoftwareShadowRenderTarget,
	SoftwareShadowRuntimeMap,
} from "./SoftwareShadowContracts";

/** @internal Owns Software shadow target allocation and retention. */
export class SoftwareShadowResources {
	public constructor(
		public readonly runtimeMap: SoftwareShadowRuntimeMap,
	) {}

	public ensure(
		light: ShadowCastingLight,
		sliceIndex: number,
		size: number,
	): SoftwareShadowRenderTarget {
		let targets = this.runtimeMap.get(light);
		if (!targets) {
			targets = [];
			this.runtimeMap.set(light, targets);
		}
		let target = targets[sliceIndex];
		if (!target || target.size !== size) {
			target = {
				size,
				depthBuffer: new Float32Array(size * size),
				transmissionBuffer: new Float32Array(size * size * 3),
				particleVolume: createParticleShadowVolumeGrid(),
			};
			targets[sliceIndex] = target;
		}
		return target;
	}

	public clearTarget(target: SoftwareShadowRenderTarget): void {
		target.depthBuffer.fill(Infinity);
		target.transmissionBuffer.fill(1.0);
		clearParticleShadowVolumeGrid(target.particleVolume);
	}

	public syncLights(activeLights: ShadowCastingLight[]): void {
		for (const [light] of this.runtimeMap) {
			if (!activeLights.includes(light)) this.runtimeMap.delete(light);
		}
	}

	public trim(light: ShadowCastingLight, sliceCount: number): void {
		const targets = this.runtimeMap.get(light);
		if (targets) targets.length = Math.max(0, sliceCount | 0);
	}

	public clear(): void {
		this.runtimeMap.clear();
	}

	public destroy(): void {
		this.clear();
	}
}
