import { WebGLCapabilityError } from "../../foundation/Error";
import type { WebGLSceneVariantDescriptor } from "./WebGLSceneProgramVariants";

export interface WebGLSceneSamplerLayout {
	readonly units: Readonly<Record<string, number>>;
	readonly activeSamplerNames: readonly string[];
	readonly required: number;
	readonly available: number;
}

/** @internal Builds the exact collision-free sampler ABI for one scene program. */
export function createWebGLSceneSamplerLayout(
	available: number,
	variant?: WebGLSceneVariantDescriptor,
	customSamplerNames: readonly string[] = [],
): WebGLSceneSamplerLayout {
	const names: string[] = [];
	const add = (name: string, active = true): void => {
		if (active && !names.includes(name)) names.push(name);
	};

	if (variant) {
		const scene = variant.scene;
		const material = variant.material;
		add("uShadowAtlas", scene.shadows);
		add("uShadowTransmittanceAtlas", scene.shadowTransmittance);
		add("uParticleShadowVolumeAtlas", scene.shadows);
		add("uClusterHeaderTexture", scene.clusteredLighting);
		add("uClusterIndexTexture", scene.clusteredLighting);
		add("uClusterLightTexture", scene.clusteredLighting);
		add("uLocalLightProbeCoeffs", scene.localLightProbes);
		add("uIrradianceProbeGridCoeffs", scene.irradianceProbeGrid);
		add("uEnvSpecularMap", scene.environmentSpecular);
		add("uEnvSpecularFallbackMap", scene.environmentSpecular);
		add("uBrdfLUT", scene.environmentSpecular);

		add("uBaseMap", material.baseMap);
		add("uMetallicRoughnessMap", material.metallicRoughnessMap);
		add("uSpecularMap", material.specularMap);
		add("uSpecularColorMap", material.specularColorMap);
		add("uNormalMap", material.normalMap);
		add("uEmissiveMap", material.emissiveMap);
		add("uOcclusionMap", material.occlusionMap);
		add("uClearcoatMap", material.clearcoatMap);
		add("uClearcoatRoughnessMap", material.clearcoatRoughnessMap);
		add("uClearcoatNormalMap", material.clearcoatNormalMap);
		add("uSheenColorMap", material.sheenColorMap);
		add("uSheenRoughnessMap", material.sheenRoughnessMap);
		add("uTransmissionMap", material.transmissionMap);
		add("uThicknessMap", material.thicknessMap);
		add("uTransmissionBackgroundMap", material.transmission);
		add("uTransmissionDepthMap", material.transmission);
		add("uIridescenceMap", material.iridescenceMap);
		add("uIridescenceThicknessMap", material.iridescenceThicknessMap);
		add("uAnisotropyMap", material.anisotropyMap);
	}
	for (const name of customSamplerNames) add(name);

	const safeAvailable = Math.max(0, Math.floor(available));
	if (names.length > safeAvailable) {
		throw new WebGLCapabilityError(
			"material-texture-unit-overflow",
			`required=${names.length}, available=${safeAvailable}, ` +
				`active=[${names.join(", ")}].`,
		);
	}
	const units: Record<string, number> = {};
	for (let index = 0; index < names.length; index++) units[names[index]] = index;
	return {
		units,
		activeSamplerNames: names,
		required: names.length,
		available: safeAvailable,
	};
}

export function getWebGLSceneSamplerUnit(
	layout: WebGLSceneSamplerLayout,
	name: string,
): number {
	const unit = layout.units[name];
	if (!Number.isInteger(unit)) {
		throw new Error(
			`WebGL scene sampler "${name}" is active in the program but missing ` +
				"from its exact sampler layout.",
		);
	}
	return unit;
}
