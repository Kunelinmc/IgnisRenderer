import {
	AlphaMode,
	ShadingModel,
	type Material,
} from "../../materials/Material";
import {
	ShaderMaterial,
	type ShaderTargetMode,
} from "../../materials/ShaderMaterial";
import type { FrameContext } from "../../pipeline/types";
import {
	getWebGLSceneDepthVariantKey,
	getWebGLSceneVariantKey,
	normalizeWebGLSceneDepthVariantDescriptor,
	normalizeWebGLSceneVariantDescriptor,
	type WebGLSceneDepthVariantDescriptor,
	type WebGLSceneMaterialModel,
	type WebGLSceneVariantDescriptor,
} from "../../shaders/webgl/sceneVariants";
import type { WebGLLightState } from "./WebGLLightCollector";
import { resolveMaterialUniforms } from "./WebGLMaterialUniformResolver";

const EPSILON = 0.000001;

export type {
	WebGLSceneDepthVariantDescriptor,
	WebGLSceneVariantDescriptor,
};
export {
	getWebGLSceneDepthVariantKey,
	getWebGLSceneVariantKey,
	normalizeWebGLSceneDepthVariantDescriptor,
	normalizeWebGLSceneVariantDescriptor,
};

export interface WebGLSceneVariantEnvironment {
	lightState: WebGLLightState | null;
	enableShadowTransmittance: boolean;
	enableIrradianceProbeGrid: boolean;
}

export function resolveWebGLBuiltinSceneVariant(
	context: FrameContext,
	material: Material,
	mode: ShaderTargetMode,
	oitPassMode: 0 | 1 | 2,
	environment: WebGLSceneVariantEnvironment,
	materialGBuffer = false
): WebGLSceneVariantDescriptor | null {
	if (material instanceof ShaderMaterial) {
		return null;
	}
	const materialVariant = resolveWebGLSceneMaterialVariant(material);
	const isLit =
		materialVariant.model === "pbr" ||
		materialVariant.model === "legacy" ||
		materialVariant.model === "full";
	const lightState = environment.lightState;
	const features = context.features as Partial<FrameContext["features"]> | undefined;
	const hasShadows =
		isLit &&
		features?.enableShadows === true &&
		!!lightState &&
		(
			lightState.directionalShadows.some((shadow) => shadow.enabled) ||
			lightState.spotShadows.some((shadow) => shadow.enabled)
		);
	const hasSH = isLit && features?.enableSH === true;
	const hasEnvironmentSpecular =
		materialVariant.model === "pbr" &&
		!!lightState?.envSpecularMap;
	return normalizeWebGLSceneVariantDescriptor({
		output: mode === "mrt" ? "mrt" : "single",
		materialGBuffer: mode === "mrt" && materialGBuffer,
		oit: oitPassMode !== 0,
		scene: {
			shadows: hasShadows,
			shadowTransmittance:
				hasShadows && environment.enableShadowTransmittance,
			clusteredLighting:
				isLit &&
				features?.enableClusteredLighting === true &&
				(lightState?.clusteredLights.length ?? 0) > 0,
			sh: hasSH,
			localLightProbes:
				hasSH && (lightState?.localLightProbeCount ?? 0) > 0,
			irradianceProbeGrid:
				hasSH &&
				environment.enableIrradianceProbeGrid &&
				!!lightState?.irradianceProbeGrid,
			reflectionProbes:
				hasEnvironmentSpecular &&
				(lightState?.reflectionProbeCount ?? 0) > 0,
			environmentSpecular: hasEnvironmentSpecular,
		},
		material: materialVariant,
	});
}

export function resolveWebGLBuiltinDepthVariant(
	material: Material
): WebGLSceneDepthVariantDescriptor | null {
	if (material instanceof ShaderMaterial) {
		return null;
	}
	const uniforms = resolveMaterialUniforms(material);
	const alphaMask = material.alphaMode === AlphaMode.Mask;
	return normalizeWebGLSceneDepthVariantDescriptor({
		alphaMask,
		baseMap: alphaMask && !!uniforms.baseMap,
	});
}

function resolveWebGLSceneMaterialVariant(
	material: Material
): WebGLSceneVariantDescriptor["material"] {
	const uniforms = resolveMaterialUniforms(material);
	const model = resolveMaterialModel(material);
	const isPBR = model === "pbr";
	const iridescence =
		isPBR &&
		(
			uniforms.iridescence[0] > EPSILON ||
			!!uniforms.iridescenceMap ||
			!!uniforms.iridescenceThicknessMap
		);
	const anisotropy =
		isPBR &&
		(uniforms.anisotropy[0] > EPSILON || !!uniforms.anisotropyMap);
	const clearcoat = isPBR && (
		uniforms.clearcoat[0] > EPSILON || !!uniforms.clearcoatMap ||
		!!uniforms.clearcoatRoughnessMap || !!uniforms.clearcoatNormalMap
	);
	const sheen = isPBR && (
		Math.max(uniforms.sheen[0], uniforms.sheen[1], uniforms.sheen[2]) > EPSILON ||
		!!uniforms.sheenColorMap || !!uniforms.sheenRoughnessMap
	);
	const transmission = isPBR && (
		uniforms.pbr[3] > EPSILON || !!uniforms.transmissionMap
	);
	return {
		model,
		baseMap: !!uniforms.baseMap,
		metallicRoughnessMap: isPBR && !!uniforms.metallicRoughnessMap,
		specularMap: isPBR && !!uniforms.specularMap,
		specularColorMap: isPBR && !!uniforms.specularColorMap,
		normalMap: isPBR && !!uniforms.normalMap,
		emissiveMap: !!uniforms.emissiveMap,
		occlusionMap: isPBR && !!uniforms.occlusionMap,
		clearcoat,
		clearcoatMap: clearcoat && !!uniforms.clearcoatMap,
		clearcoatRoughnessMap: clearcoat && !!uniforms.clearcoatRoughnessMap,
		clearcoatNormalMap: clearcoat && !!uniforms.clearcoatNormalMap,
		sheen,
		sheenColorMap: sheen && !!uniforms.sheenColorMap,
		sheenRoughnessMap: sheen && !!uniforms.sheenRoughnessMap,
		iridescence,
		iridescenceMap: iridescence && !!uniforms.iridescenceMap,
		iridescenceThicknessMap:
			iridescence && !!uniforms.iridescenceThicknessMap,
		anisotropy,
		anisotropyMap: anisotropy && !!uniforms.anisotropyMap,
		transmission,
		transmissionMap: transmission && !!uniforms.transmissionMap,
		thicknessMap: transmission && !!uniforms.thicknessMap,
		alphaMask: material.alphaMode === AlphaMode.Mask,
	};
}

function resolveMaterialModel(material: Material): WebGLSceneMaterialModel {
	if (material.shading === ShadingModel.Unlit) {
		return "unlit";
	}
	if (material.shading === ShadingModel.PBR || material.type === "PBR") {
		return "pbr";
	}
	return "legacy";
}
