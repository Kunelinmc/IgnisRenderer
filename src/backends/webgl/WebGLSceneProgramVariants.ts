import {
	AlphaMode,
	ShadingModel,
	type Material,
} from "../../materials/Material";
import {
	ShaderMaterial,
	type ShaderTargetMode,
} from "../../materials/ShaderMaterial";
import {
	PBRMaterial,
	PBRMaterialFeature,
	PBRMaterialTextureFeature,
} from "../../materials/PBRMaterial";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import { resolveShaderManifestRequest } from "../../shaders/ShaderManifest";
import { WEBGL_SHADER_MANIFEST } from "../../shaders/webgl/sources";
import {
	WEBGL_FULL_SCENE_VARIANT,
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

/** @internal Resolves the manifest-owned scene specialization and identity. */
export function resolveWebGLSceneSourceSpecialization(
	variant?: WebGLSceneVariantDescriptor,
): { specialization: WebGLSceneVariantDescriptor; identity: string } {
	const resolved = resolveShaderManifestRequest(
		WEBGL_SHADER_MANIFEST,
		"webgl.scene",
		{ specialization: variant },
	);
	return {
		specialization: (
			resolved.parameters as { specialization: WebGLSceneVariantDescriptor }
		).specialization,
		identity: resolved.identity,
	};
}

/** @internal Exact sampler-free built-in fallback for a failed ShaderMaterial. */
export function createWebGLShaderMaterialFallbackVariant(
	mode: ShaderTargetMode,
	deformation: WebGLDeformationProfile = WEBGL_STATIC_DEFORMATION_PROFILE,
): WebGLSceneVariantDescriptor {
	const full = WEBGL_FULL_SCENE_VARIANT;
	return normalizeWebGLSceneVariantDescriptor({
		...full,
		output: mode === "mrt" ? "mrt" : "single",
		materialGBuffer: false,
		oit: true,
		scene: {
			shadows: false,
			shadowTransmittance: false,
			clusteredLighting: false,
			sh: false,
			localLightProbes: false,
			irradianceProbeGrid: false,
			reflectionProbes: false,
			environmentSpecular: false,
		},
		material: {
			...full.material,
			clearcoat: false,
			sheen: false,
			iridescence: false,
			anisotropy: false,
			transmission: false,
			baseMap: false,
			metallicRoughnessMap: false,
			specularMap: false,
			specularColorMap: false,
			normalMap: false,
			emissiveMap: false,
			occlusionMap: false,
			clearcoatMap: false,
			clearcoatRoughnessMap: false,
			clearcoatNormalMap: false,
			sheenColorMap: false,
			sheenRoughnessMap: false,
			iridescenceMap: false,
			iridescenceThicknessMap: false,
			anisotropyMap: false,
			transmissionMap: false,
			thicknessMap: false,
			alphaMask: false,
		},
		skinProfile: deformation.skinProfile,
		morphSemanticMask: deformation.morphSemanticMask,
	});
}

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
	materialGBuffer = false,
	deformation: WebGLDeformationProfile = WEBGL_STATIC_DEFORMATION_PROFILE,
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
		skinProfile: deformation.skinProfile,
		morphSemanticMask: deformation.morphSemanticMask,
	});
}

export function resolveWebGLBuiltinDepthVariant(
	material: Material,
	deformation: WebGLDeformationProfile = WEBGL_STATIC_DEFORMATION_PROFILE,
): WebGLSceneDepthVariantDescriptor | null {
	if (material instanceof ShaderMaterial) {
		return null;
	}
	const uniforms = resolveMaterialUniforms(material);
	const alphaMask = material.alphaMode === AlphaMode.Mask;
	return normalizeWebGLSceneDepthVariantDescriptor({
		alphaMask,
		baseMap: alphaMask && !!uniforms.baseMap,
		skinProfile: deformation.skinProfile,
		morphPosition: (deformation.morphSemanticMask & 1) !== 0,
	});
}

export interface WebGLDeformationProfile {
	readonly skinProfile: "static" | "skin4" | "skin8";
	readonly morphSemanticMask: number;
}

export const WEBGL_STATIC_DEFORMATION_PROFILE: WebGLDeformationProfile = {
	skinProfile: "static",
	morphSemanticMask: 0,
};

export function resolveWebGLPacketDeformationProfile(
	packet: DrawPacket,
): WebGLDeformationProfile {
	const geometry = packet.primitive?.geometry;
	if (!geometry) return WEBGL_STATIC_DEFORMATION_PROFILE;
	const skinProfile =
		geometry.joints1 || geometry.weights1 ? "skin8"
		: geometry.joints0 || geometry.weights0 ? "skin4"
		: "static";
	let morphSemanticMask = 0;
	for (const target of (geometry.morphTargets ?? []).slice(0, 8)) {
		if (target.positions) morphSemanticMask |= 1;
		if (target.normals) morphSemanticMask |= 2;
	}
	return { skinProfile, morphSemanticMask };
}

function resolveWebGLSceneMaterialVariant(
	material: Material
): WebGLSceneVariantDescriptor["material"] {
	const uniforms = resolveMaterialUniforms(material);
	const model = resolveMaterialModel(material);
	const isPBR = model === "pbr";
	const pbrMaterial = material instanceof PBRMaterial ? material : null;
	const featureMask = pbrMaterial?.featureMask ?? 0;
	const textureMask = pbrMaterial?.textureMask ?? 0;
	const iridescence = isPBR && resolvePBRFeature(
		pbrMaterial,
		featureMask,
		PBRMaterialFeature.IRIDESCENCE,
		uniforms.iridescence[0] > EPSILON ||
			!!uniforms.iridescenceMap ||
			!!uniforms.iridescenceThicknessMap
	);
	const anisotropy = isPBR && resolvePBRFeature(
		pbrMaterial,
		featureMask,
		PBRMaterialFeature.ANISOTROPY,
		uniforms.anisotropy[0] > EPSILON || !!uniforms.anisotropyMap
	);
	const clearcoat = isPBR && resolvePBRFeature(
		pbrMaterial,
		featureMask,
		PBRMaterialFeature.CLEARCOAT,
		uniforms.clearcoat[0] > EPSILON ||
			!!uniforms.clearcoatMap ||
			!!uniforms.clearcoatRoughnessMap ||
			!!uniforms.clearcoatNormalMap
	);
	const sheen = isPBR && resolvePBRFeature(
		pbrMaterial,
		featureMask,
		PBRMaterialFeature.SHEEN,
		Math.max(uniforms.sheen[0], uniforms.sheen[1], uniforms.sheen[2]) >
			EPSILON || !!uniforms.sheenColorMap || !!uniforms.sheenRoughnessMap
	);
	const transmission = isPBR && resolvePBRFeature(
		pbrMaterial,
		featureMask,
		PBRMaterialFeature.TRANSMISSION,
		uniforms.pbr[3] > EPSILON || !!uniforms.transmissionMap
	);
	const specular = isPBR && resolvePBRFeature(
		pbrMaterial,
		featureMask,
		PBRMaterialFeature.SPECULAR,
		true
	);
	const specularFactorActive = uniforms.specular[3] > EPSILON;
	const specularColorActive = Math.max(
		uniforms.specular[0],
		uniforms.specular[1],
		uniforms.specular[2]
	) > EPSILON;
	return {
		model,
		baseMap: resolvePBRTexture(
			pbrMaterial,
			textureMask,
			PBRMaterialTextureFeature.BASE_COLOR_MAP,
			!!uniforms.baseMap
		),
		metallicRoughnessMap: isPBR &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.METALLIC_ROUGHNESS_MAP,
				!!uniforms.metallicRoughnessMap
			),
		specularMap: specular && specularFactorActive && specularColorActive &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.SPECULAR_MAP,
				!!uniforms.specularMap
			),
		specularColorMap: specular && specularFactorActive && specularColorActive &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.SPECULAR_COLOR_MAP,
				!!uniforms.specularColorMap
			),
		normalMap: isPBR &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.NORMAL_MAP,
				!!uniforms.normalMap
			),
		emissiveMap: resolvePBRTexture(
			pbrMaterial,
			textureMask,
			PBRMaterialTextureFeature.EMISSIVE_MAP,
			!!uniforms.emissiveMap
		),
		occlusionMap: isPBR &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.OCCLUSION_MAP,
				!!uniforms.occlusionMap
			),
		clearcoat,
		clearcoatMap: clearcoat &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.CLEARCOAT_MAP,
				!!uniforms.clearcoatMap
			),
		clearcoatRoughnessMap: clearcoat &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.CLEARCOAT_ROUGHNESS_MAP,
				!!uniforms.clearcoatRoughnessMap
			),
		clearcoatNormalMap: clearcoat &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.CLEARCOAT_NORMAL_MAP,
				!!uniforms.clearcoatNormalMap
			),
		sheen,
		sheenColorMap: sheen &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.SHEEN_COLOR_MAP,
				!!uniforms.sheenColorMap
			),
		sheenRoughnessMap: sheen &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.SHEEN_ROUGHNESS_MAP,
				!!uniforms.sheenRoughnessMap
			),
		iridescence,
		iridescenceMap: iridescence &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.IRIDESCENCE_MAP,
				!!uniforms.iridescenceMap
			),
		iridescenceThicknessMap:
			iridescence &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.IRIDESCENCE_THICKNESS_MAP,
				!!uniforms.iridescenceThicknessMap
			),
		anisotropy,
		anisotropyMap: anisotropy &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.ANISOTROPY_MAP,
				!!uniforms.anisotropyMap
			),
		transmission,
		transmissionMap: transmission &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.TRANSMISSION_MAP,
				!!uniforms.transmissionMap
			),
		thicknessMap: transmission &&
			resolvePBRTexture(
				pbrMaterial,
				textureMask,
				PBRMaterialTextureFeature.THICKNESS_MAP,
				!!uniforms.thicknessMap
			),
		alphaMask: material.alphaMode === AlphaMode.Mask,
	};
}

function resolvePBRFeature(
	material: PBRMaterial | null,
	mask: number,
	bit: PBRMaterialFeature,
	fallback: boolean
): boolean {
	return material ? hasMaskBit(mask, bit) : fallback;
}

function resolvePBRTexture(
	material: PBRMaterial | null,
	mask: number,
	bit: PBRMaterialTextureFeature,
	fallback: boolean
): boolean {
	return material ? hasMaskBit(mask, bit) : fallback;
}

function hasMaskBit(mask: number, bit: number): boolean {
	return (mask & bit) !== 0;
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
