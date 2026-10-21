import {
	AlphaMode,
	type Material,
} from "../../materials/Material";
import {
	ShaderMaterial,
	type ShaderTargetMode,
} from "../../materials/ShaderMaterial";
import type { IPrimitiveGeometry } from "../../core/types";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import {
	WEBGL_MAX_MORPH_TARGETS,
	WEBGL_MORPH_NORMAL_BIT,
	WEBGL_MORPH_POSITION_BIT,
	type WebGLSkinProfile,
} from "./WebGLGeometryRegistry";
import {
	WEBGL_FULL_SCENE_VARIANT,
	getWebGLSceneDepthVariantKey,
	getWebGLSceneVariantKey,
	normalizeWebGLSceneDepthVariantDescriptor,
	normalizeWebGLSceneVariantDescriptor,
	type WebGLSceneDepthVariantDescriptor,
	type WebGLSceneVariantDescriptor,
} from "../../shaders/webgl/sceneVariants";
import type { WebGLLightState } from "./WebGLLightCollector";
import {
	resolveWebGLMaterialState,
	type WebGLResolvedMaterialState,
} from "./WebGLMaterialState";

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
	materialState: WebGLResolvedMaterialState = resolveWebGLMaterialState(material),
	resolvedMaterialVariant?: WebGLSceneVariantDescriptor["material"],
): WebGLSceneVariantDescriptor | null {
	if (material instanceof ShaderMaterial) {
		return null;
	}
	const materialVariant =
		resolvedMaterialVariant ?? resolveWebGLSceneMaterialVariant(material, materialState);
	const isLit =
		materialVariant.model === "pbr" ||
		materialVariant.model === "phong" ||
		materialVariant.model === "flat" ||
		materialVariant.model === "full";
	const lightState = environment.lightState;
	const features = context.features as Partial<FrameContext["features"]> | undefined;
	const hasShadows =
		isLit &&
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
	materialState: WebGLResolvedMaterialState = resolveWebGLMaterialState(material),
): WebGLSceneDepthVariantDescriptor | null {
	if (material instanceof ShaderMaterial) {
		return null;
	}
	const alphaMask = material.alphaMode === AlphaMode.Mask;
	return normalizeWebGLSceneDepthVariantDescriptor({
		alphaMask,
		baseMap: alphaMask && !!materialState.common.baseMap.texture,
		skinProfile: deformation.skinProfile,
		morphPosition: (deformation.morphSemanticMask & 1) !== 0,
	});
}

export interface WebGLDeformationProfile {
	readonly skinProfile: WebGLSkinProfile;
	readonly morphSemanticMask: number;
}

export const WEBGL_STATIC_DEFORMATION_PROFILE: WebGLDeformationProfile = {
	skinProfile: "static",
	morphSemanticMask: 0,
};

interface CachedDeformationProfile {
	geometryVersion: number;
	profile: WebGLDeformationProfile;
}

// Profiles derive only from geometry content, so they are memoized per
// primitive and invalidated by `geometryVersion`, mirroring the geometry
// registry cache contract.
const deformationProfileCache = new WeakMap<
	object,
	CachedDeformationProfile
>();

/**
 * Derives the deformation profile from raw geometry content.
 *
 * @internal Owned by the WebGL scene program subsystem.
 */
export function resolveWebGLGeometryDeformationProfile(
	geometry: IPrimitiveGeometry,
): WebGLDeformationProfile {
	const skinProfile =
		geometry.joints1 || geometry.weights1 ? "skin8"
		: geometry.joints0 || geometry.weights0 ? "skin4"
		: "static";
	let morphSemanticMask = 0;
	const targets = geometry.morphTargets;
	if (targets) {
		const count = Math.min(targets.length, WEBGL_MAX_MORPH_TARGETS);
		for (let index = 0; index < count; index++) {
			const target = targets[index];
			if (target.positions) morphSemanticMask |= WEBGL_MORPH_POSITION_BIT;
			if (target.normals) morphSemanticMask |= WEBGL_MORPH_NORMAL_BIT;
		}
	}
	return { skinProfile, morphSemanticMask };
}

/**
 * Resolves a packet's deformation profile from its primitive geometry,
 * returning the memoized profile until the primitive's `geometryVersion`
 * changes.
 *
 * @internal Owned by the WebGL scene program subsystem; warmup and draw-time
 * resolution must share this resolver so variants stay identical.
 */
export function resolveWebGLPacketDeformationProfile(
	packet: DrawPacket,
): WebGLDeformationProfile {
	const binding = packet.submission.geometry;
	const geometry = binding.data;
	const geometryVersion = binding.version;
	const cached = deformationProfileCache.get(binding.resourceKey);
	if (cached && cached.geometryVersion === geometryVersion) {
		return cached.profile;
	}
	const profile = resolveWebGLGeometryDeformationProfile(geometry);
	deformationProfileCache.set(binding.resourceKey, { geometryVersion, profile });
	return profile;
}

export function resolveWebGLSceneMaterialVariant(
	material: Material,
	state: WebGLResolvedMaterialState = resolveWebGLMaterialState(material),
): WebGLSceneVariantDescriptor["material"] {
	const model = state.shadingFamily;
	const isPBR = model === "pbr";
	const pbr = state.shadingFamily === "pbr" ? state.lighting : null;
	const features = pbr?.features;
	const textures = features?.activeTextures;
	return {
		model,
		baseMap: isPBR ?
			(textures?.baseColorMap ?? false)
		: !!state.common.baseMap.texture,
		metallicRoughnessMap: isPBR &&
			(textures?.metallicRoughnessMap ?? false),
		specularMap: textures?.specularMap ?? false,
		specularColorMap: textures?.specularColorMap ?? false,
		normalMap: isPBR &&
			(textures?.normalMap ?? false),
		emissiveMap: isPBR ?
			(textures?.emissiveMap ?? false)
		: !!state.common.emissiveMap.texture,
		occlusionMap: isPBR &&
			(textures?.occlusionMap ?? false),
		clearcoat: features?.clearcoat ?? false,
		clearcoatMap: textures?.clearcoatMap ?? false,
		clearcoatRoughnessMap: textures?.clearcoatRoughnessMap ?? false,
		clearcoatNormalMap: textures?.clearcoatNormalMap ?? false,
		sheen: features?.sheen ?? false,
		sheenColorMap: textures?.sheenColorMap ?? false,
		sheenRoughnessMap: textures?.sheenRoughnessMap ?? false,
		iridescence: features?.iridescence ?? false,
		iridescenceMap: textures?.iridescenceMap ?? false,
		iridescenceThicknessMap:
			textures?.iridescenceThicknessMap ?? false,
		anisotropy: features?.anisotropy ?? false,
		anisotropyMap: textures?.anisotropyMap ?? false,
		transmission: features?.transmission ?? false,
		transmissionMap: textures?.transmissionMap ?? false,
		thicknessMap: textures?.thicknessMap ?? false,
		alphaMask: material.alphaMode === AlphaMode.Mask,
	};
}
