/**
 * @internal WebGL built-in scene shader output target encoded into source and
 * program cache keys.
 */
export type WebGLSceneOutputMode = "single" | "mrt";

/**
 * @internal WebGL built-in material branch encoded into scene shader variants.
 * `full` preserves the legacy unpruned scene source for compatibility callers.
 */
export type WebGLSceneMaterialModel = "unlit" | "legacy" | "pbr" | "full";
export type WebGLSceneSkinProfile = "static" | "skin4" | "skin8";

/**
 * @internal Scene-wide feature bits that select WebGL built-in GLSL parts.
 */
export interface WebGLSceneFeatureVariant {
	shadows: boolean;
	shadowTransmittance: boolean;
	clusteredLighting: boolean;
	sh: boolean;
	localLightProbes: boolean;
	irradianceProbeGrid: boolean;
	reflectionProbes: boolean;
	environmentSpecular: boolean;
}

/**
 * @internal Built-in material feature bits that prune WebGL scene shader code.
 */
export interface WebGLSceneMaterialVariant {
	model: WebGLSceneMaterialModel;
	baseMap: boolean;
	metallicRoughnessMap: boolean;
	specularMap: boolean;
	specularColorMap: boolean;
	normalMap: boolean;
	emissiveMap: boolean;
	occlusionMap: boolean;
	clearcoat: boolean;
	clearcoatMap: boolean;
	clearcoatRoughnessMap: boolean;
	clearcoatNormalMap: boolean;
	sheen: boolean;
	sheenColorMap: boolean;
	sheenRoughnessMap: boolean;
	iridescence: boolean;
	iridescenceMap: boolean;
	iridescenceThicknessMap: boolean;
	anisotropy: boolean;
	anisotropyMap: boolean;
	transmission: boolean;
	transmissionMap: boolean;
	thicknessMap: boolean;
	alphaMask: boolean;
}

/**
 * @internal Complete WebGL built-in scene shader variant descriptor.
 */
export interface WebGLSceneVariantDescriptor {
	output: WebGLSceneOutputMode;
	materialGBuffer: boolean;
	oit: boolean;
	scene: WebGLSceneFeatureVariant;
	material: WebGLSceneMaterialVariant;
	skinProfile: WebGLSceneSkinProfile;
	morphSemanticMask: number;
}

/**
 * @internal Limited WebGL built-in depth pre-pass variant descriptor.
 */
export interface WebGLSceneDepthVariantDescriptor {
	alphaMask: boolean;
	baseMap: boolean;
	skinProfile: WebGLSceneSkinProfile;
	morphPosition: boolean;
}

export const WEBGL_FULL_SCENE_VARIANT: WebGLSceneVariantDescriptor = {
	output: "mrt",
	materialGBuffer: false,
	oit: true,
	scene: {
		shadows: true,
		shadowTransmittance: true,
		clusteredLighting: true,
		sh: true,
		localLightProbes: true,
		irradianceProbeGrid: true,
		reflectionProbes: true,
		environmentSpecular: true,
	},
	material: {
		model: "full",
		baseMap: true,
		metallicRoughnessMap: true,
		specularMap: true,
		specularColorMap: true,
		normalMap: true,
		emissiveMap: true,
		occlusionMap: true,
		clearcoat: true,
		clearcoatMap: true,
		clearcoatRoughnessMap: true,
		clearcoatNormalMap: true,
		sheen: true,
		sheenColorMap: true,
		sheenRoughnessMap: true,
		iridescence: true,
		iridescenceMap: true,
		iridescenceThicknessMap: true,
		anisotropy: true,
		anisotropyMap: true,
		transmission: true,
		transmissionMap: true,
		thicknessMap: true,
		alphaMask: true,
	},
	skinProfile: "skin8",
	morphSemanticMask: 3,
};

export const WEBGL_OPAQUE_DEPTH_VARIANT: WebGLSceneDepthVariantDescriptor = {
	alphaMask: false,
	baseMap: false,
	skinProfile: "static",
	morphPosition: false,
};

export const WEBGL_ALPHA_MAP_DEPTH_VARIANT: WebGLSceneDepthVariantDescriptor = {
	alphaMask: true,
	baseMap: true,
	skinProfile: "static",
	morphPosition: false,
};

export function normalizeWebGLSceneVariantDescriptor(
	variant?: WebGLSceneVariantDescriptor
): WebGLSceneVariantDescriptor {
	const base = WEBGL_FULL_SCENE_VARIANT;
	if (!variant) {
		return cloneWebGLSceneVariantDescriptor(base);
	}
	return {
		output: variant.output === "single" ? "single" : "mrt",
		materialGBuffer: variant.output !== "single" && variant.materialGBuffer === true,
		oit: variant.oit === true,
		scene: {
			shadows: variant.scene.shadows === true,
			shadowTransmittance: variant.scene.shadowTransmittance === true,
			clusteredLighting: variant.scene.clusteredLighting === true,
			sh: variant.scene.sh === true,
			localLightProbes: variant.scene.localLightProbes === true,
			irradianceProbeGrid: variant.scene.irradianceProbeGrid === true,
			reflectionProbes: variant.scene.reflectionProbes === true,
			environmentSpecular: variant.scene.environmentSpecular === true,
		},
		material: {
			model: normalizeWebGLSceneMaterialModel(variant.material.model),
			baseMap: variant.material.baseMap === true,
			metallicRoughnessMap:
				variant.material.metallicRoughnessMap === true,
			specularMap: variant.material.specularMap === true,
			specularColorMap: variant.material.specularColorMap === true,
			normalMap: variant.material.normalMap === true,
			emissiveMap: variant.material.emissiveMap === true,
			occlusionMap: variant.material.occlusionMap === true,
			clearcoat: variant.material.clearcoat === true,
			clearcoatMap: variant.material.clearcoatMap === true,
			clearcoatRoughnessMap:
				variant.material.clearcoatRoughnessMap === true,
			clearcoatNormalMap: variant.material.clearcoatNormalMap === true,
			sheen: variant.material.sheen === true,
			sheenColorMap: variant.material.sheenColorMap === true,
			sheenRoughnessMap: variant.material.sheenRoughnessMap === true,
			iridescence: variant.material.iridescence === true,
			iridescenceMap: variant.material.iridescenceMap === true,
			iridescenceThicknessMap:
				variant.material.iridescenceThicknessMap === true,
			anisotropy: variant.material.anisotropy === true,
			anisotropyMap: variant.material.anisotropyMap === true,
			transmission: variant.material.transmission === true,
			transmissionMap: variant.material.transmissionMap === true,
			thicknessMap: variant.material.thicknessMap === true,
			alphaMask: variant.material.alphaMask === true,
		},
		skinProfile: normalizeSkinProfile(variant.skinProfile),
		morphSemanticMask: Math.max(0, Math.floor(variant.morphSemanticMask ?? 0)) & 3,
	};
}

export function getWebGLSceneVariantKey(
	variant?: WebGLSceneVariantDescriptor
): string {
	const normalized = normalizeWebGLSceneVariantDescriptor(variant);
	const scene = normalized.scene;
	const material = normalized.material;
	return [
		`out:${normalized.output}`,
		`gbuf:${bit(normalized.materialGBuffer)}`,
		`oit:${bit(normalized.oit)}`,
		`shd:${bit(scene.shadows)}`,
		`shdt:${bit(scene.shadowTransmittance)}`,
		`cl:${bit(scene.clusteredLighting)}`,
		`sh:${bit(scene.sh)}`,
		`lp:${bit(scene.localLightProbes)}`,
		`grid:${bit(scene.irradianceProbeGrid)}`,
		`rp:${bit(scene.reflectionProbes)}`,
		`env:${bit(scene.environmentSpecular)}`,
		`mat:${material.model}`,
		`base:${bit(material.baseMap)}`,
		`mr:${bit(material.metallicRoughnessMap)}`,
		`spm:${bit(material.specularMap)}`,
		`spcm:${bit(material.specularColorMap)}`,
		`norm:${bit(material.normalMap)}`,
		`emis:${bit(material.emissiveMap)}`,
		`occ:${bit(material.occlusionMap)}`,
		`cc:${bit(material.clearcoat)}`,
		`ccm:${bit(material.clearcoatMap)}`,
		`ccrm:${bit(material.clearcoatRoughnessMap)}`,
		`ccnm:${bit(material.clearcoatNormalMap)}`,
		`sheen:${bit(material.sheen)}`,
		`shcm:${bit(material.sheenColorMap)}`,
		`shrm:${bit(material.sheenRoughnessMap)}`,
		`iri:${bit(material.iridescence)}`,
		`irim:${bit(material.iridescenceMap)}`,
		`irit:${bit(material.iridescenceThicknessMap)}`,
		`ani:${bit(material.anisotropy)}`,
		`anim:${bit(material.anisotropyMap)}`,
		`trans:${bit(material.transmission)}`,
		`transm:${bit(material.transmissionMap)}`,
		`thickm:${bit(material.thicknessMap)}`,
		`mask:${bit(material.alphaMask)}`,
		`skin:${normalized.skinProfile}`,
		`morph:${normalized.morphSemanticMask}`,
	].join("|");
}

export function getWebGLSceneDepthVariantKey(
	variant?: WebGLSceneDepthVariantDescriptor
): string {
	const normalized = normalizeWebGLSceneDepthVariantDescriptor(variant);
	return [
		`mask:${bit(normalized.alphaMask)}`,
		`base:${bit(normalized.baseMap)}`,
		`skin:${normalized.skinProfile}`,
		`morphp:${bit(normalized.morphPosition)}`,
	].join("|");
}

export function normalizeWebGLSceneDepthVariantDescriptor(
	variant?: WebGLSceneDepthVariantDescriptor
): WebGLSceneDepthVariantDescriptor {
	if (!variant) {
		return { ...WEBGL_ALPHA_MAP_DEPTH_VARIANT };
	}
	const alphaMask = variant.alphaMask === true;
	return {
		alphaMask,
		baseMap: alphaMask && variant.baseMap === true,
		skinProfile: normalizeSkinProfile(variant.skinProfile),
		morphPosition: variant.morphPosition === true,
	};
}

function cloneWebGLSceneVariantDescriptor(
	variant: WebGLSceneVariantDescriptor
): WebGLSceneVariantDescriptor {
	return {
		output: variant.output,
		materialGBuffer: variant.materialGBuffer,
		oit: variant.oit,
		scene: { ...variant.scene },
		material: { ...variant.material },
		skinProfile: variant.skinProfile,
		morphSemanticMask: variant.morphSemanticMask,
	};
}

function normalizeSkinProfile(value: WebGLSceneSkinProfile): WebGLSceneSkinProfile {
	return value === "skin4" || value === "skin8" ? value : "static";
}

function normalizeWebGLSceneMaterialModel(
	model: WebGLSceneMaterialModel
): WebGLSceneMaterialModel {
	if (
		model === "unlit" ||
		model === "legacy" ||
		model === "pbr" ||
		model === "full"
	) {
		return model;
	}
	return "pbr";
}

function bit(value: boolean): 0 | 1 {
	return value ? 1 : 0;
}
