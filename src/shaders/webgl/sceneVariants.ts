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
	normalMap: boolean;
	emissiveMap: boolean;
	occlusionMap: boolean;
	iridescence: boolean;
	iridescenceMap: boolean;
	iridescenceThicknessMap: boolean;
	anisotropy: boolean;
	anisotropyMap: boolean;
	transmission: boolean;
	alphaMask: boolean;
}

/**
 * @internal Complete WebGL built-in scene shader variant descriptor.
 */
export interface WebGLSceneVariantDescriptor {
	output: WebGLSceneOutputMode;
	oit: boolean;
	scene: WebGLSceneFeatureVariant;
	material: WebGLSceneMaterialVariant;
}

/**
 * @internal Limited WebGL built-in depth pre-pass variant descriptor.
 */
export interface WebGLSceneDepthVariantDescriptor {
	alphaMask: boolean;
	baseMap: boolean;
}

export const WEBGL_FULL_SCENE_VARIANT: WebGLSceneVariantDescriptor = {
	output: "mrt",
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
		normalMap: true,
		emissiveMap: true,
		occlusionMap: true,
		iridescence: true,
		iridescenceMap: true,
		iridescenceThicknessMap: true,
		anisotropy: true,
		anisotropyMap: true,
		transmission: true,
		alphaMask: true,
	},
};

export const WEBGL_OPAQUE_DEPTH_VARIANT: WebGLSceneDepthVariantDescriptor = {
	alphaMask: false,
	baseMap: false,
};

export const WEBGL_ALPHA_MAP_DEPTH_VARIANT: WebGLSceneDepthVariantDescriptor = {
	alphaMask: true,
	baseMap: true,
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
			normalMap: variant.material.normalMap === true,
			emissiveMap: variant.material.emissiveMap === true,
			occlusionMap: variant.material.occlusionMap === true,
			iridescence: variant.material.iridescence === true,
			iridescenceMap: variant.material.iridescenceMap === true,
			iridescenceThicknessMap:
				variant.material.iridescenceThicknessMap === true,
			anisotropy: variant.material.anisotropy === true,
			anisotropyMap: variant.material.anisotropyMap === true,
			transmission: variant.material.transmission === true,
			alphaMask: variant.material.alphaMask === true,
		},
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
		`norm:${bit(material.normalMap)}`,
		`emis:${bit(material.emissiveMap)}`,
		`occ:${bit(material.occlusionMap)}`,
		`iri:${bit(material.iridescence)}`,
		`irim:${bit(material.iridescenceMap)}`,
		`irit:${bit(material.iridescenceThicknessMap)}`,
		`ani:${bit(material.anisotropy)}`,
		`anim:${bit(material.anisotropyMap)}`,
		`trans:${bit(material.transmission)}`,
		`mask:${bit(material.alphaMask)}`,
	].join("|");
}

export function getWebGLSceneDepthVariantKey(
	variant?: WebGLSceneDepthVariantDescriptor
): string {
	const normalized = normalizeWebGLSceneDepthVariantDescriptor(variant);
	return [
		`mask:${bit(normalized.alphaMask)}`,
		`base:${bit(normalized.baseMap)}`,
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
	};
}

function cloneWebGLSceneVariantDescriptor(
	variant: WebGLSceneVariantDescriptor
): WebGLSceneVariantDescriptor {
	return {
		output: variant.output,
		oit: variant.oit,
		scene: { ...variant.scene },
		material: { ...variant.material },
	};
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
