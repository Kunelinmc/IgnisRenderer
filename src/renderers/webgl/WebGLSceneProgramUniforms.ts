type WebGLSceneUniformLocation = WebGLUniformLocation | null;

function createUniformLookup(
	gl: WebGL2RenderingContext,
	program: WebGLProgram
): (name: string) => WebGLSceneUniformLocation {
	return (name: string): WebGLSceneUniformLocation =>
		gl.getUniformLocation(program, name);
}

export class WebGLSceneTransformUniforms {
	public readonly model: WebGLSceneUniformLocation;
	public readonly viewMatrix: WebGLSceneUniformLocation;
	public readonly viewProjection: WebGLSceneUniformLocation;
	public readonly normalMatrix: WebGLSceneUniformLocation;
	public readonly taaJitter: WebGLSceneUniformLocation;
	public readonly prevViewProjection: WebGLSceneUniformLocation;
	public readonly prevModel: WebGLSceneUniformLocation;

	public constructor(gl: WebGL2RenderingContext, program: WebGLProgram) {
		const get = createUniformLookup(gl, program);
		this.model = get("uModel");
		this.viewMatrix = get("uViewMatrix");
		this.viewProjection = get("uViewProjection");
		this.normalMatrix = get("uNormalMatrix");
		this.taaJitter = get("uTaaJitter");
		this.prevViewProjection = get("uPrevViewProjection");
		this.prevModel = get("uPrevModel");
	}
}

export class WebGLSceneFrameUniforms {
	public readonly cameraPosition: WebGLSceneUniformLocation;

	public constructor(gl: WebGL2RenderingContext, program: WebGLProgram) {
		const get = createUniformLookup(gl, program);
		this.cameraPosition = get("uCameraPosition");
	}
}

export class WebGLSceneMaterialUniforms {
	public readonly doubleSided: WebGLSceneUniformLocation;
	public readonly shadingModel: WebGLSceneUniformLocation;
	public readonly baseColor: WebGLSceneUniformLocation;
	public readonly emissive: WebGLSceneUniformLocation;
	public readonly pbr: WebGLSceneUniformLocation;
	public readonly transmissionVolume: WebGLSceneUniformLocation;
	public readonly iridescence: WebGLSceneUniformLocation;
	public readonly attenuationColor: WebGLSceneUniformLocation;
	public readonly anisotropy: WebGLSceneUniformLocation;
	public readonly phong: WebGLSceneUniformLocation;
	public readonly alpha: WebGLSceneUniformLocation;
	public readonly oitPassMode: WebGLSceneUniformLocation;

	public constructor(gl: WebGL2RenderingContext, program: WebGLProgram) {
		const get = createUniformLookup(gl, program);
		this.doubleSided = get("uDoubleSided");
		this.shadingModel = get("uShadingModel");
		this.baseColor = get("uBaseColor");
		this.emissive = get("uEmissive");
		this.pbr = get("uPBR");
		this.transmissionVolume = get("uTransmissionVolume");
		this.iridescence = get("uIridescence");
		this.attenuationColor = get("uAttenuationColor");
		this.anisotropy = get("uAnisotropy");
		this.phong = get("uPhong");
		this.alpha = get("uAlpha");
		this.oitPassMode = get("uOITPassMode");
	}
}

export class WebGLSceneMaterialTextureUniforms {
	public readonly baseMap: WebGLSceneUniformLocation;
	public readonly hasBaseMap: WebGLSceneUniformLocation;
	public readonly baseMapIsLinear: WebGLSceneUniformLocation;
	public readonly baseMapUV: WebGLSceneUniformLocation;
	public readonly baseMapTransformA: WebGLSceneUniformLocation;
	public readonly baseMapTransformB: WebGLSceneUniformLocation;
	public readonly metallicRoughnessMap: WebGLSceneUniformLocation;
	public readonly hasMetallicRoughnessMap: WebGLSceneUniformLocation;
	public readonly metallicRoughnessMapUV: WebGLSceneUniformLocation;
	public readonly metallicRoughnessMapTransformA: WebGLSceneUniformLocation;
	public readonly metallicRoughnessMapTransformB: WebGLSceneUniformLocation;
	public readonly normalMap: WebGLSceneUniformLocation;
	public readonly hasNormalMap: WebGLSceneUniformLocation;
	public readonly normalMapUV: WebGLSceneUniformLocation;
	public readonly normalMapTransformA: WebGLSceneUniformLocation;
	public readonly normalMapTransformB: WebGLSceneUniformLocation;
	public readonly normalScale: WebGLSceneUniformLocation;
	public readonly emissiveMap: WebGLSceneUniformLocation;
	public readonly hasEmissiveMap: WebGLSceneUniformLocation;
	public readonly emissiveMapIsLinear: WebGLSceneUniformLocation;
	public readonly emissiveMapUV: WebGLSceneUniformLocation;
	public readonly emissiveMapTransformA: WebGLSceneUniformLocation;
	public readonly emissiveMapTransformB: WebGLSceneUniformLocation;
	public readonly occlusionMap: WebGLSceneUniformLocation;
	public readonly hasOcclusionMap: WebGLSceneUniformLocation;
	public readonly occlusionMapUV: WebGLSceneUniformLocation;
	public readonly occlusionMapTransformA: WebGLSceneUniformLocation;
	public readonly occlusionMapTransformB: WebGLSceneUniformLocation;
	public readonly occlusionStrength: WebGLSceneUniformLocation;
	public readonly iridescenceMap: WebGLSceneUniformLocation;
	public readonly hasIridescenceMap: WebGLSceneUniformLocation;
	public readonly iridescenceMapUV: WebGLSceneUniformLocation;
	public readonly iridescenceMapTransformA: WebGLSceneUniformLocation;
	public readonly iridescenceMapTransformB: WebGLSceneUniformLocation;
	public readonly iridescenceThicknessMap: WebGLSceneUniformLocation;
	public readonly hasIridescenceThicknessMap: WebGLSceneUniformLocation;
	public readonly iridescenceThicknessMapUV: WebGLSceneUniformLocation;
	public readonly iridescenceThicknessMapTransformA: WebGLSceneUniformLocation;
	public readonly iridescenceThicknessMapTransformB: WebGLSceneUniformLocation;
	public readonly hasAnisotropyMap: WebGLSceneUniformLocation;
	public readonly anisotropyMapUV: WebGLSceneUniformLocation;
	public readonly anisotropyMapTransformA: WebGLSceneUniformLocation;
	public readonly anisotropyMapTransformB: WebGLSceneUniformLocation;

	public constructor(gl: WebGL2RenderingContext, program: WebGLProgram) {
		const get = createUniformLookup(gl, program);
		this.baseMap = get("uBaseMap");
		this.hasBaseMap = get("uHasBaseMap");
		this.baseMapIsLinear = get("uBaseMapIsLinear");
		this.baseMapUV = get("uBaseMapUV");
		this.baseMapTransformA = get("uBaseMapTransformA");
		this.baseMapTransformB = get("uBaseMapTransformB");
		this.metallicRoughnessMap = get("uMetallicRoughnessMap");
		this.hasMetallicRoughnessMap = get("uHasMetallicRoughnessMap");
		this.metallicRoughnessMapUV = get("uMetallicRoughnessMapUV");
		this.metallicRoughnessMapTransformA = get(
			"uMetallicRoughnessMapTransformA"
		);
		this.metallicRoughnessMapTransformB = get(
			"uMetallicRoughnessMapTransformB"
		);
		this.normalMap = get("uNormalMap");
		this.hasNormalMap = get("uHasNormalMap");
		this.normalMapUV = get("uNormalMapUV");
		this.normalMapTransformA = get("uNormalMapTransformA");
		this.normalMapTransformB = get("uNormalMapTransformB");
		this.normalScale = get("uNormalScale");
		this.emissiveMap = get("uEmissiveMap");
		this.hasEmissiveMap = get("uHasEmissiveMap");
		this.emissiveMapIsLinear = get("uEmissiveMapIsLinear");
		this.emissiveMapUV = get("uEmissiveMapUV");
		this.emissiveMapTransformA = get("uEmissiveMapTransformA");
		this.emissiveMapTransformB = get("uEmissiveMapTransformB");
		this.occlusionMap = get("uOcclusionMap");
		this.hasOcclusionMap = get("uHasOcclusionMap");
		this.occlusionMapUV = get("uOcclusionMapUV");
		this.occlusionMapTransformA = get("uOcclusionMapTransformA");
		this.occlusionMapTransformB = get("uOcclusionMapTransformB");
		this.occlusionStrength = get("uOcclusionStrength");
		this.iridescenceMap = get("uIridescenceMap");
		this.hasIridescenceMap = get("uHasIridescenceMap");
		this.iridescenceMapUV = get("uIridescenceMapUV");
		this.iridescenceMapTransformA = get("uIridescenceMapTransformA");
		this.iridescenceMapTransformB = get("uIridescenceMapTransformB");
		this.iridescenceThicknessMap = get("uIridescenceThicknessMap");
		this.hasIridescenceThicknessMap = get("uHasIridescenceThicknessMap");
		this.iridescenceThicknessMapUV = get("uIridescenceThicknessMapUV");
		this.iridescenceThicknessMapTransformA = get(
			"uIridescenceThicknessMapTransformA"
		);
		this.iridescenceThicknessMapTransformB = get(
			"uIridescenceThicknessMapTransformB"
		);
		this.hasAnisotropyMap = get("uHasAnisotropyMap");
		this.anisotropyMapUV = get("uAnisotropyMapUV");
		this.anisotropyMapTransformA = get("uAnisotropyMapTransformA");
		this.anisotropyMapTransformB = get("uAnisotropyMapTransformB");
	}
}

export class WebGLSceneEnvironmentUniforms {
	public readonly envSpecularMap: WebGLSceneUniformLocation;
	public readonly hasEnvSpecularMap: WebGLSceneUniformLocation;
	public readonly envSpecularMapIsLinear: WebGLSceneUniformLocation;
	public readonly envSpecularMaxMipLevel: WebGLSceneUniformLocation;
	public readonly envSpecularFallbackMap: WebGLSceneUniformLocation;
	public readonly hasEnvSpecularFallbackMap: WebGLSceneUniformLocation;
	public readonly envSpecularFallbackMapIsLinear: WebGLSceneUniformLocation;
	public readonly envSpecularFallbackMaxMipLevel: WebGLSceneUniformLocation;
	public readonly brdfLUT: WebGLSceneUniformLocation;
	public readonly localLightProbeCount: WebGLSceneUniformLocation;
	public readonly localLightProbeWorldToProbeRow0: WebGLSceneUniformLocation;
	public readonly localLightProbeWorldToProbeRow1: WebGLSceneUniformLocation;
	public readonly localLightProbeWorldToProbeRow2: WebGLSceneUniformLocation;
	public readonly localLightProbeDataA: WebGLSceneUniformLocation;
	public readonly localLightProbeDataB: WebGLSceneUniformLocation;
	public readonly localLightProbeCoeffs: WebGLSceneUniformLocation;
	public readonly localLightProbeCoeffsSize: WebGLSceneUniformLocation;
	public readonly reflectionProbeCount: WebGLSceneUniformLocation;
	public readonly reflectionProbeWorldToProbeRow0: WebGLSceneUniformLocation;
	public readonly reflectionProbeWorldToProbeRow1: WebGLSceneUniformLocation;
	public readonly reflectionProbeWorldToProbeRow2: WebGLSceneUniformLocation;
	public readonly reflectionProbeProbeToWorldRow0: WebGLSceneUniformLocation;
	public readonly reflectionProbeProbeToWorldRow1: WebGLSceneUniformLocation;
	public readonly reflectionProbeProbeToWorldRow2: WebGLSceneUniformLocation;
	public readonly reflectionProbeDataA: WebGLSceneUniformLocation;
	public readonly reflectionProbeDataB: WebGLSceneUniformLocation;
	public readonly reflectionProbeDataC: WebGLSceneUniformLocation;

	public constructor(gl: WebGL2RenderingContext, program: WebGLProgram) {
		const get = createUniformLookup(gl, program);
		this.envSpecularMap = get("uEnvSpecularMap");
		this.hasEnvSpecularMap = get("uHasEnvSpecularMap");
		this.envSpecularMapIsLinear = get("uEnvSpecularMapIsLinear");
		this.envSpecularMaxMipLevel = get("uEnvSpecularMaxMipLevel");
		this.envSpecularFallbackMap = get("uEnvSpecularFallbackMap");
		this.hasEnvSpecularFallbackMap = get("uHasEnvSpecularFallbackMap");
		this.envSpecularFallbackMapIsLinear = get(
			"uEnvSpecularFallbackMapIsLinear"
		);
		this.envSpecularFallbackMaxMipLevel = get(
			"uEnvSpecularFallbackMaxMipLevel"
		);
		this.brdfLUT = get("uBrdfLUT");
		this.localLightProbeCount = get("uLocalLightProbeCount");
		this.localLightProbeWorldToProbeRow0 = get(
			"uLocalLightProbeWorldToProbeRow0[0]"
		);
		this.localLightProbeWorldToProbeRow1 = get(
			"uLocalLightProbeWorldToProbeRow1[0]"
		);
		this.localLightProbeWorldToProbeRow2 = get(
			"uLocalLightProbeWorldToProbeRow2[0]"
		);
		this.localLightProbeDataA = get("uLocalLightProbeDataA[0]");
		this.localLightProbeDataB = get("uLocalLightProbeDataB[0]");
		this.localLightProbeCoeffs = get("uLocalLightProbeCoeffs");
		this.localLightProbeCoeffsSize = get("uLocalLightProbeCoeffsSize");
		this.reflectionProbeCount = get("uReflectionProbeCount");
		this.reflectionProbeWorldToProbeRow0 = get(
			"uReflectionProbeWorldToProbeRow0[0]"
		);
		this.reflectionProbeWorldToProbeRow1 = get(
			"uReflectionProbeWorldToProbeRow1[0]"
		);
		this.reflectionProbeWorldToProbeRow2 = get(
			"uReflectionProbeWorldToProbeRow2[0]"
		);
		this.reflectionProbeProbeToWorldRow0 = get(
			"uReflectionProbeProbeToWorldRow0[0]"
		);
		this.reflectionProbeProbeToWorldRow1 = get(
			"uReflectionProbeProbeToWorldRow1[0]"
		);
		this.reflectionProbeProbeToWorldRow2 = get(
			"uReflectionProbeProbeToWorldRow2[0]"
		);
		this.reflectionProbeDataA = get("uReflectionProbeDataA[0]");
		this.reflectionProbeDataB = get("uReflectionProbeDataB[0]");
		this.reflectionProbeDataC = get("uReflectionProbeDataC[0]");
	}
}

export class WebGLSceneLightUniforms {
	public readonly ambientColor: WebGLSceneUniformLocation;
	public readonly enableLighting: WebGLSceneUniformLocation;
	public readonly enableSH: WebGLSceneUniformLocation;
	public readonly dirLightCount: WebGLSceneUniformLocation;
	public readonly dirLightDirection: WebGLSceneUniformLocation;
	public readonly dirLightColor: WebGLSceneUniformLocation;
	public readonly pointLightCount: WebGLSceneUniformLocation;
	public readonly pointLightPositionRange: WebGLSceneUniformLocation;
	public readonly pointLightColor: WebGLSceneUniformLocation;
	public readonly spotLightCount: WebGLSceneUniformLocation;
	public readonly spotLightPositionRange: WebGLSceneUniformLocation;
	public readonly spotLightDirectionOuter: WebGLSceneUniformLocation;
	public readonly spotLightColorInner: WebGLSceneUniformLocation;
	public readonly shAmbientCoeffs: WebGLSceneUniformLocation;
	public readonly shCoeffsSize: WebGLSceneUniformLocation;

	public constructor(gl: WebGL2RenderingContext, program: WebGLProgram) {
		const get = createUniformLookup(gl, program);
		this.ambientColor = get("uAmbientColor");
		this.enableLighting = get("uEnableLighting");
		this.enableSH = get("uEnableSH");
		this.dirLightCount = get("uDirLightCount");
		this.dirLightDirection = get("uDirLightDirection");
		this.dirLightColor = get("uDirLightColor");
		this.pointLightCount = get("uPointLightCount");
		this.pointLightPositionRange = get("uPointLightPositionRange");
		this.pointLightColor = get("uPointLightColor");
		this.spotLightCount = get("uSpotLightCount");
		this.spotLightPositionRange = get("uSpotLightPositionRange");
		this.spotLightDirectionOuter = get("uSpotLightDirectionOuter");
		this.spotLightColorInner = get("uSpotLightColorInner");
		this.shAmbientCoeffs = get("uSHAmbientCoeffs[0]");
		this.shCoeffsSize = get("uSHCoeffsSize");
	}
}

export class WebGLSceneShadowUniforms {
	public readonly enableShadows: WebGLSceneUniformLocation;
	public readonly shadowAtlas: WebGLSceneUniformLocation;
	public readonly shadowTransmittanceAtlas: WebGLSceneUniformLocation;
	public readonly shadowTransmittanceAtlasAvailable: WebGLSceneUniformLocation;
	public readonly particleShadowVolumeAtlas: WebGLSceneUniformLocation;
	public readonly particleShadowVolumeAtlasSize: WebGLSceneUniformLocation;
	public readonly particleShadowVolumeGridSize: WebGLSceneUniformLocation;
	public readonly particleShadowVolumeSliceParams: WebGLSceneUniformLocation;
	public readonly dirShadowViewProjection: WebGLSceneUniformLocation;
	public readonly dirShadowCascadeViewProjection: WebGLSceneUniformLocation;
	public readonly dirShadowCascadeSplits: WebGLSceneUniformLocation;
	public readonly dirShadowParamsA: WebGLSceneUniformLocation;
	public readonly dirShadowParamsB: WebGLSceneUniformLocation;
	public readonly dirShadowParamsC: WebGLSceneUniformLocation;
	public readonly dirShadowParamsD: WebGLSceneUniformLocation;
	public readonly spotShadowViewProjection: WebGLSceneUniformLocation;
	public readonly spotShadowParamsA: WebGLSceneUniformLocation;
	public readonly spotShadowParamsB: WebGLSceneUniformLocation;
	public readonly spotShadowParamsC: WebGLSceneUniformLocation;
	public readonly spotShadowParamsD: WebGLSceneUniformLocation;

	public constructor(gl: WebGL2RenderingContext, program: WebGLProgram) {
		const get = createUniformLookup(gl, program);
		this.enableShadows = get("uEnableShadows");
		this.shadowAtlas = get("uShadowAtlas");
		this.shadowTransmittanceAtlas = get("uShadowTransmittanceAtlas");
		this.shadowTransmittanceAtlasAvailable = get(
			"uShadowTransmittanceAtlasAvailable"
		);
		this.particleShadowVolumeAtlas = get("uParticleShadowVolumeAtlas");
		this.particleShadowVolumeAtlasSize = get("uParticleShadowVolumeAtlasSize");
		this.particleShadowVolumeGridSize = get("uParticleShadowVolumeGridSize");
		this.particleShadowVolumeSliceParams = get(
			"uParticleShadowVolumeSliceParams[0]"
		);
		this.dirShadowViewProjection = get("uDirShadowViewProjection[0]");
		this.dirShadowCascadeViewProjection = get(
			"uDirShadowCascadeViewProjection[0]"
		);
		this.dirShadowCascadeSplits = get("uDirShadowCascadeSplits[0]");
		this.dirShadowParamsA = get("uDirShadowParamsA[0]");
		this.dirShadowParamsB = get("uDirShadowParamsB[0]");
		this.dirShadowParamsC = get("uDirShadowParamsC[0]");
		this.dirShadowParamsD = get("uDirShadowParamsD[0]");
		this.spotShadowViewProjection = get("uSpotShadowViewProjection[0]");
		this.spotShadowParamsA = get("uSpotShadowParamsA[0]");
		this.spotShadowParamsB = get("uSpotShadowParamsB[0]");
		this.spotShadowParamsC = get("uSpotShadowParamsC[0]");
		this.spotShadowParamsD = get("uSpotShadowParamsD[0]");
	}
}

export class WebGLSceneClusterUniforms {
	public readonly enableClusteredLighting: WebGLSceneUniformLocation;
	public readonly clusterParams0: WebGLSceneUniformLocation;
	public readonly clusterParams1: WebGLSceneUniformLocation;
	public readonly clusterHeaderTexture: WebGLSceneUniformLocation;
	public readonly clusterIndexTexture: WebGLSceneUniformLocation;
	public readonly clusterLightTexture: WebGLSceneUniformLocation;
	public readonly clusterHeaderTexSize: WebGLSceneUniformLocation;
	public readonly clusterIndexTexSize: WebGLSceneUniformLocation;
	public readonly clusterLightTexSize: WebGLSceneUniformLocation;

	public constructor(gl: WebGL2RenderingContext, program: WebGLProgram) {
		const get = createUniformLookup(gl, program);
		this.enableClusteredLighting = get("uEnableClusteredLighting");
		this.clusterParams0 = get("uClusterParams0");
		this.clusterParams1 = get("uClusterParams1");
		this.clusterHeaderTexture = get("uClusterHeaderTexture");
		this.clusterIndexTexture = get("uClusterIndexTexture");
		this.clusterLightTexture = get("uClusterLightTexture");
		this.clusterHeaderTexSize = get("uClusterHeaderTexSize");
		this.clusterIndexTexSize = get("uClusterIndexTexSize");
		this.clusterLightTexSize = get("uClusterLightTexSize");
	}
}

export class WebGLSceneFogUniforms {
	public readonly fogParams0: WebGLSceneUniformLocation;
	public readonly fogParams1: WebGLSceneUniformLocation;

	public constructor(gl: WebGL2RenderingContext, program: WebGLProgram) {
		const get = createUniformLookup(gl, program);
		this.fogParams0 = get("uFogParams0");
		this.fogParams1 = get("uFogParams1");
	}
}

export class WebGLSceneCustomSamplerUniforms {
	public readonly customSamplers: Record<string, WebGLSceneUniformLocation>;

	public constructor(
		gl: WebGL2RenderingContext,
		program: WebGLProgram,
		customSamplerUniforms: string[]
	) {
		this.customSamplers = {};
		for (const uniformName of customSamplerUniforms) {
			this.customSamplers[uniformName] = gl.getUniformLocation(
				program,
				uniformName
			);
		}
	}
}

export class WebGLSceneCustomUniforms {
	public readonly customUniforms: Record<string, WebGLSceneUniformLocation>;

	public constructor(
		gl: WebGL2RenderingContext,
		program: WebGLProgram,
		customUniforms: string[]
	) {
		this.customUniforms = {};
		for (const uniformName of customUniforms) {
			this.customUniforms[uniformName] = gl.getUniformLocation(
				program,
				uniformName
			);
		}
	}
}

export type WebGLSceneUniforms =
	WebGLSceneTransformUniforms &
	WebGLSceneFrameUniforms &
	WebGLSceneMaterialUniforms &
	WebGLSceneMaterialTextureUniforms &
	WebGLSceneEnvironmentUniforms &
	WebGLSceneLightUniforms &
	WebGLSceneShadowUniforms &
	WebGLSceneClusterUniforms &
	WebGLSceneFogUniforms &
	WebGLSceneCustomSamplerUniforms &
	WebGLSceneCustomUniforms;

/**
 * Resolves all scene-program uniform locations while keeping the declarations
 * grouped by scene concern.
 */
export function createWebGLSceneUniforms(
	gl: WebGL2RenderingContext,
	program: WebGLProgram,
	customSamplerUniforms: string[] = [],
	customUniforms: string[] = []
): WebGLSceneUniforms {
	return Object.assign(
		new WebGLSceneTransformUniforms(gl, program),
		new WebGLSceneFrameUniforms(gl, program),
		new WebGLSceneMaterialUniforms(gl, program),
		new WebGLSceneMaterialTextureUniforms(gl, program),
		new WebGLSceneEnvironmentUniforms(gl, program),
		new WebGLSceneLightUniforms(gl, program),
		new WebGLSceneShadowUniforms(gl, program),
		new WebGLSceneClusterUniforms(gl, program),
		new WebGLSceneFogUniforms(gl, program),
		new WebGLSceneCustomSamplerUniforms(gl, program, customSamplerUniforms),
		new WebGLSceneCustomUniforms(gl, program, customUniforms)
	) as WebGLSceneUniforms;
}
