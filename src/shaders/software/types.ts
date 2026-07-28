import type {
	IVector3,
	IVector4,
	SHCoefficients,
	IVector2,
} from "../../maths/types";
import type { Texture } from "../../core/Texture";
import type { Renderer } from "../../rendering/Renderer";
import type { Matrix4 } from "../../maths/Matrix4";
import type { ProjectedFace } from "../../core/types";
import type { Material } from "../../materials";
import type { RGB } from "../../foundation/Color";
import type { SceneLight, ShadowCastingLight } from "../../lights";
import type { ReflectionProbe } from "../../lights/ReflectionProbe";

export interface ShaderContext {
	cameraPos: IVector3;
	lights: SceneLight[];
	sampleShadow?: (
		light: ShadowCastingLight,
		worldPoint: IVector3,
		normal?: IVector3 | null
	) => RGB;
	shAmbientCoeffs: SHCoefficients | null;
	reflectionProbes: ReflectionProbe[];
	reflectionProbeFallbackMap: Texture | null;
	brdfLUT: Texture | null;
	enableShadows: boolean;
	enableSH: boolean;
	surfaceModifier?: ISurfaceModifier;
}

export interface FragmentOutput {
	color: RGB;
	depth?: number;
	motionVector?: IVector2;
}

export interface IBRDF {
	evaluate(
		surface: SurfaceProperties,
		lightDir: IVector3,
		viewDir: IVector3,
		normal: IVector3
	): RGB;
}

export interface FragmentInput {
	zCam: number;
	world: IVector3;
	normal: IVector3;
	tangent: IVector4;
	u: number;
	v: number;
	u2: number;
	v2: number;
	u3: number;
	v3: number;
	u4: number;
	v4: number;
	lightAmbient?: RGB;
	lightDiffuse?: RGB;
	lightSpecular?: RGB;
}

export interface BaseSurfaceProperties {
	albedo: RGB;
	opacity: number;
	normal: IVector3;
	emissive: RGB;
	emissiveIntensity: number;
}

export interface PBRSurfaceProperties extends BaseSurfaceProperties {
	type: "pbr";
	roughness: number;
	metalness: number;
	reflectance: number;
	specularFactor: number;
	specularColor: RGB;
	occlusion: number;
	clearcoat: number;
	clearcoatRoughness: number;
	clearcoatNormal: IVector3;
	sheenColor: RGB;
	sheenRoughness: number;
	transmission: number;
	ior: number;
	iridescence: number;
	iridescenceIor: number;
	iridescenceThickness: number;
	anisotropyStrength: number;
	anisotropyTangent: IVector3;
	anisotropyBitangent: IVector3;
	thickness: number;
	attenuationDistance: number;
	attenuationColor: RGB;
}

export interface PhongSurfaceProperties extends BaseSurfaceProperties {
	type: "phong";
	ambient: RGB;
	specular: RGB;
	shininess: number;
}

export type SurfaceProperties = PBRSurfaceProperties | PhongSurfaceProperties;

/**
 * @internal Backend-owned hook that modifies evaluated material properties
 * before lighting.
 */
export interface ISurfaceModifier {
	readonly active: boolean;
	apply(input: FragmentInput, surface: SurfaceProperties): void;
}

export interface IMaterialEvaluator<
	T extends SurfaceProperties = SurfaceProperties,
> {
	compile(material: Material): void;
	evaluate(input: FragmentInput, face: ProjectedFace): T | null;
}

export interface ILightingStrategy<
	T extends SurfaceProperties = SurfaceProperties,
> {
	calculate(
		world: IVector3,
		normal: IVector3,
		viewDir: IVector3,
		surface: T,
		context: ShaderContext
	): RGB;
}

export interface IShader {
	setEvaluator(evaluator: IMaterialEvaluator): void;
	initialize(face: ProjectedFace, context: ShaderContext): void;
	getOpacity(): number;
	getSurfaceNormal(): IVector3 | null;
	shade(input: FragmentInput): FragmentOutput | null;
}
