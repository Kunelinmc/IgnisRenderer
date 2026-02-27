import type {
	IVector3,
	IVector4,
	SHCoefficients,
	IVector2,
} from "../maths/types";
import type { Renderer } from "../core/Renderer";
import type { Matrix4 } from "../maths/Matrix4";
import type { ProjectedFace } from "../core/types";
import type { Material } from "../materials";
import type { RGB } from "../utils/Color";
import type { SceneLight } from "../lights";

export interface ShaderContext {
	renderer: Renderer;
	cameraPos: IVector3;
	lights: SceneLight[];
	worldMatrix?: Matrix4;
	shAmbientCoeffs: SHCoefficients | null;
	enableShadows: boolean;
	enableSH: boolean;
	enableGamma: boolean;
	enableLighting: boolean;
	gamma: number;
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
	sheenColor: RGB;
	sheenRoughness: number;
	transmission: number;
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

export interface IMaterialEvaluator<
	T extends SurfaceProperties = SurfaceProperties,
> {
	/**
	 * @deprecated Use compile(material) instead.
	 */
	setMaterial(material: Material): void;
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
	shade(input: FragmentInput): FragmentOutput | null;
}
