import type { IVector3, SHCoefficients } from "../maths/types";
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

export interface FragmentInput {
	zCam: number;
	worldX: number;
	worldY: number;
	worldZ: number;
	normalX: number;
	normalY: number;
	normalZ: number;
	u: number;
	v: number;
	lar?: number;
	lag?: number;
	lab?: number;
	ldr?: number;
	ldg?: number;
	ldb?: number;
	lsr?: number;
	lsg?: number;
	lsb?: number;
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
	f0: RGB;
	occlusion: number;
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
	setMaterial(material: Material): void;
	evaluate(u: number, v: number, face: ProjectedFace): T | null;
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
	shade(input: FragmentInput): RGB | null;
}
