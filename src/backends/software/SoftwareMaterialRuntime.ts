import {
	Material,
	ShadingModel,
} from "../../materials/Material";
import { IBLBRDF } from "../../lights/ibl/IBLBRDF";
import { collectActiveReflectionProbes } from "../../lights/runtime/reflectionProbeRuntime";
import {
	FlatLitShader,
	LitShader,
	PBRStrategy,
	BlinnPhongStrategy,
	PhongEvaluator,
	PBREvaluator,
	UnlitShader,
	type IShader,
	type ShaderContext,
	type IMaterialEvaluator,
	type ILightingStrategy,
	type FragmentInput,
	type FragmentOutput,
	type PhongSurfaceProperties,
	type PBRSurfaceProperties,
} from "../../shaders";
import { sampleSoftwareTextureAlpha } from "../../shaders/software/textureSampling";
import type {
	SceneLight,
	ShadowCastingLight,
} from "../../lights";
import type { ProjectedFace } from "../../core/types";
import type { Texture } from "../../core/Texture";
import type {
	IVector3,
	SHCoefficients,
} from "../../maths/types";
import type { DecalPacket } from "../../pipeline/types";
import { SoftwareDecalSurfaceModifier } from "./SoftwareDecalSurfaceModifier";

/**
 * @internal
 * Minimal frame state required to prepare a software fragment program.
 */
export interface SoftwareMaterialRuntimeContext {
	camera: {
		position: IVector3;
	};
	lights: SceneLight[];
	sampleShadow?: (
		light: ShadowCastingLight,
		worldPoint: IVector3,
		normal?: IVector3 | null
	) => { r: number; g: number; b: number };
	shAmbientCoeffs: SHCoefficients | null;
	environmentSpecularTexture?: Texture | null;
	enableLighting: boolean;
	enableSH: boolean;
	enableShadows: boolean;
}

/**
 * @internal
 * Prepared per-triangle software fragment program.
 */
export interface SoftwareFragmentProgram {
	readonly material: Material;
	readonly shouldWriteDepth: boolean;
	shade(input: FragmentInput): FragmentOutput | null;
	getOpacity(): number;
	getSurfaceNormal(): IVector3 | null;
}

class ShaderFragmentProgram implements SoftwareFragmentProgram {
	public constructor(
		public readonly material: Material,
		public readonly shouldWriteDepth: boolean,
		private _shader: IShader
	) {}

	public shade(input: FragmentInput): FragmentOutput | null {
		return this._shader.shade(input);
	}

	public getOpacity(): number {
		return this._shader.getOpacity();
	}

	public getSurfaceNormal(): IVector3 | null {
		return this._shader.getSurfaceNormal();
	}
}

/**
 * @internal
 * Owns software material evaluation and shader selection for `Rasterizer`.
 */
export class SoftwareMaterialRuntime {
	private _defaultMaterial: Material;
	private _evaluators: Map<string, IMaterialEvaluator> = new Map();
	private _strategies: Map<string, ILightingStrategy> = new Map();
	private _shaderCache: Map<string, IShader> = new Map();
	private _decalSurfaceModifier = new SoftwareDecalSurfaceModifier();

	public constructor() {
		this._defaultMaterial = new Material();
		this._initShaderSystem();
	}

	/**
	 * @internal
	 * Returns a concrete material, replacing missing face materials with the
	 * software default material.
	 *
	 * @param material - Optional face material.
	 * @returns A non-null material instance.
	 * @sideEffects None.
	 */
	public resolveMaterial(material: Material | null | undefined): Material {
		return material ?? this._defaultMaterial;
	}

	/**
	 * @internal
	 * Prepares the shader/evaluator pair used by one projected face.
	 *
	 * @param face - Projected face being rasterized.
	 * @param context - Current software frame material state.
	 * @param transparent - Whether the caller is drawing the transparent pass.
	 * @returns A fragment program initialized for `face`.
	 * @sideEffects Updates reusable evaluator and shader cache state.
	 */
	public prepareFragmentProgram(
		face: ProjectedFace,
		context: SoftwareMaterialRuntimeContext,
		transparent: boolean,
		decalPackets?: readonly DecalPacket[]
	): SoftwareFragmentProgram {
		const material = this.resolveMaterial(face.material);
		const shadingModel = material.shading || ShadingModel.Flat;
		const isLightingEnabled = context.enableLighting !== false;
		const shading = isLightingEnabled ? shadingModel : ShadingModel.Unlit;

		const shader = this._getShader(shading, material);
		const lights = context.lights;
		const reflectionProbes = collectActiveReflectionProbes(lights);
		const reflectionProbeFallbackMap =
			reflectionProbes.length <= 0 ?
				(context.environmentSpecularTexture ?? null)
			:	null;

		const shaderContext: ShaderContext = {
			cameraPos: context.camera.position,
			lights,
			sampleShadow: context.sampleShadow,
			shAmbientCoeffs: context.shAmbientCoeffs,
			reflectionProbes,
			reflectionProbeFallbackMap,
			brdfLUT: IBLBRDF.getLUT(),
			enableShadows: !!context.enableShadows,
			enableSH: !!context.enableSH,
			surfaceModifier: this._decalSurfaceModifier,
		};
		this._decalSurfaceModifier.prepare(transparent ? null : decalPackets);
		shader.initialize(face, shaderContext);

		return new ShaderFragmentProgram(
			material,
			!transparent && material.depthWrite,
			shader
		);
	}

	/**
	 * @internal
	 * Samples the effective alpha for material mask decisions.
	 *
	 * @param material - Material whose opacity and base texture alpha are used.
	 * @param u - Source U coordinate before texture transform.
	 * @param v - Source V coordinate before texture transform.
	 * @returns Alpha in 0..1 after material opacity is applied.
	 * @sideEffects None.
	 */
	public sampleAlphaMask(material: Material, u: number, v: number): number {
		return sampleSoftwareTextureAlpha(material.map, u, v) * (material.opacity ?? 1);
	}

	private _initShaderSystem(): void {
		this._evaluators.set(
			ShadingModel.Phong,
			new PhongEvaluator(this._defaultMaterial)
		);
		this._evaluators.set(
			ShadingModel.PBR,
			new PBREvaluator(this._defaultMaterial)
		);

		this._strategies.set(ShadingModel.Phong, new BlinnPhongStrategy());
		this._strategies.set(ShadingModel.PBR, new PBRStrategy());
	}

	private _getShader(shading: string, material: Material): IShader {
		const isPBR = shading === ShadingModel.PBR || material.type === "PBR";
		const evaluatorType = isPBR ? ShadingModel.PBR : ShadingModel.Phong;

		const evaluator = this._evaluators.get(evaluatorType)!;
		const strategy = this._strategies.get(evaluatorType)!;

		evaluator.compile(material);

		const key = `${shading}_${evaluatorType}`;
		let shader = this._shaderCache.get(key);

		if (!shader) {
			shader = this._createShaderInstance(shading, evaluator, strategy, isPBR);
			this._shaderCache.set(key, shader);
		} else {
			shader.setEvaluator(evaluator);
		}

		return shader;
	}

	private _createShaderInstance(
		shading: string,
		evaluator: IMaterialEvaluator,
		strategy: ILightingStrategy,
		isPBR: boolean
	): IShader {
		if (shading === ShadingModel.Unlit) {
			return new UnlitShader(evaluator);
		}

		if (isPBR) {
			return new LitShader(
				strategy as ILightingStrategy<PBRSurfaceProperties>,
				evaluator as IMaterialEvaluator<PBRSurfaceProperties>
			);
		}

		if (shading === ShadingModel.Flat) {
			return new FlatLitShader(
				strategy as ILightingStrategy<PhongSurfaceProperties>,
				evaluator as IMaterialEvaluator<PhongSurfaceProperties>
			);
		}

		return new LitShader(
			strategy as ILightingStrategy<PhongSurfaceProperties>,
			evaluator as IMaterialEvaluator<PhongSurfaceProperties>
		);
	}
}
