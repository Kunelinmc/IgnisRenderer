import {
	createInlineCompositeShaderSource,
	type CompositeShaderSource,
} from "../runtime";
import {
	loadWebGLShaderPart,
	loadWebGLShaderPartComposite,
	WEBGL_SHADER_PARTS,
	type WebGLShaderPart,
} from "./shaderSource";

const WEBGL_PIPELINE_SHADER_PARTS: readonly WebGLShaderPart[] =
	WEBGL_SHADER_PARTS.filter((part) => part !== "sceneVertex" && part !== "sceneFragment");

export interface WebGLSceneLightLimits {
	maxDirectionalLights: number;
	maxPointLights: number;
	maxSpotLights: number;
	/**
	 * Enables the optional transparent-shadow transmittance sampler. Leave disabled
	 * on devices that only expose the WebGL2 minimum of 16 fragment texture units.
	 */
	enableShadowTransmittance?: boolean;
}

export interface WebGLSceneShaderSource {
	vertex: string;
	fragment: string;
}

export interface WebGLSceneCompositeShaderSource {
	vertex: CompositeShaderSource;
	fragment: CompositeShaderSource;
}

function replaceLightLimit(
	source: string,
	placeholder: string,
	value: number
): string {
	return source.replaceAll(placeholder, String(Math.max(0, value | 0)));
}

function replaceOptionalDefines(
	source: string,
	limits: WebGLSceneLightLimits
): string {
	const shadowTransmittanceEnabled = !!limits.enableShadowTransmittance;
	return source
		.replaceAll(
			"__WEBGL_SHADOW_TRANSMITTANCE_DEFINE__",
			shadowTransmittanceEnabled ?
				"#define WEBGL_SHADOW_TRANSMITTANCE 1"
			:	""
		)
		.replaceAll(
			"__WEBGL_SHADOW_TRANSMITTANCE_UNIFORMS__",
			shadowTransmittanceEnabled ?
				[
					"uniform sampler2D uShadowTransmittanceAtlas;",
					"uniform int uShadowTransmittanceAtlasAvailable;",
				].join("\n")
			:	""
		);
}

function cloneCompositeSource(
	composite: CompositeShaderSource
): CompositeShaderSource {
	return {
		code: composite.code,
		sourceMap: {
			lineCount: composite.sourceMap.lineCount,
			segments: composite.sourceMap.segments.map((segment) => ({ ...segment })),
		},
	};
}

function getSceneTemplateSourcePath(
	fragmentTemplate: CompositeShaderSource
): string {
	return (
		fragmentTemplate.sourceMap.segments[0]?.sourcePath ??
		"./parts/sceneFragment.glsl"
	);
}

export class WebGLShaderSourceFactory {
	private _rawByPart = new Map<WebGLShaderPart, string>();
	private _compositeByPart = new Map<WebGLShaderPart, CompositeShaderSource>();
	private _inFlightByPart = new Map<WebGLShaderPart, Promise<void>>();

	public async preparePart(part: WebGLShaderPart): Promise<void> {
		if (this._rawByPart.has(part) && this._compositeByPart.has(part)) {
			return;
		}
		const existing = this._inFlightByPart.get(part);
		if (existing) {
			await existing;
			return;
		}
		const loading = (async () => {
			const [raw, composite] = await Promise.all([
				loadWebGLShaderPart(part),
				loadWebGLShaderPartComposite(part),
			]);
			this._rawByPart.set(part, raw);
			this._compositeByPart.set(part, composite);
		})();
		this._inFlightByPart.set(part, loading);
		try {
			await loading;
		} finally {
			this._inFlightByPart.delete(part);
		}
	}

	public async prepareParts(parts: readonly WebGLShaderPart[]): Promise<void> {
		await Promise.all(parts.map((part) => this.preparePart(part)));
	}

	public async prepareAll(): Promise<void> {
		await this.prepareParts(WEBGL_SHADER_PARTS);
	}

	public async preparePipelineParts(): Promise<void> {
		await this.prepareParts(WEBGL_PIPELINE_SHADER_PARTS);
	}

	public async prepareSceneParts(): Promise<void> {
		await this.prepareParts(["sceneVertex", "sceneFragment"]);
	}

	public hasPart(part: WebGLShaderPart): boolean {
		return this._rawByPart.has(part) && this._compositeByPart.has(part);
	}

	public getRawPart(part: WebGLShaderPart): string {
		const source = this._rawByPart.get(part);
		if (typeof source !== "string") {
			throw new Error(
				`WebGL shader source part "${part}" is not prepared. ` +
					`Migration hint: call WebGLShaderSourceFactory.preparePart()/prepareAll() during backend initialization.`
			);
		}
		return source;
	}

	public getCompositePart(part: WebGLShaderPart): CompositeShaderSource {
		const source = this._compositeByPart.get(part);
		if (!source) {
			throw new Error(
				`WebGL shader composite part "${part}" is not prepared. ` +
					`Migration hint: call WebGLShaderSourceFactory.preparePart()/prepareAll() during backend initialization.`
			);
		}
		return cloneCompositeSource(source);
	}

	public createSceneShaderSource(
		limits: WebGLSceneLightLimits
	): WebGLSceneShaderSource {
		const sceneVertexSource = this.getRawPart("sceneVertex");
		const sceneFragmentTemplate = replaceOptionalDefines(
			this.getRawPart("sceneFragment"),
			limits
		);
		const withDirectional = replaceLightLimit(
			sceneFragmentTemplate,
			"__MAX_DIRECTIONAL_LIGHTS__",
			limits.maxDirectionalLights
		);
		const withPoint = replaceLightLimit(
			withDirectional,
			"__MAX_POINT_LIGHTS__",
			limits.maxPointLights
		);
		const fragment = replaceLightLimit(
			withPoint,
			"__MAX_SPOT_LIGHTS__",
			limits.maxSpotLights
		);
		return {
			vertex: sceneVertexSource,
			fragment,
		};
	}

	public createSceneCompositeShaderSource(
		limits: WebGLSceneLightLimits
	): WebGLSceneCompositeShaderSource {
		const sceneVertexComposite = this.getCompositePart("sceneVertex");
		const sceneFragmentCompositeTemplate = this.getCompositePart("sceneFragment");
		const shader = this.createSceneShaderSource(limits);
		return {
			vertex: sceneVertexComposite,
			fragment: createInlineCompositeShaderSource(
				shader.fragment,
				getSceneTemplateSourcePath(sceneFragmentCompositeTemplate),
				"template"
			),
		};
	}
}

export function createWebGLShaderSourceFactory(): WebGLShaderSourceFactory {
	return new WebGLShaderSourceFactory();
}

export type { WebGLShaderPart } from "./shaderSource";
