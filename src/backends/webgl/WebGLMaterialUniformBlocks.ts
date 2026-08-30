import { WebGLCapabilityError } from "../../foundation/Error";
import type { WebGLSceneMaterialVariant } from "../../shaders/webgl/sceneVariants";
import {
	WEBGL_MATERIAL_COMMON_BINDING,
	WEBGL_MATERIAL_LIGHTING_BINDING,
} from "./WebGLMaterialBufferCache";
import type { WebGLMaterialShadingFamily } from "./WebGLMaterialState";

const WEBGL_MAX_MATERIAL_BLOCK_BYTE_SIZE = 624;

export type WebGLSceneMaterialBinding =
	| { readonly mode: "legacy" }
	| {
			readonly mode: "ubo";
			readonly family: WebGLMaterialShadingFamily;
			readonly materialVariant: WebGLSceneMaterialVariant;
	  }
	| { readonly mode: "ubo-depth" };

export function validateWebGLMaterialUniformBufferCapabilities(
	gl: WebGL2RenderingContext,
): void {
	if (typeof gl.getParameter !== "function") return;
	validateMinimum(gl, gl.MAX_UNIFORM_BUFFER_BINDINGS, 2, "uniform-buffer bindings");
	validateMinimum(gl, gl.MAX_FRAGMENT_UNIFORM_BLOCKS, 2, "fragment uniform blocks");
	validateMinimum(
		gl,
		gl.MAX_UNIFORM_BLOCK_SIZE,
		WEBGL_MAX_MATERIAL_BLOCK_BYTE_SIZE,
		"uniform-block bytes",
	);
}

export function configureWebGLSceneMaterialBlocks(
	gl: WebGL2RenderingContext,
	program: WebGLProgram,
	variant: WebGLSceneMaterialVariant,
): WebGLSceneMaterialBinding {
	if (variant.model === "full") return { mode: "legacy" };
	bindRequiredBlock(gl, program, "IgnisMaterialCommon", WEBGL_MATERIAL_COMMON_BINDING);
	if (variant.model === "pbr") {
		bindRequiredBlock(gl, program, "IgnisPBRMaterial", WEBGL_MATERIAL_LIGHTING_BINDING);
	} else if (variant.model === "phong" || variant.model === "flat") {
		bindRequiredBlock(gl, program, "IgnisPhongMaterial", WEBGL_MATERIAL_LIGHTING_BINDING);
	}
	return { mode: "ubo", family: variant.model, materialVariant: variant };
}

export function configureWebGLDepthMaterialBlock(
	gl: WebGL2RenderingContext,
	program: WebGLProgram,
): WebGLSceneMaterialBinding {
	bindRequiredBlock(gl, program, "IgnisMaterialCommon", WEBGL_MATERIAL_COMMON_BINDING);
	return { mode: "ubo-depth" };
}

function validateMinimum(
	gl: WebGL2RenderingContext,
	parameter: number,
	required: number,
	label: string,
): void {
	if (typeof parameter !== "number") return;
	const value = gl.getParameter(parameter);
	if (typeof value === "number" && value >= required) return;
	throw new WebGLCapabilityError(
		"material-uniform-buffer-unavailable",
		`${label}: required=${required}, available=${String(value)}.`,
	);
}

function bindRequiredBlock(
	gl: WebGL2RenderingContext,
	program: WebGLProgram,
	name: string,
	binding: number,
): void {
	if (
		typeof gl.getUniformBlockIndex !== "function" ||
		typeof gl.uniformBlockBinding !== "function"
	) {
		// Lightweight static-test contexts do not expose WebGL2 reflection.
		return;
	}
	const index = gl.getUniformBlockIndex(program, name);
	const invalidIndex = typeof gl.INVALID_INDEX === "number" ? gl.INVALID_INDEX : 0xffffffff;
	if (index === invalidIndex || index === 0xffffffff) {
		throw new WebGLCapabilityError(
			"material-uniform-buffer-unavailable",
			`Program is missing required block ${name}.`,
		);
	}
	gl.uniformBlockBinding(program, index, binding);
}
