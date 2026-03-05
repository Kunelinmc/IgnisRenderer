import { Matrix4 } from '../../../maths/Matrix4'
import type { Matrix3Arr } from '../../../maths/types'

import {
	WEBGPU_FRAME_UNIFORM_FLOATS,
	WEBGPU_MAX_DIRECTIONAL_LIGHTS,
	WEBGPU_MAX_POINT_LIGHTS,
	WEBGPU_MAX_SPOT_LIGHTS,
	WEBGPU_MODEL_UNIFORM_FLOATS,
	WEBGPU_SH_COEFFICIENT_COUNT,
} from "./constants";
import type {
	WebGPUFrameUniformInput,
	WebGPUMaterialUniformData,
} from "./types";

export function packMatrix4ForWGSL(matrix: Matrix4 | number[][]): Float32Array {
	const elements = matrix instanceof Matrix4 ? matrix.elements : matrix;

	return new Float32Array([
		elements[0][0],
		elements[1][0],
		elements[2][0],
		elements[3][0],
		elements[0][1],
		elements[1][1],
		elements[2][1],
		elements[3][1],
		elements[0][2],
		elements[1][2],
		elements[2][2],
		elements[3][2],
		elements[0][3],
		elements[1][3],
		elements[2][3],
		elements[3][3],
	]);
}

export function packNormalMatrix4ForWGSL(
	normalMatrix: Matrix3Arr | Matrix4
): Float32Array {
	const rows =
		normalMatrix instanceof Matrix4 ? normalMatrix.elements : normalMatrix;

	return packMatrix4ForWGSL([
		[rows[0][0], rows[0][1], rows[0][2], 0],
		[rows[1][0], rows[1][1], rows[1][2], 0],
		[rows[2][0], rows[2][1], rows[2][2], 0],
		[0, 0, 0, 1],
	]);
}

export function packFrameUniformData(
	input: WebGPUFrameUniformInput
): Float32Array {
	const data = new Float32Array(WEBGPU_FRAME_UNIFORM_FLOATS);
	const viewProjection = packMatrix4ForWGSL(input.viewProjectionMatrix);

	data.set(viewProjection, 0);
	data.set(
		[input.cameraPosition.x, input.cameraPosition.y, input.cameraPosition.z, 1],
		16
	);
	data.set(
		[
			input.skyboxRight[0],
			input.skyboxRight[1],
			input.skyboxRight[2],
			input.skyboxTanHalfFov,
		],
		20
	);
	data.set(
		[
			input.skyboxUp[0],
			input.skyboxUp[1],
			input.skyboxUp[2],
			input.skyboxAspect,
		],
		24
	);
	data.set(
		[
			input.skyboxBackward[0],
			input.skyboxBackward[1],
			input.skyboxBackward[2],
			input.skyboxIsOrthographic ? 1 : 0,
		],
		28
	);
	data.set(
		[input.ambientColor[0], input.ambientColor[1], input.ambientColor[2], 1],
		32
	);
	data.set(
		[
			input.directionalLights.length,
			input.pointLights.length,
			input.spotLights.length,
			0,
		],
		36
	);
	data.set(
		[
			input.enableLighting ? 1 : 0,
			input.enableGamma ? 1 : 0,
			input.enableShadows ? 1 : 0,
			0,
		],
		40
	);
	data.set(
		[
			input.enableSH ? 1 : 0,
			input.hasSHAmbient ? 1 : 0,
			input.hasSkybox ? 1 : 0,
			input.hasEnvSpecular ? 1 : 0,
		],
		44
	);
	data.set(
		[input.hasBRDFLUT ? 1 : 0, Math.max(0, input.envSpecularMaxMipLevel), 0, 0],
		48
	);

	let offset = 52;
	for (let i = 0; i < WEBGPU_MAX_DIRECTIONAL_LIGHTS; i++) {
		const light = input.directionalLights[i];
		if (light) {
			data.set(
				[light.direction[0], light.direction[1], light.direction[2], 0],
				offset
			);
			data.set([light.color[0], light.color[1], light.color[2], 0], offset + 4);
		}
		offset += 8;
	}

	for (let i = 0; i < WEBGPU_MAX_POINT_LIGHTS; i++) {
		const light = input.pointLights[i];
		if (light) {
			data.set(
				[light.position[0], light.position[1], light.position[2], light.range],
				offset
			);
			data.set([light.color[0], light.color[1], light.color[2], 0], offset + 4);
		}
		offset += 8;
	}

	for (let i = 0; i < WEBGPU_MAX_SPOT_LIGHTS; i++) {
		const light = input.spotLights[i];
		if (light) {
			data.set(
				[light.position[0], light.position[1], light.position[2], light.range],
				offset
			);
			data.set(
				[
					light.direction[0],
					light.direction[1],
					light.direction[2],
					light.outerCos,
				],
				offset + 4
			);
			data.set(
				[light.color[0], light.color[1], light.color[2], light.innerCos],
				offset + 8
			);
		}
		offset += 12;
	}

	for (let i = 0; i < WEBGPU_MAX_DIRECTIONAL_LIGHTS; i++) {
		const shadow = input.directionalShadows[i];
		if (shadow?.enabled && shadow.viewProjectionMatrix) {
			data.set(packMatrix4ForWGSL(shadow.viewProjectionMatrix), offset);
		}

		data.set(
			[
				shadow?.enabled ? 1 : 0,
				shadow?.depthBias ?? 0,
				shadow?.normalBias ?? 0,
				shadow?.normalBiasMin ?? 0,
			],
			offset + 16
		);
		data.set(
			[
				shadow?.pcfRadius ?? 0,
				shadow?.shadowStrength ?? 0,
				shadow?.shadowMapSize ?? 0,
				shadow?.atlasTileSize ?? 0,
			],
			offset + 20
		);
		offset += 24;
	}

	for (let i = 0; i < WEBGPU_MAX_SPOT_LIGHTS; i++) {
		const shadow = input.spotShadows[i];
		if (shadow?.enabled && shadow.viewProjectionMatrix) {
			data.set(packMatrix4ForWGSL(shadow.viewProjectionMatrix), offset);
		}

		data.set(
			[
				shadow?.enabled ? 1 : 0,
				shadow?.depthBias ?? 0,
				shadow?.normalBias ?? 0,
				shadow?.normalBiasMin ?? 0,
			],
			offset + 16
		);
		data.set(
			[
				shadow?.pcfRadius ?? 0,
				shadow?.shadowStrength ?? 0,
				shadow?.shadowMapSize ?? 0,
				shadow?.atlasTileSize ?? 0,
			],
			offset + 20
		);
		offset += 24;
	}

	for (let i = 0; i < WEBGPU_SH_COEFFICIENT_COUNT; i++) {
		const coefficient = input.shAmbientCoeffs?.[i];
		if (coefficient) {
			data.set([coefficient.r, coefficient.g, coefficient.b, 0], offset);
		}
		offset += 4;
	}

	return data;
}

export function remapClipSpaceDepth(clipZ: number, clipW: number): number {
	return clipZ * 0.5 + clipW * 0.5
}

export function packModelUniformData(
	modelMatrix: Matrix4 | number[][],
	normalMatrix: Matrix3Arr | Matrix4,
	materialData: WebGPUMaterialUniformData
): Float32Array {
	const data = new Float32Array(WEBGPU_MODEL_UNIFORM_FLOATS)

	data.set(packMatrix4ForWGSL(modelMatrix), 0)
	data.set(packNormalMatrix4ForWGSL(normalMatrix), 16)
	data.set(materialData.baseColorFactor, 32)
	data.set(materialData.emissiveFactor, 36)
	data.set(materialData.surfaceParams0, 40)
	data.set(materialData.surfaceParams1, 44)
	data.set(materialData.surfaceParams2, 48)
	data.set(materialData.surfaceParams3, 52)
	data.set(materialData.specularColorFactor, 56)
	data.set(materialData.phongAmbientShininess, 60)
	data.set(materialData.phongSpecularShading, 64)
	data.set(materialData.sheenColorClearcoatNormalScale, 68)
	data.set(materialData.attenuationColor, 72)
	data.set(materialData.materialFlags, 76)

	let offset = 80
	for (const slot of materialData.textureSlots) {
		data.set(slot.transformA, offset)
		offset += 4
	}

	for (const slot of materialData.textureSlots) {
		data.set(slot.transformB, offset)
		offset += 4
	}

	return data
}
