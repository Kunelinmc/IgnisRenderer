import type { RasterizerContext } from "./Rasterizer";
import type { SoftwarePassContext } from "./SoftwareFrameServices";

export interface SoftwareRasterContextOverrides {
	readonly width?: number;
	readonly height?: number;
	readonly depthBuffer?: Float32Array;
	readonly camera?: RasterizerContext["camera"];
	readonly sampleShadow?: RasterizerContext["sampleShadow"];
	readonly includeFrameAttachments?: boolean;
}

/** @internal Builds consistent main- and secondary-view raster inputs. */
export function createSoftwareRasterizerContext(
	context: SoftwarePassContext,
	overrides: SoftwareRasterContextOverrides = {},
): RasterizerContext {
	const frame = context.frame;
	const environment = frame.scene.environment;
	const includeFrameAttachments = overrides.includeFrameAttachments ?? true;
	return {
		width: overrides.width ?? frame.attachments.width,
		height: overrides.height ?? frame.attachments.height,
		depthBuffer: overrides.depthBuffer ?? frame.attachments.depthBuffer,
		normalBuffer: includeFrameAttachments ? frame.attachments.normalBuffer : null,
		motionBuffer: includeFrameAttachments ? frame.attachments.motionBuffer : null,
		temporal: includeFrameAttachments ? frame.temporal : undefined,
		camera: overrides.camera ?? {
			position: frame.camera.position,
			viewMatrix: frame.camera.viewMatrix,
		},
		lights: frame.scene.lights,
		sampleShadow: overrides.sampleShadow ?? context.services.shadow.sampler,
		shAmbientCoeffs: frame.shAmbientCoeffs,
		environmentSpecularTexture:
			environment.lightingEnabled ? environment.iblTexture : null,
		enableLighting: frame.features.enableLighting,
		enableSH: frame.features.enableSH,
		enableShadows: frame.features.enableShadows,
	};
}
