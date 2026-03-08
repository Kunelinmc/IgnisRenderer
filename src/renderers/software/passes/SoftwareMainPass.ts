import { Projector } from "../Projector";
import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import type { Rasterizer } from "../Rasterizer";
import {
	createSoftwareShadowSampler,
	getSoftwareShadowRuntimeMap,
} from "../shadows";

export class SoftwareMainPass {
	private _rasterizer: Rasterizer;

	constructor(rasterizer: Rasterizer) {
		this._rasterizer = rasterizer;
	}

	public render(
		context: FrameContext,
		packets: DrawPacket[],
		transparent: boolean
	): void {
		const runtimeMap = getSoftwareShadowRuntimeMap(context.transient);
		const sampleShadow = createSoftwareShadowSampler(
			context.shadowMaps,
			runtimeMap
		);

		const rasterizerContext = {
			width: context.attachments.width,
			height: context.attachments.height,
			depthBuffer: context.attachments.depthBuffer!,
			normalBuffer: context.attachments.normalBuffer,
			camera: {
				position: context.camera.getWorldPosition(),
				viewMatrix: context.camera.viewMatrix,
			},
			lights: context.scene.lights,
			shadowMaps: context.shadowMaps,
			sampleShadow,
			shAmbientCoeffs: context.shAmbientCoeffs,
			features: {
				enableLighting: context.features.enableLighting,
				enableSH: context.features.enableSH,
				enableShadows: context.features.enableShadows,
				enableGamma: context.features.enableGamma,
				enableReflection: context.features.enableReflection,
				worldMatrix: context.worldMatrix,
			},
		};

		for (const packet of packets) {
			const faces = Projector.projectPacket(packet, context);
			if (transparent) {
				faces.sort((left, right) => right.depthInfo.avg - left.depthInfo.avg);
			}

			for (const face of faces) {
				const projected = face.projected;
				for (let i = 1; i < projected.length - 1; i++) {
					this._rasterizer.drawTriangle(
						[projected[0], projected[i], projected[i + 1]],
						face,
						context.attachments.pixels!,
						rasterizerContext,
						transparent
					);
				}
			}
		}
	}
}
