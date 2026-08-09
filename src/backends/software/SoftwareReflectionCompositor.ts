import { AlphaMode } from "../../materials/Material";
import type { DrawPacket } from "../../pipeline/types";
import type { ProjectedFace, ProjectedVertex } from "../../core/types";
import { Projector } from "./Projector";
import type { SoftwarePassContext } from "./SoftwareFrameServices";
import { resolveSoftwarePlanarReflectionPlaneKey } from "./SoftwareReflectionPlanner";
import type { SoftwarePlanarReflectionBuffer } from "./SoftwareReflectionResources";

export interface SoftwareCompositeClipRect {
	readonly minX: number;
	readonly minY: number;
	readonly maxX: number;
	readonly maxY: number;
}

export type SoftwareReflectionTriangleComposite = (
	triangle: [ProjectedVertex, ProjectedVertex, ProjectedVertex],
	face: ProjectedFace,
	pixels: Uint8ClampedArray,
	depthBuffer: Float32Array,
	width: number,
	height: number,
	clipRect: SoftwareCompositeClipRect,
	buffer: SoftwarePlanarReflectionBuffer,
	reflectivity: number,
) => void;

/** @internal Projects reflective surfaces and schedules reflection blending. */
export class SoftwareReflectionCompositor {
	public composite(
		context: SoftwarePassContext,
		packets: readonly DrawPacket[],
		buffers: ReadonlyMap<string, SoftwarePlanarReflectionBuffer>,
		compositeTriangle: SoftwareReflectionTriangleComposite,
	): void {
		const frame = context.frame;
		if (!frame.features.enableReflection || packets.length === 0 || buffers.size === 0) {
			return;
		}
		const { pixels, depthBuffer, width, height } = frame.attachments;
		if (width <= 0 || height <= 0) return;
		const clipRects = frame.clipRegions.map((region) => ({
			minX: region.minX,
			minY: region.minY,
			maxX: region.maxXExclusive - 1,
			maxY: region.maxYExclusive - 1,
		}));
		if (clipRects.length === 0) return;

		for (const packet of packets) {
			const material = packet.material;
			const reflectivity = Math.max(0, Math.min(1, material.reflectivity ?? 0));
			if (
				reflectivity <= 0 ||
				!material.mirrorPlane ||
				material.alphaMode === AlphaMode.Blend
			) {
				continue;
			}
			const plane = material.mirrorPlane;
			const camera = frame.camera.position;
			const cameraDistance =
				camera.x * plane.normal.x +
				camera.y * plane.normal.y +
				camera.z * plane.normal.z +
				plane.constant;
			if (cameraDistance <= 0) continue;
			const key = resolveSoftwarePlanarReflectionPlaneKey(plane);
			const buffer = key ? buffers.get(key) : null;
			if (!buffer) continue;
			for (const face of Projector.projectPacket(packet, frame)) {
				for (let index = 1; index < face.projected.length - 1; index++) {
					const triangle: [ProjectedVertex, ProjectedVertex, ProjectedVertex] = [
						face.projected[0],
						face.projected[index],
						face.projected[index + 1],
					];
					for (const clipRect of clipRects) {
						compositeTriangle(
							triangle,
							face,
							pixels,
							depthBuffer,
							width,
							height,
							clipRect,
							buffer,
							reflectivity,
						);
					}
				}
			}
		}
	}
}
