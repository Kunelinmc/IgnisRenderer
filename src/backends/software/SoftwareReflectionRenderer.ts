import type { Plane } from "../../maths/Plane";
import type { SoftwarePassContext } from "./SoftwareFrameServices";
import type { SoftwareReflectionPlaneInfo } from "./SoftwareReflectionPlanner";
import type { SoftwarePlanarReflectionBuffer } from "./SoftwareReflectionResources";
import { SoftwareReflectionResources } from "./SoftwareReflectionResources";

export type SoftwareMirroredPlaneRender = (
	plane: Plane,
	buffer: SoftwarePlanarReflectionBuffer,
	context: SoftwarePassContext,
) => void;

/** @internal Allocates targets and schedules mirrored Software views. */
export class SoftwareReflectionRenderer {
	public render(
		context: SoftwarePassContext,
		planes: ReadonlyMap<string, SoftwareReflectionPlaneInfo>,
		resources: SoftwareReflectionResources,
		resolutionScale: number,
		renderPlane: SoftwareMirroredPlaneRender,
	): boolean {
		const { width, height } = context.frame.attachments;
		if (planes.size === 0 || width <= 0 || height <= 0) {
			resources.clear();
			return false;
		}
		const scaledWidth = Math.max(1, Math.floor(width * resolutionScale));
		const scaledHeight = Math.max(1, Math.floor(height * resolutionScale));
		for (const [key, info] of planes) {
			renderPlane(
				info.plane,
				resources.ensure(key, scaledWidth, scaledHeight),
				context,
			);
		}
		resources.trim(new Set(planes.keys()));
		return true;
	}
}
