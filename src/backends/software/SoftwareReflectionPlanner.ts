import type { MirrorPlane } from "../../materials/Material";
import { Plane } from "../../maths/Plane";
import type { SoftwareFrameView } from "./SoftwareFrameView";

export interface SoftwareReflectionPlaneInfo {
	readonly plane: Plane;
}

export function resolveSoftwarePlanarReflectionPlaneKey(
	plane: MirrorPlane | Plane | null | undefined,
): string | null {
	if (!plane) return null;
	return `${plane.normal.x},${plane.normal.y},${plane.normal.z},${plane.constant}`;
}

/** @internal Resolves distinct mirror planes for one Software frame. */
export class SoftwareReflectionPlanner {
	private readonly _planes = new Map<string, Plane>();

	public collect(frame: SoftwareFrameView): Map<string, SoftwareReflectionPlaneInfo> {
		const result = new Map<string, SoftwareReflectionPlaneInfo>();
		for (const packet of frame.scene.reflectivePackets) {
			const material = packet.submission.material.effective;
			const key = resolveSoftwarePlanarReflectionPlaneKey(material?.mirrorPlane);
			if (!material || material.reflectivity <= 0 || !key) continue;
			let plane = this._planes.get(key);
			if (!plane) {
				plane = new Plane(
					material.mirrorPlane!.normal,
					material.mirrorPlane!.constant,
				);
				this._planes.set(key, plane);
			}
			result.set(key, { plane });
		}
		return result;
	}

	public trim(activePlanes: ReadonlyMap<string, SoftwareReflectionPlaneInfo>): void {
		for (const key of this._planes.keys()) {
			if (!activePlanes.has(key)) this._planes.delete(key);
		}
	}

	public clear(): void {
		this._planes.clear();
	}
}
