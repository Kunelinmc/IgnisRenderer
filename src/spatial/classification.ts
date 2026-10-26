import type { MeshInstance } from "../meshes";
import { LODMeshInstance } from "../meshes/LODMeshInstance";

export function isDynamicSpatialMeshInstance(meshInstance: MeshInstance): boolean {
	if (meshInstance.skeleton) return true;
	if (meshInstance instanceof LODMeshInstance) return true;
	for (const primitive of meshInstance.mesh.primitives) {
		if ((primitive.geometry.morphTargets?.length ?? 0) > 0) {
			return true;
		}
	}
	return false;
}
