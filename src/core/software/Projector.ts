import { Matrix4 } from "../../maths/Matrix4";
import type { IModel, ProjectedFace } from "../types";
import type { Renderer } from "../Renderer";
import { getModelMatrix } from "../modelMatrix";
import { Matrix4 as Matrix4Class } from "../../maths/Matrix4";
import { PreparedSceneBuilder } from "../pipeline/PreparedSceneBuilder";
import { CpuProjector } from "./CpuProjector";
import type { DrawPacket } from "../pipeline/types";

export class Projector {
	public static projectModel(
		model: IModel,
		renderer: Renderer,
		flipCulling: boolean = false,
		overrideSize?: { width: number; height: number }
	): ProjectedFace[] {
		const worldMatrix = getModelMatrix(model);
		const normalMatrix = Matrix4.normalMatrix(worldMatrix);
		const packetCameraCenter = Matrix4Class.transformPoint(
			renderer.camera.viewMatrix,
			Matrix4Class.transformPoint(worldMatrix, model.boundingSphere.center)
		);
		const sortDepth = -packetCameraCenter.z;

		const packets: DrawPacket[] = model.primitives
			.filter((primitive) => primitive.visible !== false)
			.map((primitive) => ({
				id: `${model.id}:${primitive.id}`,
				model,
				primitive,
				material: primitive.material,
				geometry: primitive.geometry,
				worldMatrix,
				normalMatrix,
				worldBounds: primitive.boundingSphere,
				sortDepth,
				pipelineKey: "",
				passFlags: 0,
			}));

		return packets.flatMap((packet) =>
			CpuProjector.projectPacket(packet, renderer, flipCulling, overrideSize)
		);
	}

	public static getFaceAtPoint(
		projectedFaces: ProjectedFace[],
		px: number,
		py: number
	): ProjectedFace | null {
		let nearestFace: ProjectedFace | null = null;
		let minZ = Infinity;

		for (const face of projectedFaces) {
			const verts = face.projected;
			if (!verts || verts.length < 3) continue;

			let inside = false;
			for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {
				const xi = verts[i].x;
				const yi = verts[i].y;
				const xj = verts[j].x;
				const yj = verts[j].y;
				const intersect =
					yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi;
				if (intersect) inside = !inside;
			}

			if (!inside) continue;

			const avgZ =
				verts.reduce((sum, vertex) => sum + (vertex.z || 0), 0) / verts.length;
			if (avgZ < minZ) {
				minZ = avgZ;
				nearestFace = face;
			}
		}

		return nearestFace;
	}

	public static getModelMatrix(model: IModel): Matrix4 {
		return getModelMatrix(model);
	}
}
