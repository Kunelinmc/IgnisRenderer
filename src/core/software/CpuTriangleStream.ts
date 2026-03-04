import { GeometryBuilder } from "../geometry/GeometryBuilder";
import type { DrawPacket } from "../pipeline/types";
import type { PrimitiveFace } from "../types";
import { Vector3 } from "../../maths/Vector3";

export class CpuTriangleStream {
	public static getPacketFaces(packet: DrawPacket): PrimitiveFace[] {
		const triangleCount = (packet.geometry.indices.length / 3) | 0;
		const faces: PrimitiveFace[] = [];

		for (
			let triangleIndex = 0;
			triangleIndex < triangleCount;
			triangleIndex++
		) {
			const vertices = GeometryBuilder.createVerticesForTriangle(
				packet.primitive,
				triangleIndex
			);
			faces.push({
				primitive: packet.primitive,
				material: packet.material,
				vertices,
				normal: Vector3.calculateNormal(vertices),
				doubleSided: packet.material.doubleSided,
			});
		}

		return faces;
	}
}
