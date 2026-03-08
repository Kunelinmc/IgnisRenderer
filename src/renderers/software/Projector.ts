import { CameraType } from "../../cameras/Camera";
import { Matrix4 } from "../../maths/Matrix4";
import { Vector3 } from "../../maths/Vector3";
import type {
	ProjectedFace,
	ProjectedVertex,
	IVertex,
	PrimitiveFace,
} from "../../core/types";
import { MeshInstance } from "../../meshes";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import { GeometryBuilder } from "../../meshes/GeometryBuilder";
import { ANIMATION_SOFTWARE_DEFORMED_GEOMETRY_KEY } from "../../simulation/animation/types";

interface ClippedVertexPair {
	view: IVertex;
	world: IVertex;
}

export class Projector {
	public static projectModel(
		meshInstance: MeshInstance,
		context: FrameContext,
		flipCulling: boolean = false,
		overrideSize?: { width: number; height: number }
	): ProjectedFace[] {
		const worldMatrix = meshInstance.worldMatrix;
		const normalMatrix = Matrix4.normalMatrix(worldMatrix);
		const packetCameraCenter = Matrix4.transformPoint(
			context.camera.viewMatrix,
			Matrix4.transformPoint(
				worldMatrix,
				meshInstance.mesh.boundingSphere.center
			)
		);
		const sortDepth = -packetCameraCenter.z;

		const packets: DrawPacket[] = meshInstance.mesh.primitives
			.filter((primitive) => primitive.visible !== false)
			.map((primitive) => ({
				id: `${meshInstance.id}:${primitive.id}`,
				meshInstance,
				mesh: meshInstance.mesh,
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
			this.projectPacket(packet, context, flipCulling, overrideSize)
		);
	}

	public static projectPacket(
		packet: DrawPacket,
		context: FrameContext,
		flipCulling: boolean = false,
		overrideSize?: { width: number; height: number }
	): ProjectedFace[] {
		const targetWidth = overrideSize?.width ?? context.attachments.width;
		const targetHeight = overrideSize?.height ?? context.attachments.height;
		const projectionMatrix = context.camera.projectionMatrix;
		const viewMatrix = context.camera.viewMatrix;
		const projectedFaces: ProjectedFace[] = [];

		for (const face of this.getPacketFacesWithContext(packet, context)) {
			const worldVerts: IVertex[] = [];
			const viewVerts: IVertex[] = [];

			for (const vertex of face.vertices) {
				const worldPoint = Matrix4.transformPoint(packet.worldMatrix, vertex);
				const worldNormal = vertex.normal
					? Vector3.normalize(
							Matrix4.transformNormal(packet.normalMatrix, vertex.normal)
						)
					: null;
				const worldTangent = vertex.tangent
					? (() => {
							const tangent = Vector3.normalize(
								Matrix4.transformNormal(packet.normalMatrix, vertex.tangent)
							);
							return {
								x: tangent.x,
								y: tangent.y,
								z: tangent.z,
								w: vertex.tangent!.w,
							};
						})()
					: null;

				const worldVertex: IVertex = {
					x: worldPoint.x,
					y: worldPoint.y,
					z: worldPoint.z,
					u: vertex.u,
					v: vertex.v,
					u2: vertex.u2,
					v2: vertex.v2,
					normal: worldNormal,
					tangent: worldTangent,
					color: vertex.color,
				};
				worldVerts.push(worldVertex);

				const viewPoint = Matrix4.transformPoint(viewMatrix, worldVertex);
				viewVerts.push({
					x: viewPoint.x,
					y: viewPoint.y,
					z: viewPoint.z,
					u: vertex.u,
					v: vertex.v,
					u2: vertex.u2,
					v2: vertex.v2,
					normal: worldNormal,
					tangent: worldTangent,
					color: vertex.color,
				});
			}

			const clippedVerts = clipFaceToNearPlane(viewVerts, worldVerts, context);
			if (clippedVerts.length < 3) continue;

			const cullNormal = Vector3.calculateNormal(
				clippedVerts.map((vertex) => vertex.view)
			);
			const v0 = clippedVerts[0].view;
			const isOrthographic = context.camera.type === CameraType.Orthographic;
			const dot = isOrthographic
				? -cullNormal.z
				: cullNormal.x * v0.x + cullNormal.y * v0.y + cullNormal.z * v0.z;

			if (!packet.material.doubleSided) {
				if (flipCulling ? dot < 0 : dot > 0) continue;
			}

			const projectedVerts: ProjectedVertex[] = [];
			for (const clipped of clippedVerts) {
				const projected = Matrix4.transformPoint(
					projectionMatrix,
					clipped.view
				);
				const rawW = projected.w ?? 0;
				const safeW = Math.abs(rawW) > 1e-6 ? rawW : rawW >= 0 ? 1e-6 : -1e-6;
				const ndcX = projected.x / safeW;
				const ndcY = projected.y / safeW;
				const ndcZ = projected.z / safeW;

				projectedVerts.push({
					x: (ndcX * 0.5 + 0.5) * targetWidth,
					y: (0.5 - ndcY * 0.5) * targetHeight,
					z: ndcZ,
					w: 1 / safeW,
					u: clipped.view.u,
					v: clipped.view.v,
					u2: clipped.view.u2,
					v2: clipped.view.v2,
					normal: clipped.view.normal,
					tangent: clipped.view.tangent,
					world: clipped.world,
					zView: clipped.view.z,
				});
			}

			// Triangulate clipped polygon into triangles
			for (let i = 1; i < clippedVerts.length - 1; i++) {
				const triClipped = [
					clippedVerts[0],
					clippedVerts[i],
					clippedVerts[i + 1],
				];
				const triProjected = [
					projectedVerts[0],
					projectedVerts[i],
					projectedVerts[i + 1],
				];

				let minDepth = Infinity;
				let maxDepth = -Infinity;
				let sumDepth = 0;
				let centerX = 0;
				let centerY = 0;
				let centerZ = 0;

				for (const clipped of triClipped) {
					const depth = -clipped.view.z;
					if (depth < minDepth) minDepth = depth;
					if (depth > maxDepth) maxDepth = depth;
					sumDepth += depth;
					centerX += clipped.world.x;
					centerY += clipped.world.y;
					centerZ += clipped.world.z;
				}

				projectedFaces.push({
					primitive: packet.primitive,
					material: packet.material,
					vertices: triClipped.map((vertex) => vertex.world),
					color: face.color,
					doubleSided: face.doubleSided,
					projected: triProjected,
					center: {
						x: centerX / 3,
						y: centerY / 3,
						z: centerZ / 3,
					},
					normal: face.normal
						? Vector3.normalize(
								Matrix4.transformNormal(packet.normalMatrix, face.normal)
							)
						: Vector3.calculateNormal(worldVerts),
					depthInfo: {
						min: minDepth,
						max: maxDepth,
						avg: sumDepth / 3,
					},
				});
			}
		}

		return projectedFaces;
	}

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

	public static getPacketFacesWithContext(
		packet: DrawPacket,
		context: FrameContext
	): PrimitiveFace[] {
		const overrides = context.transient.get(
			ANIMATION_SOFTWARE_DEFORMED_GEOMETRY_KEY
		) as Map<string, { positions?: Float32Array; normals?: Float32Array; tangents?: Float32Array }> | undefined;
		const geometryOverride = overrides?.get(packet.primitive.id);
		const triangleCount = (packet.geometry.indices.length / 3) | 0;
		const faces: PrimitiveFace[] = [];
		for (
			let triangleIndex = 0;
			triangleIndex < triangleCount;
			triangleIndex++
		) {
			const vertices = GeometryBuilder.createVerticesForTriangle(
				packet.primitive,
				triangleIndex,
				geometryOverride
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
}

function clipFaceToNearPlane(
	viewVerts: IVertex[],
	worldVerts: IVertex[],
	context: FrameContext
): ClippedVertexPair[] {
	const nearZ = -context.camera.near;
	const clippedVerts: ClippedVertexPair[] = [];

	for (let i = 0; i < viewVerts.length; i++) {
		const v1 = viewVerts[i];
		const w1 = worldVerts[i];
		const nextIndex = (i + 1) % viewVerts.length;
		const v2 = viewVerts[nextIndex];
		const w2 = worldVerts[nextIndex];
		const in1 = v1.z <= nearZ;
		const in2 = v2.z <= nearZ;

		if (in1) {
			if (in2) {
				clippedVerts.push({ view: v2, world: w2 });
				continue;
			}

			const t = (nearZ - v1.z) / (v2.z - v1.z);
			clippedVerts.push({
				view: interpolateVertex(v1, v2, t, nearZ),
				world: interpolateVertex(w1, w2, t),
			});
			continue;
		}

		if (in2) {
			const t = (nearZ - v1.z) / (v2.z - v1.z);
			clippedVerts.push({
				view: interpolateVertex(v1, v2, t, nearZ),
				world: interpolateVertex(w1, w2, t),
			});
			clippedVerts.push({ view: v2, world: w2 });
		}
	}

	return clippedVerts;
}

function interpolateVertex(
	from: IVertex,
	to: IVertex,
	t: number,
	overrideZ?: number
): IVertex {
	return {
		x: from.x + (to.x - from.x) * t,
		y: from.y + (to.y - from.y) * t,
		z: overrideZ ?? from.z + (to.z - from.z) * t,
		u: interpolateNumber(from.u, to.u, t),
		v: interpolateNumber(from.v, to.v, t),
		u2: interpolateNumber(from.u2, to.u2, t),
		v2: interpolateNumber(from.v2, to.v2, t),
		normal: interpolateNormal(from.normal, to.normal, t),
		tangent: interpolateTangent(from.tangent, to.tangent, t),
		color: from.color ?? to.color,
	};
}

function interpolateNumber(
	from: number | undefined,
	to: number | undefined,
	t: number
): number | undefined {
	if (from === undefined && to === undefined) return undefined;
	const start = from ?? to ?? 0;
	const end = to ?? from ?? 0;
	return start + (end - start) * t;
}

function interpolateNormal(
	from: IVertex["normal"],
	to: IVertex["normal"],
	t: number
): IVertex["normal"] {
	if (from && to) {
		return Vector3.normalize({
			x: from.x + (to.x - from.x) * t,
			y: from.y + (to.y - from.y) * t,
			z: from.z + (to.z - from.z) * t,
		});
	}

	return from || to || null;
}

function interpolateTangent(
	from: IVertex["tangent"],
	to: IVertex["tangent"],
	t: number
): IVertex["tangent"] {
	if (from && to) {
		const tangent = Vector3.normalize({
			x: from.x + (to.x - from.x) * t,
			y: from.y + (to.y - from.y) * t,
			z: from.z + (to.z - from.z) * t,
		});
		return {
			x: tangent.x,
			y: tangent.y,
			z: tangent.z,
			w: from.w,
		};
	}

	return from || to || null;
}
