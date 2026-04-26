import type { Camera } from "../cameras/Camera";
import { AlphaMode } from "../materials/Material";
import { isMaterialTransparentPass } from "../materials/transparency";
import { Matrix4 } from "../maths/Matrix4";
import type { Matrix3Arr } from "../maths/types";
import type { Renderer } from "../renderers/Renderer";
import type { IPrimitive } from "../core/types";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../core/types";
import { MeshInstance } from "../meshes";
import {
	DRAW_PACKET_FLAG_REFLECTIVE,
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	type DrawPacket,
	type PreparedScene,
} from "./types";

export class PreparedSceneBuilder {
	public static build(renderer: Renderer): PreparedScene {
		const opaquePackets: DrawPacket[] = [];
		const transparentPackets: DrawPacket[] = [];
		const shadowCasterPackets: DrawPacket[] = [];
		const shadowTransmitterPackets: DrawPacket[] = [];
		const reflectivePackets: DrawPacket[] = [];
		const meshInstances = renderer.scene.getMeshInstances();
		const renderableMeshInstances = meshInstances.filter(
			(meshInstance) => meshInstance.visible !== false
		);
		const bypassFrustumMeshInstances: MeshInstance[] = [];
		const frustumCulledMeshInstances: MeshInstance[] = [];

		for (const meshInstance of renderableMeshInstances) {
			if (isAnimationDrivenMeshInstance(meshInstance)) {
				bypassFrustumMeshInstances.push(meshInstance);
			} else {
				frustumCulledMeshInstances.push(meshInstance);
			}
		}

		const frustumVisibleMeshInstances = renderer.scene.queryMeshInstancesInFrustum(
			renderer.camera,
			frustumCulledMeshInstances
		);
		const cameraVisibleMeshInstances = new Set<MeshInstance>(
			frustumVisibleMeshInstances
		);
		for (const meshInstance of bypassFrustumMeshInstances) {
			cameraVisibleMeshInstances.add(meshInstance);
		}

		for (const meshInstance of renderableMeshInstances) {
			const visibleInCamera = cameraVisibleMeshInstances.has(meshInstance);
			const packets = this._buildMeshPackets(meshInstance, renderer.camera);
			for (const packet of packets) {
				if (packet.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) {
					if (visibleInCamera) {
						transparentPackets.push(packet);
					}
				} else if (visibleInCamera) {
					opaquePackets.push(packet);
				}

				if (packet.passFlags & DRAW_PACKET_FLAG_SHADOW_CASTER) {
					shadowCasterPackets.push(packet);
				}

				if (packet.passFlags & DRAW_PACKET_FLAG_SHADOW_TRANSMITTER) {
					shadowTransmitterPackets.push(packet);
				}

				if (packet.passFlags & DRAW_PACKET_FLAG_REFLECTIVE) {
					if (visibleInCamera) {
						reflectivePackets.push(packet);
					}
				}
			}
		}

		opaquePackets.sort(compareOpaquePackets);
		transparentPackets.sort(compareTransparentPackets);

		return {
			sceneBounds: renderer.scene.getBounds(),
			lights: renderer.scene.ecs.findLights(),
			particleSystems: renderer.scene.ecs.findParticleSystems(),
			hasActiveAnimations: renderer.animationSystem.hasActiveActions(),
			camera: renderer.camera,
			skybox: renderer.scene.skybox,
			meshInstances,
			shadowMaps: renderer.shadowMaps,
			opaquePackets,
			transparentPackets,
			shadowCasterPackets,
			shadowTransmitterPackets,
			reflectivePackets,
			spatialIndex: null,
		};
	}

	private static _buildMeshPackets(
		meshInstance: MeshInstance,
		camera: Camera
	): DrawPacket[] {
		const worldMatrix = meshInstance.worldMatrix;
		const normalMatrix = Matrix4.normalMatrix(worldMatrix) as Matrix3Arr;
		const cameraSpaceCenter = Matrix4.transformPoint(
			camera.viewMatrix,
			Matrix4.transformPoint(
				worldMatrix,
				meshInstance.mesh.boundingSphere.center
			)
		);
		const meshDepth = -cameraSpaceCenter.z;
		const worldScale = getMaxScaleFromMatrix(worldMatrix) || 1;

		return meshInstance.mesh.primitives
			.filter((primitive) => primitive.visible !== false)
			.map((primitive) =>
				this._createPacket(
					meshInstance,
					primitive,
					worldMatrix,
					normalMatrix,
					worldScale,
					meshDepth
				)
			);
	}

	private static _createPacket(
		meshInstance: MeshInstance,
		primitive: IPrimitive,
		worldMatrix: Matrix4,
		normalMatrix: Matrix3Arr,
		worldScale: number,
		meshDepth: number
	): DrawPacket {
		const material = primitive.material;
		const isTransparent = isMaterialTransparentPass(material);
		const isReflective =
			material.reflectivity > 0 && material.mirrorPlane !== null;
		const supportsShadowCasting =
			(primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) ===
			DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;

		let passFlags = 0;
		if (isTransparent) {
			passFlags |= DRAW_PACKET_FLAG_TRANSPARENT;
			if (primitive.castShadows && supportsShadowCasting) {
				passFlags |= DRAW_PACKET_FLAG_SHADOW_TRANSMITTER;
			}
		} else if (primitive.castShadows && supportsShadowCasting) {
			passFlags |= DRAW_PACKET_FLAG_SHADOW_CASTER;
		}

		if (isReflective) {
			passFlags |= DRAW_PACKET_FLAG_REFLECTIVE;
		}

		const worldCenter = Matrix4.transformPoint(
			worldMatrix,
			primitive.boundingSphere.center
		);

		return {
			id: `${meshInstance.id}:${primitive.id}`,
			meshInstance,
			mesh: meshInstance.mesh,
			primitive,
			material,
			geometry: primitive.geometry,
			worldMatrix,
			normalMatrix,
			worldBounds: {
				center: {
					x: worldCenter.x,
					y: worldCenter.y,
					z: worldCenter.z,
				},
				radius: primitive.boundingSphere.radius * worldScale,
			},
			sortDepth: meshDepth,
			pipelineKey: [
				material.type,
				material.shading,
				material.alphaMode ?? AlphaMode.Opaque,
				material.doubleSided ? "double" : "single",
			].join(":"),
			passFlags,
		};
	}
}

function compareOpaquePackets(left: DrawPacket, right: DrawPacket): number {
	const keyCompare = left.pipelineKey.localeCompare(right.pipelineKey);
	if (keyCompare !== 0) return keyCompare;

	if (left.material !== right.material) {
		return left.material.name.localeCompare(right.material.name);
	}

	if (left.geometry !== right.geometry) {
		return left.id.localeCompare(right.id);
	}

	return left.sortDepth - right.sortDepth;
}

function compareTransparentPackets(
	left: DrawPacket,
	right: DrawPacket
): number {
	if (left.sortDepth !== right.sortDepth) {
		return right.sortDepth - left.sortDepth;
	}

	return left.id.localeCompare(right.id);
}

function getMaxScaleFromMatrix(matrix: Matrix4): number {
	const elements = matrix.elements;
	const sx = Math.hypot(elements[0][0], elements[1][0], elements[2][0]);
	const sy = Math.hypot(elements[0][1], elements[1][1], elements[2][1]);
	const sz = Math.hypot(elements[0][2], elements[1][2], elements[2][2]);
	return Math.max(sx, sy, sz);
}

function isAnimationDrivenMeshInstance(meshInstance: MeshInstance): boolean {
	if (meshInstance.skeleton) return true;
	for (const primitive of meshInstance.mesh.primitives) {
		if ((primitive.geometry.morphTargets?.length ?? 0) > 0) {
			return true;
		}
	}
	return false;
}
