import type { Camera } from "../cameras/Camera";
import { Matrix4 } from "../maths/Matrix4";
import type { Matrix3Arr } from "../maths/types";
import type { Renderer } from "../renderers/Renderer";
import type { IModel, IPrimitive } from "../core/types";
import { AlphaMode } from "../materials/Material";
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

		for (const model of renderer.scene.models) {
			if (model.visible === false) continue;

			const packets = this._buildModelPackets(model, renderer.camera);
			for (const packet of packets) {
				const visibleInCamera = renderer.camera.isSphereInFrustum(
					packet.worldBounds.center,
					packet.worldBounds.radius
				);

				if (packet.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) {
					if (visibleInCamera) {
						transparentPackets.push(packet);
					}
				} else {
					if (visibleInCamera) {
						opaquePackets.push(packet);
					}
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
			lights: renderer.scene.lights.slice(),
			particleSystems: renderer.scene.particleSystems.slice(),
			camera: renderer.camera,
			skybox: renderer.scene.skybox,
			models: renderer.scene.models.slice(),
			shadowMaps: renderer.shadowMaps,
			opaquePackets,
			transparentPackets,
			shadowCasterPackets,
			shadowTransmitterPackets,
			reflectivePackets,
		};
	}

	private static _buildModelPackets(
		model: IModel,
		camera: Camera
	): DrawPacket[] {
		const worldMatrix = Matrix4.fromTransform(model.transform);
		const normalMatrix = Matrix4.normalMatrix(worldMatrix) as Matrix3Arr;
		const cameraSpaceCenter = Matrix4.transformPoint(
			camera.viewMatrix,
			Matrix4.transformPoint(worldMatrix, model.boundingSphere.center)
		);
		const modelDepth = -cameraSpaceCenter.z;
		const worldScale =
			Math.max(
				Math.abs(model.transform.scale.x),
				Math.abs(model.transform.scale.y),
				Math.abs(model.transform.scale.z)
			) || 1;

		return model.primitives
			.filter((primitive) => primitive.visible !== false)
			.map((primitive) =>
				this._createPacket(
					model,
					primitive,
					worldMatrix,
					normalMatrix,
					worldScale,
					modelDepth
				)
			);
	}

	private static _createPacket(
		model: IModel,
		primitive: IPrimitive,
		worldMatrix: Matrix4,
		normalMatrix: Matrix3Arr,
		worldScale: number,
		modelDepth: number
	): DrawPacket {
		const material = primitive.material;
		const alphaMode = material.alphaMode ?? AlphaMode.Opaque;
		const isTransparent = alphaMode === AlphaMode.Blend;
		const isReflective =
			material.reflectivity > 0 && material.mirrorPlane !== null;

		let passFlags = 0;
		if (isTransparent) {
			passFlags |= DRAW_PACKET_FLAG_TRANSPARENT;
			if (primitive.castShadows) {
				passFlags |= DRAW_PACKET_FLAG_SHADOW_TRANSMITTER;
			}
		} else if (primitive.castShadows) {
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
			id: `${model.id}:${primitive.id}`,
			model,
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
			sortDepth: modelDepth,
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
