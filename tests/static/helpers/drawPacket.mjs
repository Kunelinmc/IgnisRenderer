import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import {
	DEFAULT_PRIMITIVE_DRAW_TOPOLOGY,
} from "../../../src/core/types.ts";

const EMPTY_GEOMETRY = Object.freeze({
	positions: new Float32Array(0),
	indices: new Uint32Array(0),
});

export function createTestDrawPacket(overrides = {}) {
	const id = overrides.id ?? "packet";
	const primitive = overrides.primitive ?? null;
	const material = overrides.material ?? primitive?.material ?? {};
	const geometryData = overrides.geometry ?? primitive?.geometry ?? EMPTY_GEOMETRY;
	const worldMatrix = overrides.worldMatrix ?? new Matrix4();
	const normalMatrix = overrides.normalMatrix ?? new Matrix4();
	const deformationRevision = overrides.deformationRevision ?? 0;
	const deformationMode = overrides.deformationMode ??
		(deformationRevision > 0 ? "skin" : "none");
	const meshInstance = overrides.meshInstance ?? null;
	return {
		submission: {
			id,
			source: overrides.source ?? {
				kind: "mesh-instance",
				instanceId: meshInstance?.id ?? id,
			},
			geometry: {
				resourceKey:
					overrides.geometryResourceKey ?? primitive ?? geometryData,
				id: overrides.geometryId ?? primitive?.id ?? `${id}:geometry`,
				data: geometryData,
				version:
					overrides.geometryVersion ?? primitive?.geometryVersion ?? 0,
				topology:
					overrides.topology ?? primitive?.topology ??
					DEFAULT_PRIMITIVE_DRAW_TOPOLOGY,
			},
			instance: {
				renderLayers:
					overrides.renderLayers ?? meshInstance?.renderLayers ?? 1,
				worldMatrix,
				...(overrides.previousWorldMatrix ? {
					previousWorldMatrix: overrides.previousWorldMatrix,
				} : {}),
				normalMatrix,
			},
			material: {
				effective: material,
				revision: overrides.materialRevision ?? material?.revision ?? 0,
				pipelineKey: overrides.pipelineKey ?? "test",
			},
			deformation: {
				mode: deformationMode,
				revision: deformationRevision,
				jointPayloadKey:
					overrides.jointPayloadKey ??
					(deformationMode === "skin" || deformationMode === "skin-morph" ?
						meshInstance?.id ?? id
						: null),
				morphPayloadKey:
					overrides.morphPayloadKey ??
					(deformationMode === "morph" || deformationMode === "skin-morph" ?
						id
						: null),
			},
			worldBounds: overrides.worldBounds ?? {
				center: { x: 0, y: 0, z: 0 },
				radius: 1,
			},
			passFlags: overrides.passFlags ?? 0,
		},
		sortDepth: overrides.sortDepth ?? 0,
	};
}
