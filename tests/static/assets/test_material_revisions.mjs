import assert from "node:assert/strict";

import { Texture } from "../../../src/core/Texture.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";

{
	const material = new PBRMaterial();
	const initialRevision = material.revision;
	const initialContentRevision = PBRMaterial.contentRevision;
	material.roughness = 0.25;
	assert.ok(material.revision > initialRevision);
	assert.ok(PBRMaterial.contentRevision > initialContentRevision);

	const nestedRevision = material.revision;
	material.albedo.r = 64;
	assert.ok(material.revision > nestedRevision);
}

{
	const texture = new Texture({
		width: 1,
		height: 1,
		data: new Uint8Array([255, 255, 255, 255]),
	});
	const material = new PBRMaterial({ albedoMap: texture });
	const materialRevision = material.revision;
	const samplingRevision = texture.samplingRevision;
	texture.offset.x = 0.5;
	assert.ok(texture.samplingRevision > samplingRevision);
	assert.ok(material.revision > materialRevision);
}

{
	const material = new PBRMaterial();
	const mesh = MeshAsset.fromFaces([{
		material,
		vertices: [
			{ x: 0, y: 0, z: 0 },
			{ x: 1, y: 0, z: 0 },
			{ x: 0, y: 1, z: 0 },
		],
	}]);
	const instance = new MeshInstance({ mesh });
	const initial = instance.worldTransformRevision;
	instance.updateWorldMatrix();
	assert.equal(instance.worldTransformRevision, initial);
	instance.position.x = 2;
	instance.updateWorldMatrix();
	assert.ok(instance.worldTransformRevision > initial);
}

console.log("Material and transform revision tests passed");
