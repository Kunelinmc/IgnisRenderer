import { performance } from "node:perf_hooks";

import { Material } from "../../src/materials/Material.ts";
import { MeshAsset } from "../../src/meshes/MeshAsset.ts";

const quick = process.argv.includes("--quick");
const configurations =
	quick ?
		[
			[10, 10_000],
			[10, 100_000],
		]
	:	[
			[1, 10_000],
			[10, 10_000],
			[10, 100_000],
		];

const material = new Material({ name: "BoundsBenchmark" });

function createPrimitive(id, vertexCount, offsetX) {
	const positions = new Float32Array(vertexCount * 3);
	for (let index = 0; index < vertexCount; index++) {
		positions[index * 3] = offsetX + (index % 101) * 0.01;
		positions[index * 3 + 1] = (index % 37) * 0.01;
		positions[index * 3 + 2] = (index % 17) * 0.01;
	}
	return {
		id: `bounds-${id}`,
		geometry: { positions, indices: new Uint32Array(0) },
		geometryVersion: 0,
		material,
		boundingSphere: { center: { x: 0, y: 0, z: 0 }, radius: 0 },
		boundingBox: {
			min: { x: 0, y: 0, z: 0 },
			max: { x: 0, y: 0, z: 0 },
		},
		visible: true,
		castShadows: true,
		receiveShadows: true,
	};
}

function median(values) {
	const sorted = values.slice().sort((left, right) => left - right);
	return sorted[Math.floor(sorted.length / 2)];
}

for (const [primitiveCount, verticesPerPrimitive] of configurations) {
	const primitives = Array.from({ length: primitiveCount }, (_, index) =>
		createPrimitive(index, verticesPerPrimitive, index * 4)
	);
	const mesh = new MeshAsset(primitives);
	for (let index = 0; index < 100; index++) void mesh.boundingBox;

	const cleanStart = performance.now();
	for (let index = 0; index < 100_000; index++) void mesh.boundingBox;
	const cleanGetterNanoseconds =
		((performance.now() - cleanStart) * 1_000_000) / 100_000;

	const dirtySamples = [];
	for (let index = 0; index < 12; index++) {
		const primitive = primitives[index % primitiveCount];
		primitive.geometry.positions[4] += 0.0001;
		mesh.markPrimitiveGeometryDirty(primitive);
		const start = performance.now();
		void mesh.boundingSphere;
		dirtySamples.push(performance.now() - start);
	}

	const centerChangePrimitive = primitives[primitiveCount - 1];
	centerChangePrimitive.geometry.positions[0] += 10;
	mesh.markPrimitiveGeometryDirty(centerChangePrimitive);
	const centerChangeStart = performance.now();
	void mesh.boundingSphere;
	const centerChangeMs = performance.now() - centerChangeStart;

	console.log(
		JSON.stringify({
			primitiveCount,
			totalVertices: primitiveCount * verticesPerPrimitive,
			cleanGetterNanoseconds,
			dirtyPrimitiveMedianMs: median(dirtySamples),
			centerChangeMs,
		})
	);
}
