import { performance } from "node:perf_hooks";

import { packWebGPUVertexGeometry } from "../../src/backends/webgpu/WebGPUGeometryPacking.ts";

const vertexCount = 1_000_000;
const positions = new Float32Array(vertexCount * 3);
const normals = new Float32Array(vertexCount * 3);
const uv0 = new Float32Array(vertexCount * 2);
for (let index = 0; index < vertexCount; index++) {
	normals[index * 3 + 2] = 1;
	uv0[index * 2] = (index % 1024) / 1023;
	uv0[index * 2 + 1] = (Math.floor(index / 1024) % 1024) / 1023;
}

const start = performance.now();
const packed = packWebGPUVertexGeometry({
	positions,
	normals,
	uv0,
	indices: new Uint32Array(0),
}, vertexCount);
const elapsedMs = performance.now() - start;

console.log(JSON.stringify({
	vertexCount,
	vertexBytes: packed.vertexByteLength,
	bytesPerVertex: packed.vertexByteLength / vertexCount,
	legacyVertexBytes: vertexCount * 136,
	reductionPercent:
		(1 - packed.vertexByteLength / (vertexCount * 136)) * 100,
	packingMs: elapsedMs,
}, null, 2));
