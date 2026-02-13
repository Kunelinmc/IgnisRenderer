import { GLTFLoader } from "../src/loaders/GLTFLoader";

const COMPONENT_TYPE_FLOAT = 5126;
const COMPONENT_TYPE_UNSIGNED_SHORT = 5123;
const TYPE_VEC3 = "VEC3";

async function testSparse() {
	console.log("Starting Sparse Accessor Test...");
	const loader = new GLTFLoader();

	// 5 VEC3 elements (total 15 floats)
	// Base data: all 0.0
	const baseBuffer = new Float32Array(15).fill(0);
	// index 3: [0.5, 0.5, 0.5] (this will be overridden by sparse)
	baseBuffer[9] = 0.5;
	baseBuffer[10] = 0.5;
	baseBuffer[11] = 0.5;

	// Sparse indices: [1, 3] (index 1 and index 3)
	const indicesBuffer = new Uint16Array([1, 3]);

	// Sparse values:
	// for index 1: [1.0, 2.0, 3.0]
	// for index 3: [7.0, 8.0, 9.0]
	const valuesBuffer = new Float32Array([1, 2, 3, 7, 8, 9]);

	// Construct a binary buffer
	const totalSize =
		baseBuffer.byteLength + indicesBuffer.byteLength + valuesBuffer.byteLength;
	const combined = new Uint8Array(totalSize);
	combined.set(new Uint8Array(baseBuffer.buffer), 0);
	combined.set(new Uint8Array(indicesBuffer.buffer), baseBuffer.byteLength);
	combined.set(
		new Uint8Array(valuesBuffer.buffer),
		baseBuffer.byteLength + indicesBuffer.byteLength
	);

	const json = {
		accessors: [
			{
				bufferView: 0,
				byteOffset: 0,
				componentType: COMPONENT_TYPE_FLOAT,
				count: 5,
				type: TYPE_VEC3,
				sparse: {
					count: 2,
					indices: {
						bufferView: 1,
						byteOffset: 0,
						componentType: COMPONENT_TYPE_UNSIGNED_SHORT,
					},
					values: {
						bufferView: 2,
						byteOffset: 0,
					},
				},
			},
		],
		bufferViews: [
			{ buffer: 0, byteOffset: 0, byteLength: baseBuffer.byteLength },
			{
				buffer: 0,
				byteOffset: baseBuffer.byteLength,
				byteLength: indicesBuffer.byteLength,
			},
			{
				buffer: 0,
				byteOffset: baseBuffer.byteLength + indicesBuffer.byteLength,
				byteLength: valuesBuffer.byteLength,
			},
		],
	};

	const result = loader.getAccessorData(json, [combined], 0);

	console.log("Result data length:", result.length);

	const expected = [
		0,
		0,
		0, // index 0: base
		1,
		2,
		3, // index 1: sparse
		0,
		0,
		0, // index 2: base
		7,
		8,
		9, // index 3: sparse (overrides 0.5, 0.5, 0.5)
		0,
		0,
		0, // index 4: base
	];

	let success = true;
	for (let i = 0; i < expected.length; i++) {
		if (Math.abs(result[i] - expected[i]) > 0.0001) {
			console.error(
				`Mismatch at index ${i}: expected ${expected[i]}, got ${result[i]}`
			);
			success = false;
		}
	}

	if (success) {
		console.log("✅ Sparse Accessor Test Passed!");
	} else {
		console.log("❌ Sparse Accessor Test Failed!");
		process.exit(1);
	}
}

async function testSparseNoBase() {
	console.log("\nStarting Sparse Accessor (No Base BufferView) Test...");
	const loader = new GLTFLoader();

	// Sparse indices: [0, 4]
	const indicesBuffer = new Uint16Array([0, 4]);
	const valuesBuffer = new Float32Array([10, 20, 30, 40, 50, 60]);

	// Construct a binary buffer
	const combined = new Uint8Array(
		indicesBuffer.byteLength + valuesBuffer.byteLength
	);
	combined.set(new Uint8Array(indicesBuffer.buffer), 0);
	combined.set(new Uint8Array(valuesBuffer.buffer), indicesBuffer.byteLength);

	const json = {
		accessors: [
			{
				// bufferView is undefined
				componentType: COMPONENT_TYPE_FLOAT,
				count: 5,
				type: TYPE_VEC3,
				sparse: {
					count: 2,
					indices: {
						bufferView: 0,
						byteOffset: 0,
						componentType: COMPONENT_TYPE_UNSIGNED_SHORT,
					},
					values: {
						bufferView: 1,
						byteOffset: 0,
					},
				},
			},
		],
		bufferViews: [
			{ buffer: 0, byteOffset: 0, byteLength: indicesBuffer.byteLength },
			{
				buffer: 0,
				byteOffset: indicesBuffer.byteLength,
				byteLength: valuesBuffer.byteLength,
			},
		],
	};

	const result = loader.getAccessorData(json, [combined], 0);

	console.log("Result data length:", result.length);

	const expected = [
		10,
		20,
		30, // index 0: sparse
		0,
		0,
		0, // index 1: empty (0)
		0,
		0,
		0, // index 2: empty (0)
		0,
		0,
		0, // index 3: empty (0)
		40,
		50,
		60, // index 4: sparse
	];

	let success = true;
	for (let i = 0; i < expected.length; i++) {
		if (Math.abs(result[i] - expected[i]) > 0.0001) {
			console.error(
				`Mismatch at index ${i}: expected ${expected[i]}, got ${result[i]}`
			);
			success = false;
		}
	}

	if (success) {
		console.log("✅ Sparse Accessor (No Base) Test Passed!");
	} else {
		console.log("❌ Sparse Accessor (No Base) Test Failed!");
		process.exit(1);
	}
}

async function runAll() {
	await testSparse();
	await testSparseNoBase();
}

runAll().catch((err) => {
	console.error("Test error:", err);
	process.exit(1);
});
