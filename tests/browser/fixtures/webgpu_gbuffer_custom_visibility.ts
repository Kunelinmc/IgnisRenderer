import {
	MeshAsset, MeshInstance, ShaderMaterial,
	type Renderer, type WebGPUBackend,
} from "../../../src/index.ts";
import { float16BitsToFloat32 } from "../../../src/foundation/Float16.ts";
import { Logger } from "../../../src/foundation/Logger.ts";
import { AlphaMode, type Material } from "../../../src/materials/Material.ts";

export interface CustomGBufferVisibilityResult {
	earlyZ: boolean;
	sourcePreserved: boolean;
	rows: { case: string; reverse: boolean; coplanarWinner: "front" | "rear" | null;
		error: number; baselineDifference: number;
		extensions: boolean; lighting: boolean; earlyZDraws: number }[];
	negative: { entry: "base" | "extensions"; rejected: boolean; recordedPasses: string[];
		recovered: boolean; message: string }[];
}

interface SurfaceSample {
	color: number[];
	ext0: number[];
	ext3: number[];
	labels: readonly string[];
	earlyZDraws: number;
}

const FRONT_EXT0 = [0.25, 0.5, 0.75, 400];
const REAR_EXT0 = [0.75, 0.25, 0.5, 900];
const FRONT_EXT3 = [123, 456, 40000, 7];
const REAR_EXT3 = [53123, 49456, 12000, 11];

function difference(a: readonly number[], b: readonly number[]): number {
	if (a.length !== b.length) return Number.POSITIVE_INFINITY;
	return Math.max(0, ...a.map((value, i) => Math.abs(value - b[i])));
}

function surfaceDifference(a: SurfaceSample, b: SurfaceSample): number {
	return Math.max(difference(a.color, b.color), difference(a.ext0, b.ext0),
		difference(a.ext3, b.ext3));
}

/** @internal Browser GPU regression helper; use Renderer for application rendering. */
export async function runCustomGBufferVisibility(options: {
	renderer: Renderer;
	backend: WebGPUBackend;
	device: GPUDevice;
	canvas: HTMLCanvasElement;
	earlyZ: boolean;
	sample: (reversePackets?: boolean) => Promise<number[]>;
	getPassLabels: () => readonly string[];
	getOpaqueMaterialOrder: () => readonly Material[];
}): Promise<CustomGBufferVisibilityResult> {
	const {
		renderer, backend, device, canvas, earlyZ, sample, getPassLabels, getOpaqueMaterialOrder,
	} = options;
	const response = await fetch(new URL("./webgpu_gbuffer_custom_probe.wgsl", import.meta.url));
	if (!response.ok) throw new Error(`Custom G-buffer shader fetch failed: ${response.status}`);
	const code = await response.text();
	const result: CustomGBufferVisibilityResult = {
		earlyZ, sourcePreserved: false, rows: [], negative: [],
	};
	const textures = new Map<"ext0" | "ext3", GPUTexture>();
	const createTexture = device.createTexture;
	const createEncoder = device.createCommandEncoder;
	let earlyZDraws = 0;
	device.createTexture = function (descriptor) {
		const texture = createTexture.call(this, descriptor);
		if (descriptor.label?.startsWith("WebGPUGBufferDeferredStorageRGBA16")) {
			if (descriptor.format === "rgba16float") textures.set("ext0", texture);
			if (descriptor.format === "rgba16uint") textures.set("ext3", texture);
		}
		return texture;
	};
	device.createCommandEncoder = function (descriptor) {
		const encoder = createEncoder.call(this, descriptor);
		const beginPass = encoder.beginRenderPass;
		encoder.beginRenderPass = function (passDescriptor) {
			const pass = beginPass.call(this, passDescriptor);
			if (passDescriptor.label?.startsWith("WebGPUEarlyZ")) {
				const draw = pass.draw;
				const drawIndexed = pass.drawIndexed;
				pass.draw = function (...args) { earlyZDraws++; draw.apply(this, args); };
				pass.drawIndexed = function (...args) {
					earlyZDraws++;
					drawIndexed.apply(this, args);
				};
			}
			return pass;
		};
		return encoder;
	};
	const makeMaterial = (front: boolean) => new ShaderMaterial({
		deferredLighting: true, vertexEntryPoint: "vsProbe",
		fragmentDeferredBaseEntryPoint: "fsProbeBase",
		fragmentDeferredExtensionsEntryPoint: "fsProbeExtensions",
		uniformBindings: [
			{ name: "color", type: "vec4f", value: front ? [0.8, 0.1, 0.05, 1]
				: [0.05, 0.3, 0.9, 1] },
			{ name: "ext0", type: "vec4f", value: front ? FRONT_EXT0 : REAR_EXT0 },
			{ name: "ext3", type: "vec4u", value: front ? FRONT_EXT3 : REAR_EXT3 },
			{ name: "controls", type: "vec4f", value: [-1, 0, 1, 0] },
		],
		chunks: [
			{ backend: "webgpu", language: "wgsl", stage: "vertex", code },
			{ backend: "webgpu", language: "wgsl", stage: "fragment", mode: "deferred", code },
		],
	});
	const frontMaterial = makeMaterial(true);
	const rearMaterial = makeMaterial(false);
	const makePlane = (material: ShaderMaterial) => new MeshInstance({
		mesh: MeshAsset.fromFaces([{ material, vertices: [
			{ x: -1, y: -1, z: -2, normal: { x: 0, y: 0, z: 1 } },
			{ x: 1, y: -1, z: -2, normal: { x: 0, y: 0, z: 1 } },
			{ x: 1, y: 1, z: -2, normal: { x: 0, y: 0, z: 1 } },
			{ x: -1, y: 1, z: -2, normal: { x: 0, y: 0, z: 1 } },
		] }]),
	});
	const front = makePlane(frontMaterial);
	const rear = makePlane(rearMaterial);
	const removePlanes = () => {
		renderer.scene.remove(front);
		renderer.scene.remove(rear);
	};
	const readExtensions = async (): Promise<{ ext0: number[]; ext3: number[] }> => {
		const entries = (["ext0", "ext3"] as const).map((key) => {
			const texture = textures.get(key);
			if (!texture || texture.width !== canvas.width || texture.height !== canvas.height) {
				throw new Error(`Actual renderer G-buffer ${key} texture was not captured.`);
			}
			const buffer = device.createBuffer({ size: 256 * 4,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
			return { key, texture, buffer };
		});
		const encoder = device.createCommandEncoder({ label: "CustomGBufferReadback" });
		for (const { texture, buffer } of entries) encoder.copyTextureToBuffer({ texture,
			origin: { x: canvas.width / 2 - 2, y: canvas.height / 2 - 2 } },
			{ buffer, bytesPerRow: 256 }, { width: 4, height: 4 });
		device.queue.submit([encoder.finish()]);
		const payload = { ext0: [] as number[], ext3: [] as number[] };
		try {
			await Promise.all(entries.map(({ buffer }) => buffer.mapAsync(GPUMapMode.READ)));
			for (const { key, buffer } of entries) {
				const values = new Uint16Array(buffer.getMappedRange());
				for (let y = 0; y < 4; y++) for (let x = 0; x < 16; x++) {
					const value = values[y * 128 + x];
					payload[key].push(key === "ext0" ? float16BitsToFloat32(value) : value);
				}
			}
			return payload;
		} finally {
			for (const { buffer } of entries) buffer.destroy();
		}
	};
	const capture = async (reversePackets = false): Promise<SurfaceSample> => {
		earlyZDraws = 0;
		const color = await sample(reversePackets);
		const labels = [...getPassLabels()];
		if (!labels.includes("WebGPUGBufferExtensions") ||
			!labels.includes("WebGPUDeferredLighting")) {
			throw new Error(`Custom ShaderMaterial did not execute deferred passes: ${labels.join(", ")}`);
		}
		const extensions = await readExtensions();
		return { color, ...extensions, labels, earlyZDraws };
	};
	const captureOnly = async (plane: MeshInstance) => {
		removePlanes();
		renderer.scene.add(plane);
		const value = await capture();
		removePlanes();
		return value;
	};
	const addRow = (name: string, reverse: boolean, actual: SurfaceSample,
		expected: SurfaceSample, baselineDifference: number,
		coplanarWinner: "front" | "rear" | null) => {
		result.rows.push({ case: name, reverse, coplanarWinner,
			error: surfaceDifference(actual, expected),
			baselineDifference, extensions: actual.labels.includes("WebGPUGBufferExtensions"),
			lighting: actual.labels.includes("WebGPUDeferredLighting"), earlyZDraws: actual.earlyZDraws });
	};
	try {
		result.sourcePreserved = frontMaterial.resolveWebGPUProgram("deferred")
			.fragmentCode.trim() === code.trim();
		rear.setPosition(0, 0, -0.5);
		const frontOnly = await captureOnly(front);
		const rearOnly = await captureOnly(rear);
		for (const [baseline, ext0, ext3] of [
			[frontOnly, FRONT_EXT0, FRONT_EXT3], [rearOnly, REAR_EXT0, REAR_EXT3],
		] as const) {
			for (let i = 0; i < 64; i++) {
				if (baseline.ext0[i] !== ext0[i % 4] || baseline.ext3[i] !== ext3[i % 4]) {
					throw new Error("Custom baseline did not write its declared Ext0/Ext3 payload.");
				}
			}
			if (baseline.earlyZDraws !== 0) throw new Error("Custom material entered early-Z.");
		}
		const baselineDifference = surfaceDifference(frontOnly, rearOnly);
		if (difference(frontOnly.color, rearOnly.color) < 20) {
			throw new Error("Custom base outputs do not produce distinguishable rendered pixels.");
		}
		for (const name of ["hidden-rear", "coplanar", "masked-discard", "fragment-depth-pull",
			"fragment-depth-push", "sample-mask-zero"]) {
			rear.setPosition(0, 0, name === "coplanar" ? 0 : -0.5);
			frontMaterial.alphaMode = name === "masked-discard" ? AlphaMode.Mask : AlphaMode.Opaque;
			frontMaterial.setUniform("controls", [name === "masked-discard" ? canvas.width / 2 : -1,
				name === "fragment-depth-push" ? 0.03 : 0, name === "sample-mask-zero" ? 0 : 1, 0]);
			rearMaterial.setUniform("controls", [-1, name === "fragment-depth-pull" ? -0.03 : 0, 1, 0]);
			let expected = frontOnly;
			if (["fragment-depth-pull", "fragment-depth-push", "sample-mask-zero"].includes(name)) {
				expected = rearOnly;
			}
			if (name === "masked-discard") {
				const split = (a: number[], b: number[]) => a.map((value, i) =>
					Math.floor(i / 4) % 4 < 2 ? b[i] : value);
				expected = { ...frontOnly, color: split(frontOnly.color, rearOnly.color),
					ext0: split(frontOnly.ext0, rearOnly.ext0), ext3: split(frontOnly.ext3, rearOnly.ext3) };
			}
			for (const reverse of [false, true]) {
				for (const plane of reverse ? [rear, front] : [front, rear]) renderer.scene.add(plane);
				const actual = await capture(reverse);
				let coplanarWinner: "front" | "rear" | null = null;
				if (name === "coplanar") {
					const order = getOpaqueMaterialOrder();
					const lastMaterial = order.at(-1);
					if (order.length !== 2 ||
						(lastMaterial !== frontMaterial && lastMaterial !== rearMaterial)) {
						throw new Error("Custom visibility test did not record both materials.");
					}
					coplanarWinner = lastMaterial === frontMaterial ? "front" : "rear";
					expected = coplanarWinner === "front" ? frontOnly : rearOnly;
				}
				addRow(name, reverse, actual, expected, baselineDifference, coplanarWinner);
				removePlanes();
			}
		}
		frontMaterial.alphaMode = AlphaMode.Opaque;
		frontMaterial.setUniform("controls", [-1, 0, 1, 0]);
		const runtimeMode = backend.shaderRuntime.getMode();
		backend.shaderRuntime.setMode("warn");
		try {
			for (const entry of ["base", "extensions"] as const) {
				const property = entry === "base" ? "fragmentDeferredBaseEntryPoint"
					: "fragmentDeferredExtensionsEntryPoint";
				const valid = frontMaterial[property];
				const missing = entry === "base" ? "missingProbeBase" : "missingProbeExtensions";
				frontMaterial[property] = missing;
				renderer.scene.add(front);
				const warn = Logger.warn;
				const error = Logger.error;
				let rejected = false;
				let message = "";
				device.pushErrorScope("validation");
				Logger.warn = () => true;
				Logger.error = () => true;
				let recordedPasses: string[];
				try {
					try { await sample(); }
					catch (failure) { rejected = true; message = String(failure); }
					recordedPasses = [...getPassLabels()];
				} finally {
					Logger.warn = warn;
					Logger.error = error;
					await device.popErrorScope();
					frontMaterial[property] = valid;
				}
				if (!rejected || !message.includes(missing)) {
					throw new Error(`Invalid custom ${entry} entry was not rejected clearly: ${message}`);
				}
				const actual = await capture();
				result.negative.push({ entry, rejected, recordedPasses,
					recovered: surfaceDifference(actual, frontOnly) === 0, message });
				removePlanes();
			}
		} finally {
			backend.shaderRuntime.setMode(runtimeMode);
		}
		return result;
	} finally {
		removePlanes();
		device.createTexture = createTexture;
		device.createCommandEncoder = createEncoder;
	}
}
