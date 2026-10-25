import { createBrowserTestSession, createTestCanvas } from "../helpers/browserTestSession.ts";
import type { CustomGBufferVisibilityResult } from "./webgpu_gbuffer_custom_visibility.ts";
import type { Material } from "../../../src/materials/Material.ts";

/** @internal Real-renderer regression for extension payload visibility. */
export function setup() {
	return createBrowserTestSession(async (defer) => {
		const adapter = await navigator.gpu?.requestAdapter();
		if (!adapter) return { supported: false as const, reason: "No WebGPU adapter." };
		const {
			Camera, MeshAsset, MeshInstance, PBRMaterial, Renderer, Texture,
			TextureFormat, WebGPUBackend,
		} = await import("../../../src/index.ts");
		const errors: string[] = [];
		let activeDevice: GPUDevice | null = null;
		let passLabels: string[] = [];
		const samples: { extensions: boolean; lighting: boolean }[] = [];
		const customResults: CustomGBufferVisibilityResult[] = [];
		const requestDevice = GPUAdapter.prototype.requestDevice;
		GPUAdapter.prototype.requestDevice = async function (descriptor) {
			const device = await requestDevice.call(this, descriptor);
			activeDevice = device;
			device.addEventListener("uncapturederror", (event) => errors.push(event.error.message));
			const createEncoder = device.createCommandEncoder.bind(device);
			device.createCommandEncoder = (options) => {
				const encoder = createEncoder(options);
				const beginPass = encoder.beginRenderPass.bind(encoder);
				encoder.beginRenderPass = (descriptor) => {
					passLabels.push(descriptor.label ?? "");
					return beginPass(descriptor);
				};
				return encoder;
			};
			return device;
		};
		defer(() => { GPUAdapter.prototype.requestDevice = requestDevice; });
		const rows: { lobe: string; earlyZ: boolean; coplanar: boolean;
			reverse: boolean; winner: "front" | "rear"; error: number; baselineDifference: number }[] = [];
		for (const earlyZ of [false, true]) {
			const canvas = createTestCanvas(defer, 64, 64);
			const camera = new Camera({ near: 0.1, far: 20, aspectRatio: 1 });
			const backend = new WebGPUBackend({ enableEarlyZPrepass: earlyZ, shaderMode: "warn" });
			const beginFrame = backend.beginFrame.bind(backend);
			let reverseOpaquePackets = false;
			let opaqueMaterialOrder: readonly Material[] = [];
			backend.beginFrame = (context) => {
				// Reverse prepared packets explicitly: scene insertion order may be sorted away.
				if (reverseOpaquePackets) context.scene.opaquePackets.reverse();
				opaqueMaterialOrder = context.scene.opaquePackets.map((packet) =>
					packet.submission.material.effective);
				return beginFrame(context);
			};
			const renderer = new Renderer(canvas, backend, camera);
			defer(() => renderer.destroy());
			renderer.features.enableSH = false;
			renderer.features.enableReflection = false;
			const data = new Float32Array(64 * 32 * 4);
			for (let y = 0; y < 32; y++) for (let x = 0; x < 64; x++) {
				const phi = (x + 0.5) / 64 * Math.PI * 2 - Math.PI;
				const i = (y * 64 + x) * 4;
				data[i] = 1.5 + Math.sin(phi);
				data[i + 1] = 0.2 + y / 32;
				data[i + 2] = 1.5 + Math.cos(phi);
				data[i + 3] = 1;
			}
			const environment = new Texture({ data, width: 64, height: 32,
				format: TextureFormat.RGBA16Float, colorSpace: "HDR" });
			defer(() => environment.dispose());
			renderer.scene.environment.iblTexture = environment;
			renderer.scene.environment.backgroundEnabled = false;
			await renderer.initialize();
			const capture = document.createElement("canvas");
			capture.width = capture.height = 64;
			const ctx = capture.getContext("2d", { willReadFrequently: true })!;
			const sample = async (reversePackets = false) => {
				passLabels = [];
				reverseOpaquePackets = reversePackets;
				try {
					renderer.requestRender("scene");
					await renderer.renderFrame(performance.now());
				} finally {
					reverseOpaquePackets = false;
				}
				samples.push({ extensions: passLabels.includes("WebGPUGBufferExtensions"),
					lighting: passLabels.includes("WebGPUDeferredLighting") });
				ctx.drawImage(canvas, 0, 0, 64, 64);
				return Array.from(ctx.getImageData(30, 30, 4, 4).data);
			};
			const difference = (a: number[], b: number[]) =>
				Math.max(...a.map((value, i) => Math.abs(value - b[i])));
			const { runCustomGBufferVisibility } = await import("./webgpu_gbuffer_custom_visibility.ts");
			customResults.push(await runCustomGBufferVisibility({
				renderer, backend, device: activeDevice!, canvas, earlyZ, sample,
				getPassLabels: () => passLabels,
				getOpaqueMaterialOrder: () => opaqueMaterialOrder,
			}));
			for (const lobe of ["clearcoat", "iridescence", "anisotropy"]) {
				const frontMaterial = new PBRMaterial({ albedo: { r: 60, g: 20, b: 10 },
					roughness: 0.12, clearcoat: lobe === "clearcoat" ? 1 : 0,
					iridescenceFactor: lobe === "iridescence" ? 1 : 0,
					iridescenceThicknessMaximum: 400,
					anisotropyStrength: lobe === "anisotropy" ? 0.9 : 0 });
				const rearMaterial = new PBRMaterial({ albedo: { r: 10, g: 50, b: 90 },
					roughness: 0.12, clearcoat: lobe === "clearcoat" ? 1 : 0,
					iridescenceFactor: lobe === "iridescence" ? 0.8 : 0,
					iridescenceThicknessMaximum: 950,
					anisotropyStrength: lobe === "anisotropy" ? 0.5 : 0,
					anisotropyRotation: 1.2 });
				const plane = (material: InstanceType<typeof PBRMaterial>, nx: number) => {
					const mesh = MeshAsset.fromFaces([{ material, vertices: [
						{ x: -1, y: -1, z: -2, normal: { x: nx, y: 0, z: Math.sqrt(1 - nx * nx) } },
						{ x: 1, y: -1, z: -2, normal: { x: nx, y: 0, z: Math.sqrt(1 - nx * nx) } },
						{ x: 1, y: 1, z: -2, normal: { x: nx, y: 0, z: Math.sqrt(1 - nx * nx) } },
						{ x: -1, y: 1, z: -2, normal: { x: nx, y: 0, z: Math.sqrt(1 - nx * nx) } },
					] }]);
					return new MeshInstance({ mesh });
				};
				const front = plane(frontMaterial, 0);
				const rear = plane(rearMaterial, 0.6);
				for (const coplanar of [false, true]) {
					rear.setPosition(0, 0, coplanar ? 0 : -0.5);
					renderer.scene.add(front);
					const frontOnly = await sample();
					renderer.scene.remove(front);
					renderer.scene.add(rear);
					const rearOnly = await sample();
					renderer.scene.remove(rear);
					for (const reverse of [false, true]) {
						for (const node of reverse ? [rear, front] : [front, rear]) renderer.scene.add(node);
						const combined = await sample(reverse);
						const lastMaterial = opaqueMaterialOrder.at(-1);
						if (opaqueMaterialOrder.length !== 2 ||
							(lastMaterial !== frontMaterial && lastMaterial !== rearMaterial)) {
							throw new Error("G-buffer visibility test did not record both materials.");
						}
						const winner = coplanar && lastMaterial === rearMaterial ? "rear" : "front";
						const error = difference(combined, winner === "front" ? frontOnly : rearOnly);
						rows.push({ lobe, earlyZ, coplanar, reverse, winner, error,
							baselineDifference: difference(frontOnly, rearOnly) });
						renderer.scene.remove(front);
						renderer.scene.remove(rear);
					}
				}
			}
			await activeDevice!.queue.onSubmittedWorkDone();
			await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
			await renderer.destroy();
		}
		return { supported: true as const, errors, rows, samples, customResults };
	});
}
