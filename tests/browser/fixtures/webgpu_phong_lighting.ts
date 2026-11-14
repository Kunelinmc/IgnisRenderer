import { createBrowserTestSession, createTestCanvas } from "../helpers/browserTestSession.ts";

/** @internal Verifies Phong light direction using a WebGPU render-target readback. */
export function setup() {
	return createBrowserTestSession(async (defer) => {
		const adapter = await navigator.gpu?.requestAdapter();
		if (!adapter) return { supported: false as const, reason: "No WebGPU adapter." };
		const { WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT } =
			await import("../../../src/backends/webgpu/constants.ts");
		if (adapter.limits.maxSampledTexturesPerShaderStage <
			WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT) {
			return { supported: false as const,
				reason: "WebGPU adapter cannot support the engine's sampled-texture layout." };
		}
		const {
			Camera, DirectionalLight, MeshAsset, MeshInstance, PhongMaterial, Renderer,
			TextureFormat, WebGPUBackend,
		} = await import("../../../src/index.ts");
		const errors: string[] = [];
		const requestDevice = GPUAdapter.prototype.requestDevice;
		GPUAdapter.prototype.requestDevice = async function (descriptor) {
			const device = await requestDevice.call(this, descriptor);
			device.addEventListener("uncapturederror", (event) => errors.push(event.error.message));
			return device;
		};
		defer(() => { GPUAdapter.prototype.requestDevice = requestDevice; });
		const canvas = createTestCanvas(defer, 16, 16);
		const camera = new Camera({ near: 0.1, far: 20, aspectRatio: 1 });
		const renderer = new Renderer(canvas, new WebGPUBackend(), camera);
		defer(() => renderer.destroy());
		renderer.scene.environment.backgroundEnabled = false;
		renderer.features.enableReflection = false;
		const material = new PhongMaterial({
			diffuse: { r: 255, g: 255, b: 255 }, ambient: { r: 0, g: 0, b: 0 },
		});
		const mesh = MeshAsset.fromFaces([{
			material,
			vertices: [
				{ x: -2, y: -2, z: -2, normal: { x: 0, y: 0, z: 1 } },
				{ x: 2, y: -2, z: -2, normal: { x: 0, y: 0, z: 1 } },
				{ x: 2, y: 2, z: -2, normal: { x: 0, y: 0, z: 1 } },
				{ x: -2, y: 2, z: -2, normal: { x: 0, y: 0, z: 1 } },
			],
		}]);
		renderer.scene.add(new MeshInstance({ mesh }));
		const light = new DirectionalLight({
			intensity: 0.1, direction: { x: 0, y: 0, z: -1 },
		});
		renderer.scene.add(light);
		await renderer.initialize();
		const target = renderer.renderTargets.create({
			size: { mode: "fixed", width: 1, height: 1 },
			color: [{ format: TextureFormat.RGBA16Float }],
			depth: { format: TextureFormat.Depth32Float },
		});
		defer(() => target.destroy());
		const sample = async () => {
			const ticket = target.enqueueJob({
				kind: "scene-view", camera,
				content: { environment: false, particles: false, shadows: "disabled" },
				readback: { attachmentIndex: 0 },
			});
			renderer.requestRender("unknown");
			await renderer.renderFrame(performance.now());
			const pixel = (await ticket.done).readback?.toRGBAFloat32();
			if (!pixel) throw new Error("Missing Phong lighting readback.");
			return pixel[0];
		};
		const lit = await sample();
		light.direction = { x: 0, y: 0, z: 1 };
		const backlit = await sample();
		return { supported: true as const, errors, lit, backlit };
	});
}
