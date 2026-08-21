import { expect, test } from "@playwright/test";

test("WebGPU batches compatible static meshes", async ({ page }) => {
	await page.goto("/");
	const result = await page.evaluate(async () => {
		if (!navigator.gpu) return { supported: false as const };
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) return { supported: false as const };

		const { Camera } = await import("/src/cameras/Camera.ts");
		const { PBRMaterial } = await import("/src/materials/PBRMaterial.ts");
		const { MeshAsset } = await import("/src/meshes/MeshAsset.ts");
		const { MeshInstance } = await import("/src/meshes/MeshInstance.ts");
		const { Renderer } = await import("/src/rendering/Renderer.ts");
		const { WebGPUBackend } = await import("/src/backends/webgpu/WebGPUBackend.ts");
		const { WebGPUCommandEncoder } = await import(
			"/src/backends/webgpu/WebGPUCommandEncoder.ts"
		);
		const instanceCounts: number[] = [];
		const originalDrawIndexed = WebGPUCommandEncoder.prototype.drawIndexed;
		WebGPUCommandEncoder.prototype.drawIndexed = function (...args) {
			instanceCounts.push(args[1] ?? 1);
			return originalDrawIndexed.apply(this, args);
		};

		const canvas = document.createElement("canvas");
		canvas.width = 64;
		canvas.height = 64;
		document.body.append(canvas);
		const camera = new Camera();
		const renderer = new Renderer(
			canvas,
			new WebGPUBackend({
				enableDeferredLighting: false,
				enableEarlyZPrepass: false,
			}),
			camera,
		);
		try {
			const material = new PBRMaterial({ name: "StaticBatchMaterial" });
			const mesh = MeshAsset.fromFaces([{
				material,
				vertices: [
					{ x: -0.3, y: -0.3, z: 0, normal: { x: 0, y: 0, z: 1 } },
					{ x: 0.3, y: -0.3, z: 0, normal: { x: 0, y: 0, z: 1 } },
					{ x: 0, y: 0.3, z: 0, normal: { x: 0, y: 0, z: 1 } },
				],
			}]);
			for (const x of [-0.4, 0.4]) {
				const instance = new MeshInstance({ mesh });
				instance.position.set(x, 0, -2);
				renderer.scene.add(instance);
			}
			try {
				await renderer.renderFrame(0);
			} catch (error) {
				if (String(error).includes("maxSampledTexturesPerShaderStage")) {
					return { supported: false as const };
				}
				throw error;
			}
			return { supported: true as const, instanceCounts };
		} finally {
			WebGPUCommandEncoder.prototype.drawIndexed = originalDrawIndexed;
			await renderer.destroy();
			canvas.remove();
		}
	});

	test.skip(!result.supported, "Chromium did not expose a WebGPU adapter.");
	if (!result.supported) return;
	expect(result.instanceCounts).toContain(2);
});
