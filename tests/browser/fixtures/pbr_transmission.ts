import { createBrowserTestSession, createTestCanvas } from "../helpers/browserTestSession.ts";

/** @internal Browser lighting regression; invoke through withBrowserFixture(). */
export function setup(backendKind: "webgpu" | "webgl") {
	return createBrowserTestSession(async (defer) => {
		if (backendKind === "webgpu") {
			const adapter = await navigator.gpu?.requestAdapter();
			if (!adapter) return { supported: false as const, reason: "No WebGPU adapter." };
			const { WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT } =
				await import("../../../src/backends/webgpu/constants.ts");
			if (adapter.limits.maxSampledTexturesPerShaderStage <
				WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT) {
				return { supported: false as const,
					reason: "WebGPU adapter cannot support the engine's sampled-texture layout." };
			}
		}
		const {
			AreaLight, Camera, DirectionalLight, MeshAsset, MeshInstance,
			PBRMaterial, PointLight, Renderer, SpotLight, TextureFormat, WebGLBackend, WebGPUBackend,
		} = await import("../../../src/index.ts");
		const errors: string[] = [];
		if (backendKind === "webgpu") {
			const requestDevice = GPUAdapter.prototype.requestDevice;
			GPUAdapter.prototype.requestDevice = async function (descriptor) {
				const device = await requestDevice.call(this, descriptor);
				device.addEventListener("uncapturederror", (event) => errors.push(event.error.message));
				return device;
			};
			defer(() => { GPUAdapter.prototype.requestDevice = requestDevice; });
		}
		const camera = new Camera({ near: 0.1, far: 20, aspectRatio: 1 });
		const canvas = createTestCanvas(defer, 16, 16);
		const backend = backendKind === "webgl" ?
			new WebGLBackend({ shaderMode: "strict" }) : new WebGPUBackend();
		const renderer = new Renderer(canvas, backend, camera);
		defer(() => renderer.destroy());
		renderer.features.enableEnvironment = false;
		renderer.features.enableReflection = false;
		const material = new PBRMaterial({
			albedo: { r: 255, g: 255, b: 255 }, metalness: 0,
			roughness: 0.08, transmissionFactor: 1, ior: 1.5,
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
		for (const primitive of mesh.primitives) primitive.castShadows = false;
		renderer.scene.add(new MeshInstance({ mesh }));
		await renderer.initialize();
		const gl = backendKind === "webgl" ? canvas.getContext("webgl2") : null;
		const target = renderer.renderTargets.create({
			size: { mode: "fixed", width: 1, height: 1 },
			color: [{ format: TextureFormat.RGBA16Float }],
			depth: { format: TextureFormat.Depth32Float },
		});
		defer(() => target.destroy());
		const sample = async (roughness: number, shadows: "reuse" | "disabled" = "disabled") => {
			material.roughness = roughness;
			if (backendKind === "webgl") {
				renderer.requestRender("unknown");
				await renderer.renderFrame(performance.now());
			}
			const ticket = target.enqueueJob({
				kind: "scene-view", camera,
				content: { environment: false, particles: false, shadows },
				readback: { attachmentIndex: 0 },
			});
			renderer.requestRender("unknown");
			await renderer.renderFrame(performance.now());
			const completion = await ticket.done;
			const pixel = completion.readback?.toRGBAFloat32();
			if (!pixel) throw new Error("Missing transmission readback.");
			if (gl) {
				const error = gl.getError();
				if (error !== gl.NO_ERROR) errors.push(`WebGL error: ${error}`);
			}
			return pixel[0];
		};
		const rows: { kind: string; clustered: boolean; aligned: number;
			offAxis: number; roughOffAxis: number; metal: number; noTransmission: number }[] = [];
		const kinds = backendKind === "webgl" ?
			["directional", "point", "spot"] as const :
			["directional", "point", "spot", "area"] as const;
		for (const kind of kinds) {
			for (const clustered of kind === "directional" ? [false] : [false, true]) {
				renderer.features.enableClusteredLighting = clustered;
				const light = kind === "directional" ? new DirectionalLight({ intensity: 0.01 })
					: kind === "point" ? new PointLight({ intensity: 0.04, range: 100 })
					: kind === "spot" ? new SpotLight({ intensity: 0.04, range: 100 })
					: new AreaLight({ intensity: 400, width: 0.01, height: 0.01, range: 100 });
				const setAngle = (angle: number) => {
					const x = Math.sin(angle);
					const z = Math.cos(angle);
					light.setPosition(2 * x, 0, -2 - 2 * z);
					if (light instanceof DirectionalLight || light instanceof SpotLight) {
						light.direction = { x: -x, y: 0, z };
					}
					if (light instanceof AreaLight) light.setRotationFromEuler(Math.PI / 2, 0, 0);
				};
				renderer.scene.add(light);
				setAngle(0);
				const aligned = await sample(0.08);
				setAngle(Math.PI / 4);
				const offAxis = await sample(0.08);
				const roughOffAxis = await sample(0.6);
				material.metalness = 1;
				const metal = await sample(0.6);
				material.metalness = 0;
				material.transmissionFactor = 0;
				const noTransmission = await sample(0.6);
				material.transmissionFactor = 1;
				rows.push({ kind, clustered, aligned, offAxis, roughOffAxis, metal, noTransmission });
				renderer.scene.remove(light);
			}
		}
		renderer.features.enableClusteredLighting = false;
		const sun = new DirectionalLight({ intensity: 0.01, direction: { x: 0, y: 0, z: 1 } });
		renderer.scene.add(sun);
		const iorRatios: number[] = [];
		for (const ior of [1.5, 1.1]) {
			material.ior = ior;
			sun.direction = { x: 0, y: 0, z: 1 };
			const aligned = await sample(0.6);
			sun.direction = { x: -Math.SQRT1_2, y: 0, z: Math.SQRT1_2 };
			iorRatios.push(await sample(0.6) / aligned);
		}
		material.ior = 1.5;
		sun.direction = { x: 0, y: 0, z: 1 };
		// Keep the volume/composition path fixed while varying only absorption.
		material.thicknessFactor = 1;
		material.attenuationDistance = 1;
		material.attenuationColor = { r: 255, g: 255, b: 255 };
		const bare = await sample(0.25);
		material.clearcoat = 1;
		const clearcoated = await sample(0.25);
		material.clearcoat = 0;
		material.attenuationColor = { r: 64, g: 64, b: 64 };
		const absorbed = await sample(0.25);
		material.thicknessFactor = 0;
		const blockerMaterial = new PBRMaterial({ metalness: 0, doubleSided: true });
		const blocker = MeshAsset.fromFaces([{
			material: blockerMaterial,
			vertices: [
				{ x: -2, y: -2, z: -3 }, { x: 2, y: -2, z: -3 },
				{ x: 2, y: 2, z: -3 }, { x: -2, y: 2, z: -3 },
			],
		}]);
		renderer.scene.add(new MeshInstance({ mesh: blocker }));
		// Zero bias isolates occlusion from this small fixture's perspective depth range.
		const shadowOptions = {
			size: 512, bias: { constant: 0, slope: 0, normal: 0, normalMin: 0, texel: 0 },
		};
		const shadow = renderer.scene.shadows.createSingle(shadowOptions);
		renderer.scene.shadows.bind(sun, shadow);
		await renderer.warmup();
		const unshadowed = await sample(0.25);
		const shadowed = await sample(0.25, "reuse");
		const shadowRows = [{ kind: "directional", clustered: false, unshadowed, shadowed }];
		renderer.scene.remove(sun);
		for (const clustered of [false, true]) {
			renderer.features.enableClusteredLighting = clustered;
			const spot = new SpotLight({
				intensity: 0.04, range: 100,
				position: { x: 0, y: 0, z: -4 }, direction: { x: 0, y: 0, z: 1 },
			});
			renderer.scene.add(spot);
			renderer.scene.shadows.bind(spot, renderer.scene.shadows.createSingle(shadowOptions));
			await renderer.warmup();
			shadowRows.push({ kind: "spot", clustered,
				unshadowed: await sample(0.25), shadowed: await sample(0.25, "reuse") });
			renderer.scene.remove(spot);
			renderer.scene.shadows.unbindLight(spot);
		}
		renderer.features.enableClusteredLighting = false;
		renderer.scene.add(sun);
		const viewAngle = 89 * Math.PI / 180;
		camera.setPosition(2 * Math.sin(viewAngle), 0, -2 + 2 * Math.cos(viewAngle));
		camera.setRotationFromEuler(0, viewAngle, 0);
		sun.direction = { x: Math.sin(viewAngle), y: 0, z: Math.cos(viewAngle) };
		const grazing = await sample(0.04);
		camera.setPosition(Math.sqrt(3), 0, -1);
		camera.setRotationFromEuler(0, Math.PI / 3, 0);
		sun.direction = { x: -Math.sqrt(3) / 2, y: 0, z: 0.5 };
		const rejectedMicrofacet = await sample(0.6);
		return { supported: true as const, errors, rows,
			attenuation: { bare, clearcoated, absorbed, unshadowed, shadowed },
			iorRatios, grazing, rejectedMicrofacet, shadowRows };
	});
}
