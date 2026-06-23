import { createRenderBackendExtensionRegistry } from "../../src/renderers/BackendExtensions.ts";

export class TestRenderBackend {
	constructor() {
		this.id = "test";
		this.type = "test";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			postProcess: false,
			clusteredLighting: false,
			oit: false,
			occlusionCulling: false,
			customRenderTargets: false,
			customRenderPasses: false,
			renderTargetReadback: false,
		};
		this.frameScheduling = "on-demand";
		this.extensions = createRenderBackendExtensionRegistry([]);
		this.attachContext = null;
		this.attached = false;
	}

	get profile() {
		return {
			id: this.type,
			capabilities: this.capabilities,
			frameScheduling: this.frameScheduling,
			shadow: {
				backendKey: this.type,
				supportsFilterModes: ["pcf"],
				supportsDirectionalCSM: false,
				supportsSpotCSM: false,
				supportsPointCSM: false,
				maxDynamicShadowCost: 0,
			},
			lighting: { localizedProbeMode: "accumulate-globally" },
		};
	}

	attach(context) {
		if (this.attached) {
			throw new Error("TestRenderBackend is already attached to a renderer.");
		}
		this.attachContext = context;
		this.attached = true;
	}

	async initialize() {}

	async restore() {}

	resize() {}

	getAttachments({ width, height }) {
		return { width, height };
	}

	beginFrame() {}

	executePass() {}

	endFrame() {}

	abortFrame() {}

	destroy() {}
}

export function attachBackend(backend, canvas = {}) {
	backend.attach({
		surface: { canvas },
		events: { emit: () => {} },
	});
	return backend;
}
