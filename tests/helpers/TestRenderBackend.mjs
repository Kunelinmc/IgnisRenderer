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
		};
		this.frameScheduling = "on-demand";
		this.extensions = createRenderBackendExtensionRegistry([]);
		this.sessionContext = null;
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

	createSession(context) {
		this.sessionContext = context;
		return this;
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

export function createBackendSession(provider, canvas = {}) {
	return provider.createSession({
		surface: { canvas },
		events: { emit: () => {} },
	});
}
