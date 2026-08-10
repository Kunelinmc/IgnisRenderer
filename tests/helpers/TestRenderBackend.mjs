import { createRenderBackendExtensionRegistry } from "../../src/backends/BackendExtensions.ts";
import {
	DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	createSDRDisplayOutputState,
	resolveDisplayOutputOptions,
} from "../../src/rendering/DisplayOutput.ts";

export class TestRenderBackend {
	constructor() {
		this.type = "test";
		this.capabilities = {
			displayHDR: false,
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
		this.displayOutputState = createSDRDisplayOutputState(
			DEFAULT_DISPLAY_OUTPUT_OPTIONS,
		);
	}

	get profile() {
		return {
			id: this.type,
			capabilities: this.capabilities,
			frameScheduling: this.frameScheduling,
			lighting: { localizedProbeMode: "accumulate-globally" },
		};
	}

	attach(context) {
		if (this.attached) {
			throw new Error("TestRenderBackend is already attached to a renderer.");
		}
		this.attachContext = context;
		const requested = resolveDisplayOutputOptions(
			context.surface.displayOutput,
		);
		this.displayOutputState = createSDRDisplayOutputState(
			requested,
			requested.mode === "hdr" ? "backend-unsupported" : undefined,
		);
		this.attached = true;
	}

	async initialize() {}

	getDebugInfo() {
		return {
			backend: this.type,
			api: "software",
			available: false,
			unavailableReason: "Test backend has not been initialized.",
		};
	}

	getDisplayOutputState() {
		return this.displayOutputState;
	}

	async setDisplayOutput(options) {
		const previous = this.displayOutputState;
		const requested = resolveDisplayOutputOptions(
			options,
			previous.requested,
		);
		const current = createSDRDisplayOutputState(
			requested,
			requested.mode === "hdr" ? "backend-unsupported" : undefined,
		);
		this.displayOutputState = current;
		this.attachContext.events.emit({
			type: "display-output-change",
			previous,
			current,
		});
		return current;
	}

	async restore() {}

	resize() {}

	getAttachments({ width, height }) {
		return { width, height };
	}

	beginFrame() {}

	executePass() {}

	endFrame() {}

	getCompletedFrameCoverage() {
		return "full-frame";
	}

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
