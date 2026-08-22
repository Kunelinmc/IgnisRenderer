import { BackendPostProcessRuntime } from "../../src/postprocess/BackendPostProcessRuntime.ts";
import { createRenderBackendExtensionRegistry } from "../../src/backends/BackendExtensions.ts";
import { WebGLFrameServices } from "../../src/backends/webgl/WebGLFrameServices.ts";
import { WebGLPostProcessExecutor } from "../../src/backends/webgl/WebGLPostProcessExecutor.ts";

const TEST_BACKEND = {
	profile: {
		id: "webgl",
		capabilities: {
			sh: true,
			shadows: true,
			reflection: false,
			environment: true,
			postProcess: true,
			clusteredLighting: true,
			oit: true,
			occlusionCulling: false,
			customRenderTargets: true,
			customRenderPasses: true,
			renderTargetReadback: true,
		},
		frameScheduling: "on-demand",
		lighting: { localizedProbeMode: "backend-local" },
	},
	extensions: createRenderBackendExtensionRegistry([]),
	attach() {},
	async initialize() {},
	getDebugInfo() {
		return {
			backend: "webgl",
			api: "webgl2",
			available: false,
			unavailableReason: "Static WebGL frame-service harness has no surface.",
		};
	},
	restore() {},
	resize() {},
	destroy() {},
	getAttachments(size) {
		return { width: size.width, height: size.height };
	},
	beginFrame() {},
	abortFrame() {},
	executePass() {},
	endFrame() {},
	getCompletedFrameCoverage() {
		return "full-frame";
	},
};

/** Static-test harness that explicitly owns and injects post-process runtime state. */
export class WebGLFrameServiceTestHarness extends WebGLFrameServices {
	constructor(gl, shaderRuntime, shaderCompileStage, options = {}) {
		let services = null;
		const postProcessRuntime = options.postProcessRuntime ??
			new BackendPostProcessRuntime({
				executor: new WebGLPostProcessExecutor({
					getDeviceServices: () => services,
				}),
				backend: TEST_BACKEND,
				warn: () => {},
			});
		super(gl, shaderRuntime, shaderCompileStage, {
			...options,
			postProcessRuntime,
		});
		services = this;
	}
}
