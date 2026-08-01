import type {
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../../shaders/runtime";

import {
	WebGLFrameServiceOwner,
	type WebGLFrameServiceOwnerOptions,
} from "./WebGLFrameServiceOwner";
import { WebGLIBLPrefilterRuntime } from "./WebGLIBLPrefilterRuntime";

/** Owns all services tied to one WebGL context generation. */
export class WebGLContextServiceOwner {
	public readonly frame: WebGLFrameServiceOwner;
	public readonly iblPrefilter: WebGLIBLPrefilterRuntime;

	public constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime: ShaderRuntime,
		shaderCompileStage: ShaderBackendCompileStage,
		options: WebGLFrameServiceOwnerOptions,
	) {
		this.frame = new WebGLFrameServiceOwner(
			gl,
			shaderRuntime,
			shaderCompileStage,
			options,
		);
		try {
			this.iblPrefilter = new WebGLIBLPrefilterRuntime({
				gl,
				programs: this.frame._programs,
				getFullscreenVao: () => this.frame._fullscreenVao,
			});
		} catch (error) {
			this.frame.destroy();
			throw error;
		}
	}

	/** Restores the active or idle framebuffer baseline after auxiliary work. */
	public restoreContextWorkBaseline(): void {
		this.frame.restoreContextWorkBaseline();
	}

	public destroy(): void {
		this.iblPrefilter.destroy();
		this.frame.destroy();
	}
}
