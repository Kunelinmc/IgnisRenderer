import type {
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../../shaders/runtime";

import {
	WebGLFrameServices,
	type WebGLFrameServicesOptions,
} from "./WebGLFrameServices";
import { WebGLAuxiliaryRasterRuntime } from "./WebGLAuxiliaryRasterRuntime";

/** Owns all services tied to one WebGL context generation. */
export class WebGLContextServiceOwner {
	public readonly frame: WebGLFrameServices;
	public readonly auxiliaryRaster: WebGLAuxiliaryRasterRuntime;

	public constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime: ShaderRuntime,
		shaderCompileStage: ShaderBackendCompileStage,
		options: WebGLFrameServicesOptions,
	) {
		this.frame = new WebGLFrameServices(
			gl,
			shaderRuntime,
			shaderCompileStage,
			options,
		);
		try {
			this.auxiliaryRaster = new WebGLAuxiliaryRasterRuntime(
				gl,
				shaderRuntime,
				shaderCompileStage,
			);
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
		this.frame.destroy();
		this.auxiliaryRaster.destroy();
	}
}
