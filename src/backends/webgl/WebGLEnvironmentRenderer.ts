import { CameraType } from "../../cameras/Camera";
import type { FrameContext } from "../../pipeline/types";
import { ShaderSource } from "../../shaders/ShaderSource";

import type { WebGLFrameTargetManager } from "./WebGLFrameTargetManager";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
	WebGLProgramWarmupHandle,
} from "./WebGLProgramCompiler";
import type { WebGLTextureRegistry } from "./WebGLTextureRegistry";
import type {
	WebGLProgramWarmupContributor,
	WebGLProgramWarmupRequest,
	WebGLProgramWarmupTask,
} from "./WebGLWarmupCoordinator";

interface WebGLEnvironmentProgram {
	program: WebGLProgram;
	uniforms: {
		environmentMap: WebGLUniformLocation | null;
		environmentBasisRight: WebGLUniformLocation | null;
		environmentBasisUp: WebGLUniformLocation | null;
		environmentBasisBackward: WebGLUniformLocation | null;
		environmentIsOrthographic: WebGLUniformLocation | null;
		environmentMapIsLinear: WebGLUniformLocation | null;
		environmentBackgroundTint: WebGLUniformLocation | null;
		environmentBackgroundExposure: WebGLUniformLocation | null;
		environmentBackgroundStrength: WebGLUniformLocation | null;
	};
}

export interface WebGLEnvironmentRendererHost {
	readonly gl: WebGL2RenderingContext;
	readonly programCompiler: WebGLProgramCompiler;
	readonly targets: WebGLFrameTargetManager;
	readonly textures: WebGLTextureRegistry;
	getFullscreenVao(): WebGLVertexArrayObject | null;
	getWidth(): number;
	getHeight(): number;
}

/** Owns WebGL environment-background program state and fullscreen execution. */
export class WebGLEnvironmentRenderer implements WebGLProgramWarmupContributor {
	private readonly _host: WebGLEnvironmentRendererHost;
	private readonly _program: WebGLProgramSlot<WebGLEnvironmentProgram>;

	constructor(host: WebGLEnvironmentRendererHost) {
		this._host = host;
		this._program = host.programCompiler.createSlot({
			label: "WebGLEnvironmentProgram",
			vertex: () => ShaderSource.get("webgl.part.environmentVertex").source.code,
			fragment: () => ShaderSource.get("webgl.part.environmentFragment").source.code,
			reflect: (gl, program) => ({
				program,
				uniforms: {
					environmentMap: gl.getUniformLocation(program, "uEnvironmentMap"),
					environmentBasisRight: gl.getUniformLocation(program, "uEnvironmentBasisRight"),
					environmentBasisUp: gl.getUniformLocation(program, "uEnvironmentBasisUp"),
					environmentBasisBackward: gl.getUniformLocation(
						program,
						"uEnvironmentBasisBackward",
					),
					environmentIsOrthographic: gl.getUniformLocation(
						program,
						"uEnvironmentIsOrthographic",
					),
					environmentMapIsLinear: gl.getUniformLocation(
						program,
						"uEnvironmentMapIsLinear",
					),
					environmentBackgroundTint: gl.getUniformLocation(
						program,
						"uEnvironmentBackgroundTint",
					),
					environmentBackgroundExposure: gl.getUniformLocation(
						program,
						"uEnvironmentBackgroundExposure",
					),
					environmentBackgroundStrength: gl.getUniformLocation(
						program,
						"uEnvironmentBackgroundStrength",
					),
				},
			}),
		});
	}

	public warmupEnvironmentProgram(): WebGLProgramWarmupHandle {
		return this._program.warmup();
	}

	public collectWarmupTasks(
		request: WebGLProgramWarmupRequest,
	): readonly WebGLProgramWarmupTask[] {
		return request.plan.enableEnvironment
			? [
					{
						label: "WebGLEnvironmentProgram",
						priority: "optional",
						run: () => this.warmupEnvironmentProgram(),
					},
				]
			: [];
	}

	public render(context: FrameContext): boolean {
		const backgroundTexture = context.scene.environment.backgroundTexture;
		const vao = this._host.getFullscreenVao();
		if (!backgroundTexture || !vao) return false;
		const program = this._program.tryGet();
		if (!program) return false;

		const environment = context.scene.environment;
		const resolved = this._host.textures.getEnvironmentTexture(backgroundTexture);
		const view = context.viewCamera.viewMatrix.elements;
		const isOrthographic = context.viewCamera.type === CameraType.Orthographic;
		const tanHalfFov = isOrthographic ? 0 : Math.tan((context.viewCamera.fov * Math.PI) / 360);
		const aspect =
			context.viewCamera.aspectRatio || this._host.getWidth() / this._host.getHeight();
		const gl = this._host.gl;
		const uniforms = program.uniforms;

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._host.targets._sceneFramebuffer);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
		gl.useProgram(program.program);
		gl.bindVertexArray(vao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.BLEND);
		gl.disable(gl.DEPTH_TEST);
		gl.depthMask(false);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, resolved.texture);
		if (uniforms.environmentMap) gl.uniform1i(uniforms.environmentMap, 0);
		if (uniforms.environmentBasisRight) {
			gl.uniform4f(
				uniforms.environmentBasisRight,
				view[0][0],
				view[0][1],
				view[0][2],
				tanHalfFov,
			);
		}
		if (uniforms.environmentBasisUp) {
			gl.uniform4f(uniforms.environmentBasisUp, view[1][0], view[1][1], view[1][2], aspect);
		}
		if (uniforms.environmentBasisBackward) {
			gl.uniform3f(uniforms.environmentBasisBackward, view[2][0], view[2][1], view[2][2]);
		}
		if (uniforms.environmentIsOrthographic) {
			gl.uniform1f(uniforms.environmentIsOrthographic, isOrthographic ? 1 : 0);
		}
		if (uniforms.environmentMapIsLinear) {
			gl.uniform1i(uniforms.environmentMapIsLinear, resolved.isLinear ? 1 : 0);
		}
		if (uniforms.environmentBackgroundTint) {
			gl.uniform3f(
				uniforms.environmentBackgroundTint,
				environment.backgroundTintLinear.r,
				environment.backgroundTintLinear.g,
				environment.backgroundTintLinear.b,
			);
		}
		if (uniforms.environmentBackgroundExposure) {
			gl.uniform1f(
				uniforms.environmentBackgroundExposure,
				Math.max(1e-6, environment.backgroundExposure),
			);
		}
		if (uniforms.environmentBackgroundStrength) {
			gl.uniform1f(
				uniforms.environmentBackgroundStrength,
				Math.max(0, environment.backgroundStrength),
			);
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.depthMask(true);
		gl.enable(gl.DEPTH_TEST);
		gl.bindVertexArray(null);
		return true;
	}

	public destroy(): void {
		this._program.destroy();
	}
}
