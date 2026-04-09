import { clamp } from "../../maths/Common";
import { ParticleBlendMode } from "../../particles";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type FogOptions,
	type FrameContext,
	type ParticleRenderBatch,
} from "../../pipeline/types";
import { finiteOr, toColumnMajorMat4 } from "./WebGLFrameMath";
import { resolveTextureUVTransform } from "./WebGLMaterialUniformResolver";
import { Logger } from "../../foundation/Logger";

const PARTICLE_QUAD_VERTICES = new Float32Array([
	-0.5,
	-0.5,
	0,
	1,
	0.5,
	-0.5,
	1,
	1,
	0.5,
	0.5,
	1,
	0,
	-0.5,
	-0.5,
	0,
	1,
	0.5,
	0.5,
	1,
	0,
	-0.5,
	0.5,
	0,
	0,
]);

const PARTICLE_QUAD_STRIDE = 16;
const PARTICLE_INSTANCE_FLOATS = 13;
const PARTICLE_INSTANCE_STRIDE = PARTICLE_INSTANCE_FLOATS * 4;
const PARTICLE_INITIAL_CAPACITY = 256;
const PARTICLE_MAX_INSTANCES_PER_DRAW = 1 << 16;

function logWebGLParticlePassWarning(key: string, message: string): void {
	Logger.warn(`[${key}] ${message}`, {
		scope: "WebGLParticlePass",
		onceKey: key,
	});
}

export interface WebGLParticlePassHost {
	_gl: WebGL2RenderingContext;
	_programs: {
		getParticleProgram(): {
			program: WebGLProgram;
			uniforms: {
				viewProjection?: WebGLUniformLocation | string | null;
				basisRight?: WebGLUniformLocation | string | null;
				basisUp?: WebGLUniformLocation | string | null;
				cameraPosition?: WebGLUniformLocation | string | null;
				fogParams0?: WebGLUniformLocation | string | null;
				fogParams1?: WebGLUniformLocation | string | null;
				particleMap?: WebGLUniformLocation | string | null;
				mapIsLinear?: WebGLUniformLocation | string | null;
				uvTransformA?: WebGLUniformLocation | string | null;
				uvTransformB?: WebGLUniformLocation | string | null;
			};
		};
	};
	_textures: {
		getBaseColorTexture(texture: any | null): {
			texture: WebGLTexture | null;
			isLinear: boolean;
		};
	};
	_sceneFramebuffer: WebGLFramebuffer | null;
	_particleVao: WebGLVertexArrayObject | null;
	_particleQuadBuffer: WebGLBuffer | null;
	_particleInstanceBuffer: WebGLBuffer | null;
	_particleInstanceCapacity: number;
	_particleScratch: Float32Array;
	_width: number;
	_height: number;
	_fogParams0: Float32Array;
	_fogParams1: Float32Array;
	_isIncrementalPartial(context: FrameContext | null | undefined): boolean;
	_resolveDirtyRects(
		context: FrameContext | null | undefined,
		viewportWidth: number,
		viewportHeight: number
	): Array<{ x: number; y: number; width: number; height: number }>;
	_setScissorRect(
		x: number,
		y: number,
		width: number,
		height: number,
		viewportHeight: number
	): void;
	_updateFogParams(options: FogOptions | undefined, enabled: boolean): void;
	_ensureParticleResources(): void;
	_ensureParticleCapacity(requiredInstances: number): void;
	_writeParticleInstances(batch: ParticleRenderBatch): number;
	_bindParticleInstanceAttributes(): void;
}

export function renderWebGLParticles(
	host: WebGLParticlePassHost,
	context: FrameContext
): void {
	if (!host._sceneFramebuffer) {
		return;
	}

	const batches = context.transient.get(
		PARTICLE_TRANSIENT_BATCHES_KEY
	) as ParticleRenderBatch[] | undefined;
	if (!Array.isArray(batches) || batches.length === 0) {
		return;
	}

	host._ensureParticleResources();
	if (
		!host._particleVao ||
		!host._particleQuadBuffer ||
		!host._particleInstanceBuffer
	) {
		return;
	}

	const gl = host._gl;
	const particleProgram = host._programs.getParticleProgram();
	const view = context.camera.viewMatrix.elements;
	const incrementalPartial = host._isIncrementalPartial(context);
	const dirtyRects = host._resolveDirtyRects(context, host._width, host._height);
	if (incrementalPartial && dirtyRects.length === 0) {
		return;
	}

	gl.bindFramebuffer(gl.FRAMEBUFFER, host._sceneFramebuffer);
	gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
	gl.useProgram(particleProgram.program);
	gl.bindVertexArray(host._particleVao);
	gl.enable(gl.DEPTH_TEST);
	gl.depthMask(false);
	gl.disable(gl.CULL_FACE);
	gl.enable(gl.BLEND);
	if (incrementalPartial) {
		gl.enable(gl.SCISSOR_TEST);
	}

	if (particleProgram.uniforms.viewProjection) {
		gl.uniformMatrix4fv(
			particleProgram.uniforms.viewProjection as WebGLUniformLocation,
			false,
			toColumnMajorMat4(context.camera.viewProjectionMatrix)
		);
	}
	if (particleProgram.uniforms.basisRight) {
		gl.uniform3f(
			particleProgram.uniforms.basisRight as WebGLUniformLocation,
			view[0][0],
			view[0][1],
			view[0][2]
		);
	}
	if (particleProgram.uniforms.basisUp) {
		gl.uniform3f(
			particleProgram.uniforms.basisUp as WebGLUniformLocation,
			view[1][0],
			view[1][1],
			view[1][2]
		);
	}
	if (particleProgram.uniforms.cameraPosition) {
		const cameraPosition = context.camera.getWorldPosition();
		gl.uniform3f(
			particleProgram.uniforms.cameraPosition as WebGLUniformLocation,
			finiteOr(cameraPosition.x, 0),
			finiteOr(cameraPosition.y, 0),
			finiteOr(cameraPosition.z, 0)
		);
	}
	const sceneFogEnabled =
		context.features.enableFog &&
		(context.features.fogOptions?.application ?? "postprocess") === "scene";
	host._updateFogParams(context.features.fogOptions, sceneFogEnabled);
	if (particleProgram.uniforms.fogParams0) {
		gl.uniform4fv(
			particleProgram.uniforms.fogParams0 as WebGLUniformLocation,
			host._fogParams0
		);
	}
	if (particleProgram.uniforms.fogParams1) {
		gl.uniform4fv(
			particleProgram.uniforms.fogParams1 as WebGLUniformLocation,
			host._fogParams1
		);
	}
	if (particleProgram.uniforms.particleMap) {
		gl.uniform1i(
			particleProgram.uniforms.particleMap as WebGLUniformLocation,
			0
		);
	}

	for (const batch of batches) {
		const preflightCount = Math.min(
			PARTICLE_MAX_INSTANCES_PER_DRAW,
			batch?.particles?.length ?? 0
		);
		if (preflightCount <= 0) {
			continue;
		}

		host._ensureParticleCapacity(preflightCount);
		const instanceCount = host._writeParticleInstances(batch);
		if (instanceCount <= 0 || !host._particleInstanceBuffer) {
			continue;
		}
		gl.bindBuffer(gl.ARRAY_BUFFER, host._particleInstanceBuffer);
		gl.bufferSubData(
			gl.ARRAY_BUFFER,
			0,
			host._particleScratch.subarray(0, instanceCount * PARTICLE_INSTANCE_FLOATS)
		);

		const resolvedTexture = host._textures.getBaseColorTexture(batch.texture ?? null);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, resolvedTexture.texture);
		if (particleProgram.uniforms.mapIsLinear) {
			gl.uniform1i(
				particleProgram.uniforms.mapIsLinear as WebGLUniformLocation,
				resolvedTexture.isLinear ? 1 : 0
			);
		}

		const uvTransform = resolveTextureUVTransform(batch.texture);
		if (particleProgram.uniforms.uvTransformA) {
			gl.uniform4f(
				particleProgram.uniforms.uvTransformA as WebGLUniformLocation,
				uvTransform.repeatX,
				uvTransform.repeatY,
				uvTransform.offsetX,
				uvTransform.offsetY
			);
		}
		if (particleProgram.uniforms.uvTransformB) {
			gl.uniform2f(
				particleProgram.uniforms.uvTransformB as WebGLUniformLocation,
				uvTransform.cosRotation,
				uvTransform.sinRotation
			);
		}

		if (batch.blendMode === ParticleBlendMode.Additive) {
			gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE, gl.ONE, gl.ONE);
		} else {
			gl.blendFuncSeparate(
				gl.SRC_ALPHA,
				gl.ONE_MINUS_SRC_ALPHA,
				gl.ONE,
				gl.ONE_MINUS_SRC_ALPHA
			);
		}

		if (incrementalPartial) {
			for (const rect of dirtyRects) {
				host._setScissorRect(
					rect.x,
					rect.y,
					rect.width,
					rect.height,
					host._height
				);
				gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, instanceCount);
			}
		} else {
			gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, instanceCount);
		}
	}

	gl.blendFuncSeparate(
		gl.SRC_ALPHA,
		gl.ONE_MINUS_SRC_ALPHA,
		gl.ONE,
		gl.ONE_MINUS_SRC_ALPHA
	);
	gl.depthMask(true);
	gl.disable(gl.BLEND);
	if (incrementalPartial) {
		gl.disable(gl.SCISSOR_TEST);
	}
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	gl.bindVertexArray(null);
}

export function writeWebGLParticleInstances(
	host: WebGLParticlePassHost,
	batch: ParticleRenderBatch
): number {
	const particles = batch.particles;
	if (!Array.isArray(particles) || particles.length === 0) {
		return 0;
	}

	let cappedCount = particles.length;
	if (cappedCount > PARTICLE_MAX_INSTANCES_PER_DRAW) {
		logWebGLParticlePassWarning(
			"webgl-particle-cap",
			`WebGL particle pass truncates system "${batch.systemId}" to ${PARTICLE_MAX_INSTANCES_PER_DRAW} instances per draw`
		);
		cappedCount = PARTICLE_MAX_INSTANCES_PER_DRAW;
	}

	if (host._particleScratch.length < cappedCount * PARTICLE_INSTANCE_FLOATS) {
		host._particleScratch = new Float32Array(cappedCount * PARTICLE_INSTANCE_FLOATS);
	}

	let writeCount = 0;
	for (let i = 0; i < cappedCount; i++) {
		const particle = particles[i];
		if (!particle) {
			continue;
		}

		const x = particle.position?.x;
		const y = particle.position?.y;
		const z = particle.position?.z;
		const size = particle.size;
		const rotation = particle.rotation;
		if (
			!Number.isFinite(x) ||
			!Number.isFinite(y) ||
			!Number.isFinite(z) ||
			!Number.isFinite(size) ||
			!Number.isFinite(rotation)
		) {
			continue;
		}

		const safeSize = Math.max(0, size);
		if (safeSize <= 0) {
			continue;
		}

		const color = particle.color;
		if (!color) {
			continue;
		}
		const alpha = clamp(Number.isFinite(color.a) ? color.a : 0, 0, 1);
		if (alpha <= 0) {
			continue;
		}
		const red = clamp((Number.isFinite(color.r) ? color.r : 0) / 255, 0, 1);
		const green = clamp((Number.isFinite(color.g) ? color.g : 0) / 255, 0, 1);
		const blue = clamp((Number.isFinite(color.b) ? color.b : 0) / 255, 0, 1);

		const uvRect = particle.uvRect;
		const u0 = Number.isFinite(uvRect?.u0) ? uvRect.u0 : 0;
		const v0 = Number.isFinite(uvRect?.v0) ? uvRect.v0 : 0;
		const u1 = Number.isFinite(uvRect?.u1) ? uvRect.u1 : 1;
		const v1 = Number.isFinite(uvRect?.v1) ? uvRect.v1 : 1;

		const offset = writeCount * PARTICLE_INSTANCE_FLOATS;
		host._particleScratch[offset] = x;
		host._particleScratch[offset + 1] = y;
		host._particleScratch[offset + 2] = z;
		host._particleScratch[offset + 3] = safeSize;
		host._particleScratch[offset + 4] = red;
		host._particleScratch[offset + 5] = green;
		host._particleScratch[offset + 6] = blue;
		host._particleScratch[offset + 7] = alpha;
		host._particleScratch[offset + 8] = u0;
		host._particleScratch[offset + 9] = v0;
		host._particleScratch[offset + 10] = u1;
		host._particleScratch[offset + 11] = v1;
		host._particleScratch[offset + 12] = rotation;
		writeCount++;
	}

	return writeCount;
}

export function ensureWebGLParticleResources(host: WebGLParticlePassHost): void {
	if (host._particleVao && host._particleQuadBuffer && host._particleInstanceBuffer) {
		return;
	}

	const gl = host._gl;
	const vao = gl.createVertexArray();
	const quadBuffer = gl.createBuffer();
	const instanceBuffer = gl.createBuffer();
	if (!vao || !quadBuffer || !instanceBuffer) {
		if (vao) {
			gl.deleteVertexArray(vao);
		}
		if (quadBuffer) {
			gl.deleteBuffer(quadBuffer);
		}
		if (instanceBuffer) {
			gl.deleteBuffer(instanceBuffer);
		}
		logWebGLParticlePassWarning(
			"webgl-particle-buffer-allocation",
			"Failed to allocate WebGL particle buffers; particle rendering is disabled for this frame"
		);
		return;
	}

	host._particleVao = vao;
	host._particleQuadBuffer = quadBuffer;
	host._particleInstanceBuffer = instanceBuffer;
	host._particleInstanceCapacity = PARTICLE_INITIAL_CAPACITY;
	host._particleScratch = new Float32Array(
		PARTICLE_INITIAL_CAPACITY * PARTICLE_INSTANCE_FLOATS
	);

	gl.bindVertexArray(vao);

	gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
	gl.bufferData(gl.ARRAY_BUFFER, PARTICLE_QUAD_VERTICES, gl.STATIC_DRAW);
	gl.enableVertexAttribArray(0);
	gl.vertexAttribPointer(0, 2, gl.FLOAT, false, PARTICLE_QUAD_STRIDE, 0);
	gl.enableVertexAttribArray(1);
	gl.vertexAttribPointer(1, 2, gl.FLOAT, false, PARTICLE_QUAD_STRIDE, 8);

	gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
	gl.bufferData(
		gl.ARRAY_BUFFER,
		host._particleInstanceCapacity * PARTICLE_INSTANCE_STRIDE,
		gl.DYNAMIC_DRAW
	);
	host._bindParticleInstanceAttributes();

	gl.bindVertexArray(null);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
}

export function ensureWebGLParticleCapacity(
	host: WebGLParticlePassHost,
	requiredInstances: number
): void {
	if (!host._particleInstanceBuffer || !host._particleVao) {
		return;
	}
	if (requiredInstances <= host._particleInstanceCapacity) {
		return;
	}

	const nextCapacity = Math.max(
		host._particleInstanceCapacity,
		1 << Math.ceil(Math.log2(Math.max(1, requiredInstances)))
	);
	const gl = host._gl;
	const newBuffer = gl.createBuffer();
	if (!newBuffer) {
		logWebGLParticlePassWarning(
			"webgl-particle-buffer-grow",
			`Failed to grow WebGL particle instance buffer to ${nextCapacity}; keeping previous capacity`
		);
		return;
	}

	gl.deleteBuffer(host._particleInstanceBuffer);
	host._particleInstanceBuffer = newBuffer;
	host._particleInstanceCapacity = nextCapacity;
	if (host._particleScratch.length < nextCapacity * PARTICLE_INSTANCE_FLOATS) {
		host._particleScratch = new Float32Array(nextCapacity * PARTICLE_INSTANCE_FLOATS);
	}

	gl.bindVertexArray(host._particleVao);
	gl.bindBuffer(gl.ARRAY_BUFFER, newBuffer);
	gl.bufferData(
		gl.ARRAY_BUFFER,
		nextCapacity * PARTICLE_INSTANCE_STRIDE,
		gl.DYNAMIC_DRAW
	);
	host._bindParticleInstanceAttributes();
	gl.bindVertexArray(null);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
}

export function bindWebGLParticleInstanceAttributes(
	host: WebGLParticlePassHost
): void {
	const gl = host._gl;
	if (!host._particleInstanceBuffer) {
		return;
	}
	gl.bindBuffer(gl.ARRAY_BUFFER, host._particleInstanceBuffer);

	gl.enableVertexAttribArray(2);
	gl.vertexAttribPointer(2, 4, gl.FLOAT, false, PARTICLE_INSTANCE_STRIDE, 0);
	gl.vertexAttribDivisor(2, 1);

	gl.enableVertexAttribArray(3);
	gl.vertexAttribPointer(3, 4, gl.FLOAT, false, PARTICLE_INSTANCE_STRIDE, 16);
	gl.vertexAttribDivisor(3, 1);

	gl.enableVertexAttribArray(4);
	gl.vertexAttribPointer(4, 4, gl.FLOAT, false, PARTICLE_INSTANCE_STRIDE, 32);
	gl.vertexAttribDivisor(4, 1);

	gl.enableVertexAttribArray(5);
	gl.vertexAttribPointer(5, 1, gl.FLOAT, false, PARTICLE_INSTANCE_STRIDE, 48);
	gl.vertexAttribDivisor(5, 1);
}

export function destroyWebGLParticleResources(host: WebGLParticlePassHost): void {
	const gl = host._gl;
	if (host._particleVao) {
		gl.deleteVertexArray(host._particleVao);
		host._particleVao = null;
	}
	if (host._particleQuadBuffer) {
		gl.deleteBuffer(host._particleQuadBuffer);
		host._particleQuadBuffer = null;
	}
	if (host._particleInstanceBuffer) {
		gl.deleteBuffer(host._particleInstanceBuffer);
		host._particleInstanceBuffer = null;
	}
	host._particleInstanceCapacity = 0;
	host._particleScratch = new Float32Array(0);
}

