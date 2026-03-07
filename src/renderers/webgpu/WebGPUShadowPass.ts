import {
	LightType,
	isShadowCastingLight,
	type ShadowCastingLight,
} from "../../lights";
import { Matrix4 } from "../../maths/Matrix4";
import type {
	DrawPacket,
	FrameContext,
	PreparedScene,
} from "../../pipeline/types";
import {
	syncShadowMapRegistry,
	updateShadowMapMetadata,
} from "../../pipeline/ShadowMetadata";
import type { ShadowMap } from "../../utils/ShadowMapping";
import type { WebGPUBackend } from "../WebGPUBackend";
import {
	WEBGPU_MAX_DIRECTIONAL_LIGHTS,
	WEBGPU_MAX_SPOT_LIGHTS,
} from "./constants";
import type { WebGPUGeometryRegistry } from "./WebGPUGeometryRegistry";
import type { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";

const WEBGPU_SHADOW_DEPTH_SHADER = /* wgsl */ `
struct ShadowUniforms {
	mvp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> shadow: ShadowUniforms;

struct ShadowVertexInput {
	@location(0) position: vec3<f32>,
}

@vertex
fn vsMain(input: ShadowVertexInput) -> @builtin(position) vec4<f32> {
	return shadow.mvp * vec4<f32>(input.position, 1.0);
}
`;

interface ShadowRenderSlot {
	shadowMap: ShadowMap;
	tileX: number;
	tileY: number;
}

export class WebGPUShadowPass {
	private _backend: WebGPUBackend;
	private _geometryRegistry: WebGPUGeometryRegistry;
	private _shadowAtlases: WebGPUShadowAtlasAllocator;
	private _depthRemapMatrix = new Matrix4([
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 0.5, 0.5],
		[0, 0, 0, 1],
	]);
	private _shadowViewProjectionMatrix = Matrix4.identity();
	private _mvpMatrix = Matrix4.identity();
	private _uniformData = new Float32Array(16);
	private _shaderModule: GPUShaderModule | null = null;
	private _bindGroupLayout: GPUBindGroupLayout | null = null;
	private _pipelineLayout: GPUPipelineLayout | null = null;
	private _pipeline: GPURenderPipeline | null = null;
	private _drawUniformBuffers: GPUBuffer[] = [];
	private _drawBindGroups: GPUBindGroup[] = [];
	private _drawResourceCursor = 0;

	constructor(
		backend: WebGPUBackend,
		geometryRegistry: WebGPUGeometryRegistry,
		shadowAtlases: WebGPUShadowAtlasAllocator
	) {
		this._backend = backend;
		this._geometryRegistry = geometryRegistry;
		this._shadowAtlases = shadowAtlases;
	}

	public render(context: FrameContext): void {
		if (!context.features.enableShadows) return;

		const frame = context.scene;
		const shadowMaps = context.shadowMaps;
		const slots = this._collectShadowSlots(frame, shadowMaps);
		const maxShadowSize = getMaxShadowSize(slots);
		const atlasTileSize = Math.max(1, maxShadowSize);
		const atlasTexture =
			this._shadowAtlases.ensureAtlasForTileSize(atlasTileSize);
		const atlasView = (atlasTexture as { _gpuView?: GPUTextureView })._gpuView;
		if (!atlasView) return;

		this._ensurePipelineResources();
		if (!this._pipeline || !this._bindGroupLayout) return;

		this._drawResourceCursor = 0;

		const commandEncoder = this._backend.device.createCommandEncoder({
			label: "WebGPUShadowEncoder",
		});
		const passEncoder = commandEncoder.beginRenderPass({
			label: "WebGPUShadowPass",
			colorAttachments: [],
			depthStencilAttachment: {
				view: atlasView,
				depthClearValue: 1,
				depthLoadOp: "clear",
				depthStoreOp: "store",
			},
		});

		passEncoder.setPipeline(this._pipeline);

		for (const slot of slots) {
			const shadowMapSize = Math.max(1, slot.shadowMap.size | 0);
			const viewportX = slot.tileX * atlasTileSize;
			const viewportY = slot.tileY * atlasTileSize;
			passEncoder.setViewport(
				viewportX,
				viewportY,
				shadowMapSize,
				shadowMapSize,
				0,
				1
			);
			passEncoder.setScissorRect(
				viewportX,
				viewportY,
				shadowMapSize,
				shadowMapSize
			);
			Matrix4.multiply(
				this._depthRemapMatrix,
				slot.shadowMap.viewProjectionMatrix!,
				this._shadowViewProjectionMatrix
			);

			this._drawShadowCasters(
				passEncoder,
				frame.shadowCasterPackets,
				this._shadowViewProjectionMatrix
			);
		}

		passEncoder.end();
		this._backend.queue.submit([commandEncoder.finish()]);
		this._trimDrawResources();
	}

	private _drawShadowCasters(
		passEncoder: GPURenderPassEncoder,
		packets: DrawPacket[],
		viewProjectionMatrix: Matrix4
	): void {
		for (const packet of packets) {
			const geometry = this._geometryRegistry.getGeometry(packet.primitive);
			const vertexBuffer = (
				geometry.vertexBuffer as { _gpuResource?: GPUBuffer }
			)._gpuResource;
			const indexBuffer = (geometry.indexBuffer as { _gpuResource?: GPUBuffer })
				._gpuResource;
			if (!vertexBuffer || !indexBuffer) continue;

			Matrix4.multiply(
				viewProjectionMatrix,
				packet.worldMatrix,
				this._mvpMatrix
			);
			const bindGroup = this._nextDrawBindGroup();
			if (!bindGroup) continue;
			this._writeUniformMatrix(this._mvpMatrix, bindGroup.buffer);

			passEncoder.setVertexBuffer(0, vertexBuffer);
			passEncoder.setIndexBuffer(indexBuffer, "uint32");
			passEncoder.setBindGroup(0, bindGroup.group);
			passEncoder.drawIndexed(geometry.indexCount);
		}
	}

	private _writeUniformMatrix(matrix: Matrix4, buffer: GPUBuffer): void {
		const elements = matrix.elements;
		const data = this._uniformData;
		data[0] = elements[0][0];
		data[1] = elements[1][0];
		data[2] = elements[2][0];
		data[3] = elements[3][0];
		data[4] = elements[0][1];
		data[5] = elements[1][1];
		data[6] = elements[2][1];
		data[7] = elements[3][1];
		data[8] = elements[0][2];
		data[9] = elements[1][2];
		data[10] = elements[2][2];
		data[11] = elements[3][2];
		data[12] = elements[0][3];
		data[13] = elements[1][3];
		data[14] = elements[2][3];
		data[15] = elements[3][3];
		this._backend.queue.writeBuffer(buffer, 0, data);
	}

	private _collectShadowSlots(
		scene: PreparedScene,
		shadowMaps: Map<ShadowCastingLight, ShadowMap>
	): ShadowRenderSlot[] {
		const slots: ShadowRenderSlot[] = [];
		let directionalIndex = 0;
		let spotIndex = 0;

		for (const light of scene.lights) {
			if (light.type === LightType.Directional) {
				if (directionalIndex >= WEBGPU_MAX_DIRECTIONAL_LIGHTS) continue;
				if (isShadowCastingLight(light)) {
					const shadowMap = shadowMaps.get(light);
					if (shadowMap?.viewProjectionMatrix) {
						slots.push({
							shadowMap,
							tileX: directionalIndex,
							tileY: 0,
						});
					}
				}
				directionalIndex++;
				continue;
			}

			if (light.type === LightType.Spot) {
				if (spotIndex >= WEBGPU_MAX_SPOT_LIGHTS) continue;
				if (isShadowCastingLight(light)) {
					const shadowMap = shadowMaps.get(light);
					if (shadowMap?.viewProjectionMatrix) {
						slots.push({
							shadowMap,
							tileX: spotIndex,
							tileY: 1,
						});
					}
				}
				spotIndex++;
			}
		}

		return slots;
	}

	private _ensurePipelineResources(): void {
		if (this._pipeline && this._bindGroupLayout) return;

		const device = this._backend.device;
		if (!this._shaderModule) {
			this._shaderModule = device.createShaderModule({
				label: "WebGPUShadowDepthShader",
				code: WEBGPU_SHADOW_DEPTH_SHADER,
			});
		}

		if (!this._bindGroupLayout) {
			this._bindGroupLayout = device.createBindGroupLayout({
				label: "WebGPUShadowDepthBindGroupLayout",
				entries: [
					{
						binding: 0,
						visibility: GPUShaderStage.VERTEX,
						buffer: { type: "uniform" },
					},
				],
			});
		}

		if (!this._pipelineLayout && this._bindGroupLayout) {
			this._pipelineLayout = device.createPipelineLayout({
				label: "WebGPUShadowDepthPipelineLayout",
				bindGroupLayouts: [this._bindGroupLayout],
			});
		}

		if (!this._pipeline && this._shaderModule && this._pipelineLayout) {
			this._pipeline = device.createRenderPipeline({
				label: "WebGPUShadowDepthPipeline",
				layout: this._pipelineLayout,
				vertex: {
					module: this._shaderModule,
					entryPoint: "vsMain",
					buffers: [
						{
							arrayStride: 56,
							attributes: [
								{
									shaderLocation: 0,
									offset: 0,
									format: "float32x3",
								},
							],
						},
					],
				},
				primitive: {
					topology: "triangle-list",
					cullMode: "none",
					frontFace: "ccw",
				},
				depthStencil: {
					format: "depth32float",
					depthWriteEnabled: true,
					depthCompare: "less",
				},
			});
		}
	}

	private _nextDrawBindGroup(): {
		buffer: GPUBuffer;
		group: GPUBindGroup;
	} | null {
		if (!this._bindGroupLayout) return null;

		const slot = this._drawResourceCursor++;
		let buffer = this._drawUniformBuffers[slot];
		let group = this._drawBindGroups[slot];

		if (!buffer) {
			buffer = this._backend.device.createBuffer({
				label: `WebGPUShadowDepthUniforms_${slot}`,
				size: 16 * 4,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			});
			this._drawUniformBuffers[slot] = buffer;
		}

		if (!group) {
			group = this._backend.device.createBindGroup({
				label: `WebGPUShadowDepthBindGroup_${slot}`,
				layout: this._bindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: { buffer },
					},
				],
			});
			this._drawBindGroups[slot] = group;
		}

		return { buffer, group };
	}

	private _trimDrawResources(): void {
		const used = this._drawResourceCursor;
		const allocated = this._drawUniformBuffers.length;
		// Trim when usage drops below 1/3 of allocated capacity and there
		// are at least 16 excess slots, to avoid trimming on small
		// fluctuations.
		if (allocated > 16 && used < allocated / 3) {
			const keep = Math.max(used, 8);
			for (let i = keep; i < allocated; i++) {
				this._drawUniformBuffers[i]?.destroy();
			}
			this._drawUniformBuffers.length = keep;
			this._drawBindGroups.length = keep;
		}
	}
}

function getMaxShadowSize(slots: ShadowRenderSlot[]): number {
	let maxSize = 0;
	for (const slot of slots) {
		maxSize = Math.max(maxSize, slot.shadowMap.size | 0);
	}
	return maxSize;
}
