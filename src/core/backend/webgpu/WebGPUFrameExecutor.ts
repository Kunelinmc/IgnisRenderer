import type {
	DrawPacket,
	FramePass,
	PreparedScene,
} from "../../pipeline/types";
import type { RenderResources } from "../../resources/RenderResources";
import type { WebGPUBackend } from "../WebGPUBackend";

export class WebGPUFrameExecutor {
	private _backend: WebGPUBackend;
	private _resources: RenderResources;
	private _encoder: any = null;

	constructor(backend: WebGPUBackend, resources: RenderResources) {
		this._backend = backend;
		this._resources = resources;
	}

	public beginFrame(): void {
		this._encoder = this._backend.createCommandEncoder();
	}

	public async executePass(
		pass: FramePass,
		frame: PreparedScene
	): Promise<void> {
		if (!this._encoder) return;

		if (pass.stage === "main-opaque") {
			await this._recordMainPass(frame.opaquePackets, true);
			return;
		}

		if (pass.stage === "main-transparent") {
			await this._recordMainPass(frame.transparentPackets, false);
		}
	}

	public endFrame(): void {
		if (!this._encoder) return;
		this._backend.submit([this._encoder.finish()]);
		this._encoder = null;
	}

	private async _recordMainPass(
		packets: DrawPacket[],
		clearAttachments: boolean
	): Promise<void> {
		const colorTexture = this._backend.getCanvasColorTexture();
		const depthTexture = this._backend.getCanvasDepthTexture();

		this._encoder.beginRenderPass({
			colorAttachments: [
				{
					view: colorTexture,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: clearAttachments ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthTexture,
				depthClearValue: 1,
				depthLoadOp: clearAttachments ? "clear" : "load",
				depthStoreOp: "store",
			},
		});

		for (const packet of packets) {
			const resources = await this._resources.getDrawResources(packet);
			if (!resources) continue;

			this._encoder.setPipeline(resources.pipeline);
			this._encoder.setBindingGroup(0, resources.frameBinding);
			this._encoder.setBindingGroup(1, resources.modelBinding);
			this._encoder.setVertexBuffer(0, resources.vertexBuffer);
			this._encoder.setIndexBuffer(resources.indexBuffer, "uint32");
			this._encoder.drawIndexed(resources.indexCount);
		}

		this._encoder.endRenderPass();
	}
}
