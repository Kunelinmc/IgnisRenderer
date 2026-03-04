import type { DrawPacket } from "../../pipeline/types";
import type { Renderer } from "../../Renderer";
import { CpuProjector } from "../CpuProjector";

export class SoftwareMainPass {
	private _renderer: Renderer;

	constructor(renderer: Renderer) {
		this._renderer = renderer;
	}

	public render(packets: DrawPacket[], transparent: boolean): void {
		for (const packet of packets) {
			const faces = CpuProjector.projectPacket(packet, this._renderer);
			if (transparent) {
				faces.sort((left, right) => right.depthInfo.avg - left.depthInfo.avg);
			}

			for (const face of faces) {
				const projected = face.projected;
				for (let i = 1; i < projected.length - 1; i++) {
					this._renderer.rasterizer.drawTriangle(
						[projected[0], projected[i], projected[i + 1]],
						face,
						this._renderer.pixels,
						transparent
					);
				}
			}
		}
	}
}
