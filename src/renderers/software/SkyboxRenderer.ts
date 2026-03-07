import type { Camera } from "../../cameras/Camera";
import { CameraType } from "../../cameras/Camera";
import type { Texture } from "../../core/Texture";

export class SkyboxRenderer {
	public static render(
		skybox: Texture,
		pixels: Uint8ClampedArray,
		camera: Camera,
		width: number,
		height: number
	): void {
		const view = camera.viewMatrix.elements;
		const right = { x: view[0][0], y: view[0][1], z: view[0][2] };
		const up = { x: view[1][0], y: view[1][1], z: view[1][2] };
		const backward = { x: view[2][0], y: view[2][1], z: view[2][2] };
		const isOrthographic = camera.type === CameraType.Orthographic;
		const fovRad = (camera.fov * Math.PI) / 180;
		const tanHalfFov = isOrthographic ? 0 : Math.tan(fovRad * 0.5);
		const aspect = camera.aspectRatio || width / height;

		for (let y = 0; y < height; y++) {
			const ndcY = 1 - ((y + 0.5) / height) * 2;
			const cy = ndcY * tanHalfFov;
			const rowBase = y * width * 4;

			for (let x = 0; x < width; x++) {
				const ndcX = ((x + 0.5) / width) * 2 - 1;
				const cx = ndcX * aspect * tanHalfFov;
				const dirX = right.x * cx + up.x * cy - backward.x;
				const dirY = right.y * cx + up.y * cy - backward.y;
				const dirZ = right.z * cx + up.z * cy - backward.z;
				const invLen = 1 / Math.sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
				const dx = dirX * invLen;
				const dy = dirY * invLen;
				const dz = dirZ * invLen;
				const phi = Math.atan2(dx, dz);
				const theta = Math.acos(Math.max(-1, Math.min(1, dy)));
				const u = (phi + Math.PI) / (2 * Math.PI);
				const v = theta / Math.PI;
				const color = skybox.sample(u, v);
				const idx = rowBase + x * 4;
				pixels[idx] = color.r;
				pixels[idx + 1] = color.g;
				pixels[idx + 2] = color.b;
				pixels[idx + 3] = 255;
			}
		}
	}
}
