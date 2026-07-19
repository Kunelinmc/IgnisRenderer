import type { Camera } from "../../cameras/Camera";
import { CameraType } from "../../cameras/Camera";
import type { Texture } from "../../core/Texture";
import { sRGBToLinear } from "../../maths/Common";

export interface SkyboxRenderOptions {
	strength: number;
	tintLinear: {
		r: number;
		g: number;
		b: number;
	};
	exposure: number;
}

export class SkyboxRenderer {
	public static render(
		environmentBackgroundTexture: Texture,
		options: SkyboxRenderOptions,
		pixels: Uint8ClampedArray,
		camera: Camera,
		width: number,
		height: number
	): void {
		const decodeSRGB = environmentBackgroundTexture.colorSpace === "sRGB";
		const strength = Math.max(0, options.strength);
		const exposure = Math.max(1e-6, options.exposure);
		const tint = options.tintLinear;
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
				const color = environmentBackgroundTexture.sample(u, v);
				const idx = rowBase + x * 4;
				const linearR = decodeSRGB ? sRGBToLinear(color.r / 255) : color.r / 255;
				const linearG = decodeSRGB ? sRGBToLinear(color.g / 255) : color.g / 255;
				const linearB = decodeSRGB ? sRGBToLinear(color.b / 255) : color.b / 255;
				const scaledR = linearR * tint.r * exposure * strength;
				const scaledG = linearG * tint.g * exposure * strength;
				const scaledB = linearB * tint.b * exposure * strength;
				pixels[idx] = Math.max(0, Math.min(255, Math.round(scaledR * 255)));
				pixels[idx + 1] = Math.max(0, Math.min(255, Math.round(scaledG * 255)));
				pixels[idx + 2] = Math.max(0, Math.min(255, Math.round(scaledB * 255)));
				pixels[idx + 3] = 255;
			}
		}
	}
}
