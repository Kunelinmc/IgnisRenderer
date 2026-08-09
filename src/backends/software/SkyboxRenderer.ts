import { CameraType } from "../../cameras/Camera";
import type { Texture } from "../../core/Texture";
import type { Matrix4 } from "../../maths/Matrix4";
import { sampleEnvironmentTextureLevelLinear } from "../../lights/runtime/environmentMapRuntime";

export interface SkyboxRenderOptions {
	strength: number;
	tintLinear: {
		r: number;
		g: number;
		b: number;
	};
	exposure: number;
}

/** @internal Minimal camera view required for software skybox projection. */
export interface SoftwareSkyboxCamera {
	readonly type: CameraType;
	readonly fov: number;
	readonly aspectRatio: number;
	readonly viewMatrix: Matrix4;
}

export interface SoftwareSkyboxClipRegion {
	readonly minX: number;
	readonly minY: number;
	readonly maxXExclusive: number;
	readonly maxYExclusive: number;
}

export class SkyboxRenderer {
	public static render(
		environmentBackgroundTexture: Texture,
		options: SkyboxRenderOptions,
		pixels: Float32Array,
		camera: SoftwareSkyboxCamera,
		width: number,
		height: number,
		clipRegions?: readonly SoftwareSkyboxClipRegion[],
	): void {
		const strength = Math.max(0, options.strength);
		const exposure = Math.max(0, options.exposure);
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
				if (
					clipRegions &&
					!clipRegions.some(
						(region) =>
							x >= region.minX &&
							x < region.maxXExclusive &&
							y >= region.minY &&
							y < region.maxYExclusive,
					)
				) {
					continue;
				}
				const ndcX = ((x + 0.5) / width) * 2 - 1;
				const cx = ndcX * aspect * tanHalfFov;
				const dirX = right.x * cx + up.x * cy - backward.x;
				const dirY = right.y * cx + up.y * cy - backward.y;
				const dirZ = right.z * cx + up.z * cy - backward.z;
				const invLen = 1 / Math.sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
				const dx = dirX * invLen;
				const dy = dirY * invLen;
				const dz = dirZ * invLen;
				const color = sampleEnvironmentTextureLevelLinear(
					environmentBackgroundTexture,
					{ x: dx, y: dy, z: dz },
				);
				const idx = rowBase + x * 4;
				pixels[idx] = Math.max(0, color.r * tint.r * exposure * strength);
				pixels[idx + 1] = Math.max(0, color.g * tint.g * exposure * strength);
				pixels[idx + 2] = Math.max(0, color.b * tint.b * exposure * strength);
				pixels[idx + 3] = 1;
			}
		}
	}
}
