import { Matrix4 } from "../maths/Matrix4";
import { Projector } from "./Projector";
import type { IVector3 } from "../maths/types";
import type { SceneLight } from "../lights";
import type { Texture } from "./Texture";
import type { IModel, BoundingSphere } from "./types";

export class Scene {
	public models: IModel[];
	public lights: SceneLight[];
	public skybox: Texture | null;

	constructor() {
		this.models = [];
		this.lights = [];
		this.skybox = null;
	}

	public addModel(model: IModel): IModel {
		this.models.push(model);
		return model;
	}

	public removeModel(model: IModel): boolean {
		const index = this.models.indexOf(model);
		if (index !== -1) {
			this.models.splice(index, 1);
			return true;
		}
		return false;
	}

	public addLight(light: SceneLight): SceneLight {
		this.lights.push(light);
		return light;
	}

	public removeLight(light: SceneLight): boolean {
		const index = this.lights.indexOf(light);
		if (index !== -1) {
			this.lights.splice(index, 1);
			return true;
		}
		return false;
	}

	public clear(): void {
		this.models = [];
		this.lights = [];
	}

	public getBounds(): BoundingSphere {
		let min: IVector3 = { x: Infinity, y: Infinity, z: Infinity };
		let max: IVector3 = { x: -Infinity, y: -Infinity, z: -Infinity };

		for (const model of this.models) {
			const transform = model.transform;
			const radius =
				model.boundingSphere.radius *
				Math.max(
					Math.abs(transform.scale.x),
					Math.abs(transform.scale.y),
					Math.abs(transform.scale.z)
				);

			const worldCenter = Matrix4.transformPoint(
				Projector.getModelMatrix(model),
				model.boundingSphere.center
			);

			min.x = Math.min(min.x, worldCenter.x - radius);
			min.y = Math.min(min.y, worldCenter.y - radius);
			min.z = Math.min(min.z, worldCenter.z - radius);
			max.x = Math.max(max.x, worldCenter.x + radius);
			max.y = Math.max(max.y, worldCenter.y + radius);
			max.z = Math.max(max.z, worldCenter.z + radius);
		}

		if (min.x === Infinity) {
			return { center: { x: 0, y: 0, z: 0 }, radius: 100 };
		}

		const center: IVector3 = {
			x: (min.x + max.x) / 2,
			y: (min.y + max.y) / 2,
			z: (min.z + max.z) / 2,
		};
		const size: IVector3 = {
			x: max.x - min.x,
			y: max.y - min.y,
			z: max.z - min.z,
		};
		const radius =
			Math.sqrt(size.x * size.x + size.y * size.y + size.z * size.z) / 2;

		return { center, radius };
	}
}
