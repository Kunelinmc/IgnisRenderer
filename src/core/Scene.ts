import { Matrix4 } from '../maths/Matrix4'
import type { SceneLight } from '../lights'
import type { IVector3 } from '../maths/types'
import type { Texture } from './Texture'
import type { BoundingSphere, IModel } from './types'

export class Scene {
	public models: IModel[]
	public lights: SceneLight[]
	public skybox: Texture | null
	private _version: number
	private _boundsCache: BoundingSphere | null
	private _boundsVersion: number

	constructor() {
		this.models = []
		this.lights = []
		this.skybox = null
		this._version = 0
		this._boundsCache = null
		this._boundsVersion = -1
	}

	public addModel(model: IModel): IModel {
		this.models.push(model)
		this.invalidate()
		return model
	}

	public removeModel(model: IModel): boolean {
		const index = this.models.indexOf(model)
		if (index === -1) {
			return false
		}

		this.models.splice(index, 1)
		this.invalidate()
		return true
	}

	public addLight(light: SceneLight): SceneLight {
		this.lights.push(light)
		this.invalidate()
		return light
	}

	public removeLight(light: SceneLight): boolean {
		const index = this.lights.indexOf(light)
		if (index === -1) {
			return false
		}

		this.lights.splice(index, 1)
		this.invalidate()
		return true
	}

	public clear(): void {
		this.models = []
		this.lights = []
		this.invalidate()
	}

	public invalidate(): void {
		this._version++
		this._boundsCache = null
	}

	public get version(): number {
		return this._version
	}

	public getBounds(): BoundingSphere {
		if (this._boundsCache && this._boundsVersion === this._version) {
			return this._boundsCache
		}

		let min: IVector3 = { x: Infinity, y: Infinity, z: Infinity }
		let max: IVector3 = { x: -Infinity, y: -Infinity, z: -Infinity }

		for (const model of this.models) {
			if (model.visible === false) continue

			const transform = model.transform
			const radius =
				model.boundingSphere.radius *
				Math.max(
					Math.abs(transform.scale.x),
					Math.abs(transform.scale.y),
					Math.abs(transform.scale.z)
				)

			const worldCenter = Matrix4.transformPoint(
				Matrix4.fromTransform(model.transform),
				model.boundingSphere.center
			)

			min.x = Math.min(min.x, worldCenter.x - radius)
			min.y = Math.min(min.y, worldCenter.y - radius)
			min.z = Math.min(min.z, worldCenter.z - radius)
			max.x = Math.max(max.x, worldCenter.x + radius)
			max.y = Math.max(max.y, worldCenter.y + radius)
			max.z = Math.max(max.z, worldCenter.z + radius)
		}

		if (min.x === Infinity) {
			this._boundsCache = { center: { x: 0, y: 0, z: 0 }, radius: 100 }
			this._boundsVersion = this._version
			return this._boundsCache
		}

		const center: IVector3 = {
			x: (min.x + max.x) / 2,
			y: (min.y + max.y) / 2,
			z: (min.z + max.z) / 2,
		}
		const size: IVector3 = {
			x: max.x - min.x,
			y: max.y - min.y,
			z: max.z - min.z,
		}
		const radius =
			Math.sqrt(size.x * size.x + size.y * size.y + size.z * size.z) / 2

		this._boundsCache = { center, radius }
		this._boundsVersion = this._version

		return this._boundsCache
	}
}
