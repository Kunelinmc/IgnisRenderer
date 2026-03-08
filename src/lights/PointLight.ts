import type { IVector3 } from '../maths/types'
import { Light, LightType, type LightParams } from './Light'

export interface PointLightParams extends LightParams {
	position?: IVector3
	range?: number
}

export class PointLight extends Light<LightType.Point> {
	public range: number

	constructor(params: PointLightParams = {}) {
		super(LightType.Point, params)
		if (params.position) {
			this.position.copy(params.position)
		}
		this.range = params.range ?? 1000
	}
}
