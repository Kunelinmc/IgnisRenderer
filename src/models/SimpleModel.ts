import { EventEmitter } from '../core/EventEmitter'
import { GeometryBuilder, type GeometryFace } from './GeometryBuilder'
import { Matrix4 } from '../maths/Matrix4'
import type { IVector3 } from '../maths/types'
import { Vector3 } from '../maths/Vector3'
import type { BoundingBox, BoundingSphere, IModel, IPrimitive } from '../core/types'
import { IdGenerator } from '../utils/IdGenerator'

export type ModelVertex = GeometryFace['vertices'][number]
export type ModelFace = GeometryFace

export class SimpleModel extends EventEmitter implements IModel {
	public readonly id: string
	public primitives: IPrimitive[]
	public visible: boolean
	public transform: {
		rotation: Vector3
		position: Vector3
		scale: Vector3
	}
	public boundingSphere: BoundingSphere
	public boundingBox: BoundingBox

	constructor(primitives: IPrimitive[] = []) {
		super()
		this.id = IdGenerator.nextId('model')
		this.primitives = primitives
		this.visible = true

		this.transform = {
			rotation: new Vector3(0, 0, 0),
			position: new Vector3(0, 0, 0),
			scale: new Vector3(1, 1, 1),
		}

		this.boundingBox = GeometryBuilder.computeModelBoundingBox(this.primitives)
		this.boundingSphere = GeometryBuilder.computeModelBoundingSphere(
			this.primitives,
			this.boundingBox
		)
	}

	public static fromFaces(faces: ModelFace[] = []): SimpleModel {
		return new SimpleModel(GeometryBuilder.buildPrimitivesFromFaces(faces))
	}

	public getWorldBoundingBox(): BoundingBox {
		const box = this.boundingBox
		const modelMatrix = Matrix4.fromTransform(this.transform)

		const corners: IVector3[] = [
			{ x: box.min.x, y: box.min.y, z: box.min.z },
			{ x: box.max.x, y: box.min.y, z: box.min.z },
			{ x: box.min.x, y: box.max.y, z: box.min.z },
			{ x: box.max.x, y: box.max.y, z: box.min.z },
			{ x: box.min.x, y: box.min.y, z: box.max.z },
			{ x: box.max.x, y: box.min.y, z: box.max.z },
			{ x: box.min.x, y: box.max.y, z: box.max.z },
			{ x: box.max.x, y: box.max.y, z: box.max.z },
		]

		let minX = Infinity
		let minY = Infinity
		let minZ = Infinity
		let maxX = -Infinity
		let maxY = -Infinity
		let maxZ = -Infinity

		for (const corner of corners) {
			const worldPoint = Matrix4.transformPoint(modelMatrix, corner)
			if (worldPoint.x < minX) minX = worldPoint.x
			if (worldPoint.y < minY) minY = worldPoint.y
			if (worldPoint.z < minZ) minZ = worldPoint.z
			if (worldPoint.x > maxX) maxX = worldPoint.x
			if (worldPoint.y > maxY) maxY = worldPoint.y
			if (worldPoint.z > maxZ) maxZ = worldPoint.z
		}

		return {
			min: { x: minX, y: minY, z: minZ },
			max: { x: maxX, y: maxY, z: maxZ },
		}
	}
}
