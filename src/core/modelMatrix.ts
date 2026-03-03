import { Matrix4 } from '../maths/Matrix4'
import type { IModel } from './types'

export function getModelMatrix(model: IModel): Matrix4 {
	const transform = model.transform
	const scaleMatrix = [
		[transform.scale.x, 0, 0, 0],
		[0, transform.scale.y, 0, 0],
		[0, 0, transform.scale.z, 0],
		[0, 0, 0, 1],
	]
	const rotationMatrix = Matrix4.rotationFromEuler(
		transform.rotation.x,
		transform.rotation.y,
		transform.rotation.z
	)
	const translationMatrix = [
		[1, 0, 0, transform.position.x],
		[0, 1, 0, transform.position.y],
		[0, 0, 1, transform.position.z],
		[0, 0, 0, 1],
	]

	return Matrix4.multiply(
		new Matrix4(translationMatrix),
		Matrix4.multiply(rotationMatrix, new Matrix4(scaleMatrix))
	)
}
