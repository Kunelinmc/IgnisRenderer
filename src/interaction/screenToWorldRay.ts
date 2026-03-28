import { Camera, CameraType } from "../cameras/Camera";
import { OrthographicCamera } from "../cameras/OrthographicCamera";
import type { IVector3 } from "../maths/types";

export interface ScreenRayInput {
	screenX: number;
	screenY: number;
	viewportWidth: number;
	viewportHeight: number;
}

export interface ScreenRay {
	origin: IVector3;
	direction: IVector3;
	ndcX: number;
	ndcY: number;
}

interface CameraBasis {
	right: IVector3;
	up: IVector3;
	backward: IVector3;
}

/**
 * Builds a world-space ray from a screen-space pixel coordinate.
 */
export function screenToWorldRay(
	camera: Camera,
	input: ScreenRayInput
): ScreenRay {
	const width = Math.max(1, Math.floor(input.viewportWidth));
	const height = Math.max(1, Math.floor(input.viewportHeight));
	const ndcX = ((input.screenX + 0.5) / width) * 2 - 1;
	const ndcY = 1 - ((input.screenY + 0.5) / height) * 2;
	const basis = getCameraBasis(camera);
	const cameraPosition = camera.getWorldPosition({ x: 0, y: 0, z: 0 });

	if (camera.type === CameraType.Orthographic) {
		const ortho =
			camera instanceof OrthographicCamera ?
				camera
			: (camera as unknown as OrthographicCamera);
		const bounds =
			typeof ortho.getBounds === "function" ?
				ortho.getBounds()
			: {
					left: -0.5,
					right: 0.5,
					bottom: -0.5,
					top: 0.5,
				};

		const offsetX =
			((ndcX + 1) * 0.5) * (bounds.right - bounds.left) + bounds.left;
		const offsetY =
			((ndcY + 1) * 0.5) * (bounds.top - bounds.bottom) + bounds.bottom;
		const forward = {
			x: -basis.backward.x,
			y: -basis.backward.y,
			z: -basis.backward.z,
		};

		return {
			origin: {
				x:
					cameraPosition.x +
					basis.right.x * offsetX +
					basis.up.x * offsetY +
					forward.x * camera.near,
				y:
					cameraPosition.y +
					basis.right.y * offsetX +
					basis.up.y * offsetY +
					forward.y * camera.near,
				z:
					cameraPosition.z +
					basis.right.z * offsetX +
					basis.up.z * offsetY +
					forward.z * camera.near,
			},
			direction: normalize(forward),
			ndcX,
			ndcY,
		};
	}

	const fovRad = (camera.fov * Math.PI) / 180;
	const tanHalfFov = Math.tan(fovRad * 0.5);
	const aspect = camera.aspectRatio || width / Math.max(1, height);
	const cx = ndcX * aspect * tanHalfFov;
	const cy = ndcY * tanHalfFov;
	const cz = -1;
	const invLen = 1 / Math.max(1e-8, Math.hypot(cx, cy, cz));
	const dirCamX = cx * invLen;
	const dirCamY = cy * invLen;
	const dirCamZ = cz * invLen;

	const direction = normalize({
		x:
			basis.right.x * dirCamX +
			basis.up.x * dirCamY +
			basis.backward.x * dirCamZ,
		y:
			basis.right.y * dirCamX +
			basis.up.y * dirCamY +
			basis.backward.y * dirCamZ,
		z:
			basis.right.z * dirCamX +
			basis.up.z * dirCamY +
			basis.backward.z * dirCamZ,
	});

	return {
		origin: {
			x: cameraPosition.x,
			y: cameraPosition.y,
			z: cameraPosition.z,
		},
		direction,
		ndcX,
		ndcY,
	};
}

function getCameraBasis(camera: Camera): CameraBasis {
	const view = camera.viewMatrix.elements;
	return {
		right: { x: view[0][0], y: view[0][1], z: view[0][2] },
		up: { x: view[1][0], y: view[1][1], z: view[1][2] },
		backward: { x: view[2][0], y: view[2][1], z: view[2][2] },
	};
}

function normalize(vector: IVector3): IVector3 {
	const length = Math.hypot(vector.x, vector.y, vector.z);
	if (!(length > 1e-8)) {
		return { x: 0, y: 0, z: -1 };
	}
	const invLength = 1 / length;
	return {
		x: vector.x * invLength,
		y: vector.y * invLength,
		z: vector.z * invLength,
	};
}
