import { CameraType } from "../cameras/Camera";
import type { OrthographicCamera } from "../cameras/OrthographicCamera";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import type { FrameContext } from "../pipeline/types";

export const MAX_INTERACTION_OUTLINE_CIRCLES = 64;

export interface ProjectedOutlineCircle {
	entityId: number;
	centerX: number;
	centerY: number;
	radius: number;
	depth: number;
}

export function collectProjectedOutlineCircles(
	context: FrameContext,
	selectedEntityIds: readonly number[],
	maxCircles: number = MAX_INTERACTION_OUTLINE_CIRCLES
): ProjectedOutlineCircle[] {
	if (selectedEntityIds.length === 0) {
		return [];
	}

	const width = Math.max(1, Math.floor(context.attachments.width));
	const height = Math.max(1, Math.floor(context.attachments.height));
	const selected = new Set<number>(selectedEntityIds);
	const perEntity = new Map<number, ProjectedOutlineCircle>();

	for (const meshInstance of context.scene.meshInstances) {
		if (meshInstance.visible === false) {
			continue;
		}
		const entityId = meshInstance.entityId;
		if (typeof entityId !== "number" || !selected.has(entityId)) {
			continue;
		}
		const sphere = meshInstance.getWorldBoundingSphere();
		const projected = Matrix4.transformPoint(
			context.camera.viewProjectionMatrix,
			sphere.center
		);
		const clipW = projected.w ?? 1;
		if (Math.abs(clipW) < 1e-6) {
			continue;
		}
		const invW = 1 / clipW;
		const ndcX = projected.x * invW;
		const ndcY = projected.y * invW;
		const ndcZ = projected.z * invW;
		const screenX = (ndcX * 0.5 + 0.5) * width;
		const screenY = (0.5 - ndcY * 0.5) * height;
		if (!Number.isFinite(screenX) || !Number.isFinite(screenY)) {
			continue;
		}

		const radius = estimateProjectedRadiusPixels(context, sphere.center, sphere.radius);
		if (!Number.isFinite(radius) || radius <= 1) {
			continue;
		}

		const current = perEntity.get(entityId);
		if (
			!current ||
			radius > current.radius ||
			(radius === current.radius && ndcZ < current.depth)
		) {
			perEntity.set(entityId, {
				entityId,
				centerX: screenX,
				centerY: screenY,
				radius,
				depth: ndcZ,
			});
		}
	}

	const circles = Array.from(perEntity.values()).sort((left, right) => {
		if (left.radius !== right.radius) {
			return right.radius - left.radius;
		}
		if (left.depth !== right.depth) {
			return left.depth - right.depth;
		}
		return left.entityId - right.entityId;
	});
	if (circles.length <= maxCircles) {
		return circles;
	}
	return circles.slice(0, Math.max(1, maxCircles));
}

function estimateProjectedRadiusPixels(
	context: FrameContext,
	worldCenter: IVector3,
	worldRadius: number
): number {
	const camera = context.camera;
	if (camera.type === CameraType.Orthographic) {
		const orthoCamera = camera as OrthographicCamera;
		const bounds = orthoCamera.getBounds();
		const worldHeight = Math.max(1e-6, bounds.top - bounds.bottom);
		return (worldRadius / worldHeight) * context.attachments.height * 2;
	}
	const fovRad = (camera.fov * Math.PI) / 180;
	const focal =
		context.attachments.height / (2 * Math.max(1e-6, Math.tan(fovRad * 0.5)));
	const centerView = Matrix4.transformPoint(camera.viewMatrix, worldCenter);
	const distance = Math.max(1e-3, -centerView.z);
	return (worldRadius * focal) / distance;
}
