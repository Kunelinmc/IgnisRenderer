import type { Camera } from "../../cameras/Camera";
import type { DrawPacket, FrameContext, PreparedSceneEnvironment } from "../../pipeline/types";
import type { DecalPacket } from "../../pipeline/types";
import type { PreparedSceneSpatialIndex, ResolvedFeatureState } from "../../pipeline/types";
import type {
	ParticleMeshRenderBatch,
	ParticleRenderBatch,
} from "../../particles/ParticleRenderBatch";
import type { SceneLight } from "../../lights";
import type { ShadowFramePlan } from "../../lights/shadows/ShadowFramePlan";
import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3, SHCoefficients } from "../../maths/types";
import type { DeformedGeometryMap } from "../../simulation/animation/types";
import { ANIMATION_SOFTWARE_DEFORMED_GEOMETRY_KEY } from "../../simulation/animation/types";
import type { SoftwareTemporalRenderState } from "./SoftwareTemporalRenderState";

const DEFAULT_SOFTWARE_ENVIRONMENT: PreparedSceneEnvironment = {
	backgroundEnabled: true,
	lightingEnabled: true,
	backgroundTexture: null,
	iblTexture: null,
	backgroundStrength: 1,
	diffuseStrength: 1,
	specularStrength: 1,
	backgroundTintLinear: { r: 1, g: 1, b: 1 },
	backgroundExposure: 1,
};

/** @internal Validated CPU attachments used by every Software pass. */
export interface SoftwareFrameAttachments {
	readonly color: Float32Array;
	readonly pixels: Uint8ClampedArray;
	readonly depthBuffer: Float32Array;
	readonly normalBuffer: Float32Array | null;
	readonly motionBuffer: Float32Array | null;
	readonly width: number;
	readonly height: number;
}

/** @internal Clamped half-open pixel region. */
export interface SoftwareClipRegion {
	readonly minX: number;
	readonly minY: number;
	readonly maxXExclusive: number;
	readonly maxYExclusive: number;
}

/** @internal Immutable camera data resolved once for a Software frame. */
export interface SoftwareCameraView {
	readonly type: Camera["type"];
	readonly near: number;
	readonly fov: number;
	readonly aspectRatio: number;
	readonly position: IVector3;
	readonly viewMatrix: Matrix4;
	readonly projectionMatrix: Matrix4;
	readonly viewProjectionMatrix: Matrix4;
}

/** @internal Prepared scene fields consumed by Software render passes. */
export interface SoftwareSceneView {
	readonly lights: SceneLight[];
	readonly environment: PreparedSceneEnvironment;
	readonly opaquePackets: DrawPacket[];
	readonly transparentPackets: DrawPacket[];
	readonly shadowCasterPackets: DrawPacket[];
	readonly shadowTransmitterPackets: DrawPacket[];
	readonly reflectivePackets: DrawPacket[];
	readonly decalPackets: DecalPacket[];
	readonly spatialIndex: PreparedSceneSpatialIndex | null;
}

/** @internal Backend-local view imported from one renderer frame context. */
export interface SoftwareFrameView {
	readonly attachments: SoftwareFrameAttachments;
	readonly camera: SoftwareCameraView;
	readonly scene: SoftwareSceneView;
	readonly features: ResolvedFeatureState;
	readonly shadowPlan: ShadowFramePlan;
	readonly shAmbientCoeffs: SHCoefficients;
	readonly clipRegions: readonly SoftwareClipRegion[];
	readonly incrementalPartial: boolean;
	readonly temporalHistoryReset: boolean;
	readonly temporal: SoftwareTemporalRenderState;
	readonly animationDeformedGeometry: DeformedGeometryMap | undefined;
}

/** @internal Mutable particle payload imported after the simulation stage. */
export interface SoftwareParticleFrameState {
	batches: ParticleRenderBatch[];
	meshBatches: ParticleMeshRenderBatch[];
}

/** @internal Builds the single normalized view consumed by Software passes. */
export function createSoftwareFrameView(
	context: FrameContext,
	temporal: SoftwareTemporalRenderState,
	sceneColor?: Float32Array,
): SoftwareFrameView {
	const attachments = validateSoftwareFrameAttachments(context, sceneColor);
	const incrementalPartial =
		context.incremental.enabled &&
		!context.incremental.forceFullFrame &&
		context.incremental.dirtyRects.length > 0;
	const camera = context.viewCamera;
	const cameraPosition = camera.getWorldPosition();

	return Object.freeze({
		attachments,
		camera: Object.freeze({
			type: camera.type,
			near: camera.near,
			fov: camera.fov,
			aspectRatio: camera.aspectRatio,
			position: Object.freeze({
				x: cameraPosition.x,
				y: cameraPosition.y,
				z: cameraPosition.z,
			}),
			viewMatrix: camera.viewMatrix,
			projectionMatrix: camera.projectionMatrix,
			viewProjectionMatrix: camera.viewProjectionMatrix,
		}),
		scene: Object.freeze({
			lights: context.scene.lights,
			environment: context.scene.environment ?? DEFAULT_SOFTWARE_ENVIRONMENT,
			opaquePackets: context.scene.opaquePackets,
			transparentPackets: context.scene.transparentPackets,
			shadowCasterPackets: context.scene.shadowCasterPackets,
			shadowTransmitterPackets: context.scene.shadowTransmitterPackets,
			reflectivePackets: context.scene.reflectivePackets,
			decalPackets: context.scene.decalPackets,
			spatialIndex: context.scene.spatialIndex,
		}),
		features: context.features,
		shadowPlan: context.shadowPlan,
		shAmbientCoeffs: context.shAmbientCoeffs,
		clipRegions: Object.freeze(resolveSoftwareClipRegions(context, attachments)),
		incrementalPartial,
		temporalHistoryReset: context.incremental.temporalHistoryReset,
		temporal,
		animationDeformedGeometry: context.transient.get(
			ANIMATION_SOFTWARE_DEFORMED_GEOMETRY_KEY,
		),
	});
}

function validateSoftwareFrameAttachments(
	context: FrameContext,
	sceneColor?: Float32Array,
): SoftwareFrameAttachments {
	const attachments = context.attachments;
	const width = attachments.width;
	const height = attachments.height;
	if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
		throw new Error(
			`Software frame attachments invalid: dimensions must be positive integers; received ${width}x${height}.`,
		);
	}
	const pixelCount = width * height;
	const color = sceneColor ?? createSoftwareTestSceneColor(pixelCount);
	if (!(color instanceof Float32Array) || color.length !== pixelCount * 4) {
		throw new Error(
			`Software frame attachments invalid: authoritative color must be a ` +
				`Float32Array with length ${pixelCount * 4}.`,
		);
	}
	if (
		!(attachments.pixels instanceof Uint8ClampedArray) ||
		attachments.pixels.length !== pixelCount * 4
	) {
		throw new Error(
			`Software frame attachments invalid: color must be a Uint8ClampedArray ` +
				`with length ${pixelCount * 4}.`,
		);
	}
	if (
		!(attachments.depthBuffer instanceof Float32Array) ||
		attachments.depthBuffer.length !== pixelCount
	) {
		throw new Error(
			`Software frame attachments invalid: depth must be a Float32Array with ` +
				`length ${pixelCount}.`,
		);
	}
	if (
		attachments.normalBuffer !== null &&
		attachments.normalBuffer !== undefined &&
		(!(attachments.normalBuffer instanceof Float32Array) ||
			attachments.normalBuffer.length !== pixelCount * 3)
	) {
		throw new Error(
			`Software frame attachments invalid: normal must be a Float32Array with ` +
				`length ${pixelCount * 3}.`,
		);
	}
	if (
		attachments.motionBuffer !== null &&
		attachments.motionBuffer !== undefined &&
		(!(attachments.motionBuffer instanceof Float32Array) ||
			attachments.motionBuffer.length !== pixelCount * 4)
	) {
		throw new Error(
			`Software frame attachments invalid: motion must be a Float32Array with ` +
				`length ${pixelCount * 4}.`,
		);
	}
	return Object.freeze({
		color,
		pixels: attachments.pixels,
		depthBuffer: attachments.depthBuffer,
		normalBuffer: attachments.normalBuffer ?? null,
		motionBuffer: attachments.motionBuffer ?? null,
		width,
		height,
	});
}

function createSoftwareTestSceneColor(pixelCount: number): Float32Array {
	const color = new Float32Array(pixelCount * 4);
	for (let pixel = 0; pixel < pixelCount; pixel++) color[(pixel << 2) + 3] = 1;
	return color;
}

function resolveSoftwareClipRegions(
	context: FrameContext,
	attachments: SoftwareFrameAttachments,
): SoftwareClipRegion[] {
	if (
		!context.incremental.enabled ||
		context.incremental.forceFullFrame ||
		context.incremental.dirtyRects.length === 0
	) {
		return [fullSoftwareClipRegion(attachments)];
	}

	const regions: SoftwareClipRegion[] = [];
	for (const rect of context.incremental.dirtyRects) {
		const minX = Math.max(0, Math.floor(rect.x));
		const minY = Math.max(0, Math.floor(rect.y));
		const maxXExclusive = Math.min(
			attachments.width,
			Math.ceil(rect.x + rect.width),
		);
		const maxYExclusive = Math.min(
			attachments.height,
			Math.ceil(rect.y + rect.height),
		);
		if (minX >= maxXExclusive || minY >= maxYExclusive) continue;
		regions.push({ minX, minY, maxXExclusive, maxYExclusive });
	}
	return regions;
}

function fullSoftwareClipRegion(
	attachments: SoftwareFrameAttachments,
): SoftwareClipRegion {
	return {
		minX: 0,
		minY: 0,
		maxXExclusive: attachments.width,
		maxYExclusive: attachments.height,
	};
}

/** @internal Returns whether an inclusive pixel rectangle touches any region. */
export function softwareRectIntersectsClipRegions(
	minX: number,
	minY: number,
	maxX: number,
	maxY: number,
	regions: readonly SoftwareClipRegion[],
): boolean {
	for (const region of regions) {
		if (
			maxX >= region.minX &&
			minX < region.maxXExclusive &&
			maxY >= region.minY &&
			minY < region.maxYExclusive
		) {
			return true;
		}
	}
	return false;
}
