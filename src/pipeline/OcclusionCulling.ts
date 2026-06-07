import type { Camera } from "../cameras/Camera";
import { AlphaMode } from "../materials/Material";
import { ShaderMaterial } from "../materials/ShaderMaterial";
import { materialUsesTransmission } from "../materials/transparency";
import { Matrix4 } from "../maths/Matrix4";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../core/types";
import type { BoundingSphere } from "../core/types";
import type { DirtyRect } from "./incremental";
import type {
	DrawPacket,
	OcclusionCullingOptions,
} from "./types";
import {
	DEFAULT_OCCLUSION_CULLING_OPTIONS,
	DRAW_PACKET_FLAG_TRANSPARENT,
} from "./types";
import { computePacketScreenRect } from "./screenBounds";

export type NormalizedOcclusionCullingOptions = Required<
	Pick<
		OcclusionCullingOptions,
		| "minCandidateScreenAreaPx"
		| "minOccluderScreenAreaPx"
		| "hysteresisFrames"
		| "maxReadbackLatencyFrames"
		| "debug"
	>
>;

export interface OcclusionCandidate {
	readonly packetId: string;
	readonly packet: DrawPacket;
	readonly screenRect: DirtyRect;
	readonly worldBounds: BoundingSphere;
	readonly nearDepth: number;
	readonly farDepth: number;
	readonly screenAreaPx: number;
	readonly signatureA: number;
	readonly signatureB: number;
	readonly eligible: boolean;
}

export interface OcclusionVisibilityProvider {
	readonly sourceFrameIndex: number;
	isPacketVisible(candidate: OcclusionCandidate): boolean;
}

export interface OcclusionCullingBackendAdapter {
	getVisibilityProvider(
		options: NormalizedOcclusionCullingOptions
	): OcclusionVisibilityProvider | null;
	resetOcclusionCulling?(): void;
}

const HASH_INIT_A = 2166136261;
const HASH_INIT_B = 2246822519;
const HASH_PRIME = 16777619;
const FLOAT64_SCRATCH = new DataView(new ArrayBuffer(8));

export function normalizeOcclusionCullingOptions(
	options: OcclusionCullingOptions | undefined
): NormalizedOcclusionCullingOptions {
	return {
		minCandidateScreenAreaPx: resolvePositiveNumber(
			options?.minCandidateScreenAreaPx,
			DEFAULT_OCCLUSION_CULLING_OPTIONS.minCandidateScreenAreaPx
		),
		minOccluderScreenAreaPx: resolvePositiveNumber(
			options?.minOccluderScreenAreaPx,
			DEFAULT_OCCLUSION_CULLING_OPTIONS.minOccluderScreenAreaPx
		),
		hysteresisFrames: resolvePositiveInteger(
			options?.hysteresisFrames,
			DEFAULT_OCCLUSION_CULLING_OPTIONS.hysteresisFrames
		),
		maxReadbackLatencyFrames: resolvePositiveInteger(
			options?.maxReadbackLatencyFrames,
			DEFAULT_OCCLUSION_CULLING_OPTIONS.maxReadbackLatencyFrames
		),
		debug: options?.debug === true,
	};
}

export function buildOcclusionCandidate(
	packet: DrawPacket,
	camera: Camera,
	viewportWidth: number,
	viewportHeight: number,
	options: NormalizedOcclusionCullingOptions
): OcclusionCandidate | null {
	const screenRect = computePacketScreenRect(
		packet,
		camera,
		viewportWidth,
		viewportHeight
	);
	if (!screenRect) {
		return null;
	}
	const depthRange = resolveLinearDepthRange(packet.worldBounds, camera);
	if (!depthRange) {
		return null;
	}
	const screenAreaPx = screenRect.width * screenRect.height;
	const signature = computeCandidateSignature(packet, screenRect, depthRange);
	return {
		packetId: packet.id,
		packet,
		screenRect,
		worldBounds: packet.worldBounds,
		nearDepth: depthRange.nearDepth,
		farDepth: depthRange.farDepth,
		screenAreaPx,
		signatureA: signature.a,
		signatureB: signature.b,
		eligible:
			screenAreaPx >= options.minCandidateScreenAreaPx &&
			isPacketEligibleForOcclusionCulling(packet),
	};
}

export function isPacketEligibleForOcclusionCulling(packet: DrawPacket): boolean {
	const material = packet.material;
	if (packet.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) {
		return false;
	}
	if (material instanceof ShaderMaterial) {
		return false;
	}
	if (material.alphaMode === AlphaMode.Mask || material.alphaMode === AlphaMode.Blend) {
		return false;
	}
	if (materialUsesTransmission(material)) {
		return false;
	}
	if (material.wireframe || material.depthWrite === false) {
		return false;
	}
	return (
		(packet.primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) ===
		DEFAULT_PRIMITIVE_DRAW_TOPOLOGY
	);
}

function resolveLinearDepthRange(
	worldBounds: BoundingSphere,
	camera: Camera
): { nearDepth: number; farDepth: number } | null {
	const center = Matrix4.transformPoint(camera.viewMatrix, worldBounds.center);
	const centerDepth = -center.z;
	const radius = worldBounds.radius;
	if (
		!Number.isFinite(centerDepth) ||
		!Number.isFinite(radius) ||
		radius <= 0
	) {
		return null;
	}
	const nearDepth = Math.max(0, centerDepth - radius);
	const farDepth = Math.max(nearDepth, centerDepth + radius);
	if (farDepth <= 0) {
		return null;
	}
	return { nearDepth, farDepth };
}

function computeCandidateSignature(
	packet: DrawPacket,
	screenRect: DirtyRect,
	depthRange: { nearDepth: number; farDepth: number }
): { a: number; b: number } {
	let a = HASH_INIT_A;
	let b = HASH_INIT_B;
	const mixFloat = (value: number) => {
		FLOAT64_SCRATCH.setFloat64(0, value, true);
		a = mix32(a, FLOAT64_SCRATCH.getUint32(0, true));
		a = mix32(a, FLOAT64_SCRATCH.getUint32(4, true));
		b = mix32(b, FLOAT64_SCRATCH.getUint32(4, true) ^ 0x9e3779b9);
		b = mix32(b, FLOAT64_SCRATCH.getUint32(0, true) ^ 0x85ebca6b);
	};
	mixFloat(packet.worldBounds.center.x);
	mixFloat(packet.worldBounds.center.y);
	mixFloat(packet.worldBounds.center.z);
	mixFloat(packet.worldBounds.radius);
	mixFloat(screenRect.x);
	mixFloat(screenRect.y);
	mixFloat(screenRect.width);
	mixFloat(screenRect.height);
	mixFloat(depthRange.nearDepth);
	mixFloat(depthRange.farDepth);
	a = mix32(a, packet.primitive.geometryVersion ?? 0);
	b = mix32(b, (packet.material as { version?: number }).version ?? 0);
	return { a: a >>> 0, b: b >>> 0 };
}

function mix32(hash: number, value: number): number {
	return Math.imul(hash ^ (value >>> 0), HASH_PRIME) >>> 0;
}

function resolvePositiveNumber(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
		return fallback;
	}
	return value;
}

function resolvePositiveInteger(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
		return fallback;
	}
	return Math.max(1, Math.floor(value));
}
