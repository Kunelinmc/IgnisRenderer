import { CameraType } from "../../cameras/Camera";
import type { DrawPacket, FrameContext, PreparedScene } from "../../pipeline/types";
import { createTransientStore } from "../../pipeline/types";
import { Matrix4 } from "../../maths/Matrix4";
import { Plane } from "../../maths/Plane";
import type { IRenderTexture, IBindingGroup } from "../types";
import { TextureFormat, TextureUsage } from "../types";
import type { ICommandEncoder } from "../ICommandEncoder";
import type { WebGPUBackend } from "../WebGPUBackend";
import type { WebGPURenderResources } from "./WebGPURenderResources";
import type { WebGPUFrameTargets } from "./WebGPUPostProcessContracts";
import {
	submitWebGPUDraws,
} from "./WebGPUDrawSubmission";
import { Logger } from "../../foundation/Logger";

export const WEBGPU_PLANAR_REFLECTION_MAX_PLANES = 2;
export const WEBGPU_PLANAR_REFLECTION_RESOLUTION_SCALE = 0.5;

interface PlanarReflectionTargetSet {
	sceneColor: IRenderTexture;
	gAlbedoAlpha: IRenderTexture;
	gNormalRoughMetal: IRenderTexture;
	gEmissiveOcclusion: IRenderTexture;
	gMotionDepth: IRenderTexture;
	depth: IRenderTexture;
	width: number;
	height: number;
}

interface ActivePlanarReflection {
	key: string;
	plane: Plane;
	targets: PlanarReflectionTargetSet;
}

export interface WebGPUPlanarReflectionMSAATargets {
	sceneColorMain: IRenderTexture;
	planarReflectionMask: IRenderTexture;
	depth: IRenderTexture;
}

export interface WebGPUPlanarReflectionCompositeRequest {
	encoder: ICommandEncoder;
	context: FrameContext;
	frameTargets: WebGPUFrameTargets;
	msaaTargets: WebGPUPlanarReflectionMSAATargets | null;
}

/**
 * Captures and composites bounded planar reflections for the WebGPU backend.
 */
export class WebGPUPlanarReflectionPass {
	private _backend: WebGPUBackend;
	private _resources: WebGPURenderResources;
	private _targets = new Map<string, PlanarReflectionTargetSet>();
	private _activeReflections: ActivePlanarReflection[] = [];
	private _bindings = new Map<IRenderTexture, IBindingGroup>();

	constructor(backend: WebGPUBackend, resources: WebGPURenderResources) {
		this._backend = backend;
		this._resources = resources;
	}

	/**
	 * Captures all active planar reflection textures for the frame.
	 *
	 * @param context Current frame context.
	 * @returns Number of planes captured.
	 * @sideEffects Submits one WebGPU command buffer for each active capture and
	 * temporarily rewrites WebGPU frame bindings for mirrored camera rendering.
	 */
	public async capture(context: FrameContext): Promise<number> {
		this._activeReflections = [];
		if (context.camera.type === CameraType.Orthographic) {
			Logger.warn(
				"[webgpu-planar-reflection-orthographic-disabled] WebGPU planar reflections are disabled for orthographic cameras.",
				{
					scope: "WebGPUPlanarReflectionPass",
					onceKey: "webgpu-planar-reflection-orthographic-disabled",
				}
			);
			this._releaseStaleTargets(new Set());
			return 0;
		}

		const planes = collectActiveReflectionPlanes(context);
		if (planes.length <= 0) {
			this._releaseStaleTargets(new Set());
			return 0;
		}

		const width = Math.max(
			1,
			Math.floor(
				context.attachments.width * WEBGPU_PLANAR_REFLECTION_RESOLUTION_SCALE
			)
		);
		const height = Math.max(
			1,
			Math.floor(
				context.attachments.height * WEBGPU_PLANAR_REFLECTION_RESOLUTION_SCALE
			)
		);
		const activeKeys = new Set<string>();
		const restoreSceneTargetMode =
			this._backend.getFrameSceneTargetMode?.() ?? "mrt";
		const restoreMSAASampleCount = this._backend.getMSAASampleCount?.() ?? 1;

		try {
			if (restoreMSAASampleCount !== 1) {
				this._backend.setMSAASampleCount?.(1);
			}
			for (const planeInfo of planes) {
				activeKeys.add(planeInfo.key);
				const targets = this._getTargets(planeInfo.key, width, height);
				const captureContext = createPlanarCaptureContext(
					context,
					planeInfo.plane,
					planeInfo.key,
					width,
					height
				);
				this._resources.setSceneTargetMode("mrt");
				this._resources.prepareFrame(captureContext, {
					temporalStateMode: "disabled",
				});
				const encoder = this._backend.createCommandEncoder();
				await this._resources.buildClusteredLighting(encoder);
				await this._recordCapture(encoder, captureContext, targets);
				this._backend.submit([encoder.finish()]);
				this._activeReflections.push({
					key: planeInfo.key,
					plane: planeInfo.plane,
					targets,
				});
			}
		} finally {
			if (restoreMSAASampleCount !== 1) {
				this._backend.setMSAASampleCount?.(restoreMSAASampleCount);
			}
			this._resources.setSceneTargetMode(restoreSceneTargetMode);
			this._resources.prepareFrame(context, { temporalStateMode: "reuse" });
			this._releaseStaleTargets(activeKeys);
		}

		return this._activeReflections.length;
	}

	/**
	 * Composites captured planar reflections into the current scene color.
	 *
	 * @param request Composite draw inputs for the active frame.
	 * @sideEffects Records render commands into `request.encoder`.
	 */
	public async composite(
		request: WebGPUPlanarReflectionCompositeRequest
	): Promise<void> {
		if (
			!request.context.features.enableReflection ||
			request.context.scene.reflectivePackets.length <= 0
		) {
			this._activeReflections = [];
			this._releaseStaleTargets(new Set());
			return;
		}
		if (this._activeReflections.length <= 0) {
			return;
		}
		const frameTargets = request.frameTargets;
		if (!frameTargets.planarReflectionMask) {
			Logger.warn(
				"[webgpu-planar-reflection-mask-unavailable] WebGPU planar reflection mask target is unavailable; skipping planar reflection composite.",
				{
					scope: "WebGPUPlanarReflectionPass",
					onceKey: "webgpu-planar-reflection-mask-unavailable",
				}
			);
			return;
		}

		const activeByKey = new Map(
			this._activeReflections.map((reflection) => [reflection.key, reflection])
		);
		const packets = request.context.scene.reflectivePackets.filter((packet) =>
			activeByKey.has(resolvePlaneKey(packet.material.mirrorPlane))
		);
		if (packets.length <= 0) {
			return;
		}

		const msaaTargets = request.msaaTargets;
		const sceneColorAttachment =
			msaaTargets?.sceneColorMain ?? frameTargets.sceneColorMain;
		const maskAttachment =
			msaaTargets?.planarReflectionMask ?? frameTargets.planarReflectionMask;
		const depthAttachment = msaaTargets?.depth ?? frameTargets.depth;
		const dirtyRects = resolveDirtyRects(
			request.context,
			sceneColorAttachment.width,
			sceneColorAttachment.height
		);
		this._resources.setSceneTargetMode("mrt");
		request.encoder.beginRenderPass({
			label: "WebGPUPlanarReflectionComposite",
			colorAttachments: [
				{
					view: sceneColorAttachment,
					resolveTarget:
						msaaTargets ? frameTargets.sceneColorMain : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: maskAttachment,
					resolveTarget:
						msaaTargets ? frameTargets.planarReflectionMask : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});

		await submitWebGPUDraws({
			encoder: request.encoder,
			resources: this._resources,
			packets,
			dirtyRects,
			resolveDrawOptions: () => ({
				sceneTargetMode: "mrt",
				drawMode: "planar-reflection-composite",
			}),
			resolveBindings: (draw, packet) => {
				const reflection = activeByKey.get(
					resolvePlaneKey(packet.material.mirrorPlane)
				);
				const binding = this._getReflectionBinding(
					reflection!.targets.sceneColor
				);
				return [
					{ slot: 0, group: draw.frameBinding },
					{ slot: 1, group: draw.modelBinding },
					{ slot: 2, group: binding },
				];
			},
		});

		request.encoder.endRenderPass();
	}

	/**
	 * Releases all GPU resources owned by this pass.
	 *
	 * @sideEffects Destroys cached render targets and bind groups.
	 */
	public destroy(): void {
		for (const targets of this._targets.values()) {
			this._destroyTargets(targets);
		}
		this._targets.clear();
		for (const binding of this._bindings.values()) {
			destroyBindingGroup(binding);
		}
		this._bindings.clear();
		this._activeReflections = [];
	}

	private async _recordCapture(
		encoder: ICommandEncoder,
		context: FrameContext,
		targets: PlanarReflectionTargetSet
	): Promise<void> {
		const drewEnvironment = await this._recordEnvironmentCapture(
			encoder,
			targets
		);

		encoder.beginRenderPass({
			label: "WebGPUPlanarReflectionCaptureMain",
			colorAttachments: [
				{
					view: targets.sceneColor,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: drewEnvironment ? "load" : "clear",
					storeOp: "store",
				},
				{
					view: targets.gAlbedoAlpha,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
				{
					view: targets.gNormalRoughMetal,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
				{
					view: targets.gEmissiveOcclusion,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
				{
					view: targets.gMotionDepth,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: targets.depth,
				depthClearValue: 1,
				depthLoadOp: drewEnvironment ? "load" : "clear",
				depthStoreOp: "store",
			},
		});

		const packets = [
			...context.scene.opaquePackets,
			...context.scene.transparentPackets,
		];
		await submitWebGPUDraws({
			encoder,
			resources: this._resources,
			packets,
			resolveDrawOptions: () => ({
				sceneTargetMode: "mrt",
				drawMode: "reflection-capture",
			}),
		});
		encoder.endRenderPass();
	}

	private async _recordEnvironmentCapture(
		encoder: ICommandEncoder,
		targets: PlanarReflectionTargetSet
	): Promise<boolean> {
		const environmentResources =
			await this._resources.getEnvironmentResources("mrt");
		if (!environmentResources) {
			return false;
		}

		encoder.beginRenderPass({
			label: "WebGPUPlanarReflectionCaptureEnvironment",
			colorAttachments: [
				{
					view: targets.sceneColor,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: targets.depth,
				depthClearValue: 1,
				depthLoadOp: "clear",
				depthStoreOp: "store",
			},
		});
		encoder.setPipeline(environmentResources.pipeline);
		encoder.setBindingGroup(0, environmentResources.frameBinding);
		encoder.draw(3);
		encoder.endRenderPass();
		return true;
	}

	private _getTargets(
		key: string,
		width: number,
		height: number
	): PlanarReflectionTargetSet {
		const existing = this._targets.get(key);
		if (
			existing &&
			existing.width === width &&
			existing.height === height
		) {
			return existing;
		}
		if (existing) {
			this._destroyTargets(existing);
		}
		const targets = createTargets(this._backend, key, width, height);
		this._targets.set(key, targets);
		return targets;
	}

	private _releaseStaleTargets(activeKeys: Set<string>): void {
		for (const [key, targets] of Array.from(this._targets.entries())) {
			if (activeKeys.has(key)) {
				continue;
			}
			this._destroyTargets(targets);
			this._targets.delete(key);
		}
	}

	private _getReflectionBinding(texture: IRenderTexture): IBindingGroup {
		const cached = this._bindings.get(texture);
		if (cached) {
			return cached;
		}
		const binding = this._backend.createBindingGroup({
			layout: this._resources.getPlanarReflectionLayout(),
			entries: [
				{ binding: 0, resource: texture },
			],
			label: "WebGPUPlanarReflectionBinding",
		});
		this._bindings.set(texture, binding);
		return binding;
	}

	private _destroyTargets(targets: PlanarReflectionTargetSet): void {
		const binding = this._bindings.get(targets.sceneColor);
		if (binding) {
			destroyBindingGroup(binding);
			this._bindings.delete(targets.sceneColor);
		}
		destroyTargets(targets);
	}
}

function collectActiveReflectionPlanes(context: FrameContext): Array<{
	key: string;
	plane: Plane;
}> {
	const result: Array<{ key: string; plane: Plane }> = [];
	const seen = new Set<string>();
	for (const packet of context.scene.reflectivePackets) {
		const key = resolvePlaneKey(packet.material.mirrorPlane);
		if (!key || seen.has(key)) {
			continue;
		}
		const plane = normalizePlane(packet.material.mirrorPlane);
		if (!plane) {
			continue;
		}
		seen.add(key);
		result.push({ key, plane });
		if (result.length >= WEBGPU_PLANAR_REFLECTION_MAX_PLANES) {
			break;
		}
	}
	return result;
}

function createPlanarCaptureContext(
	context: FrameContext,
	plane: Plane,
	planeKey: string,
	width: number,
	height: number
): FrameContext {
	const camera = context.camera.clone(false);
	const originalPosition = context.camera.getWorldPosition();
	const reflectionMatrix = Matrix4.reflection(plane);
	const mirroredPosition = Matrix4.transformPoint(
		reflectionMatrix,
		originalPosition
	);
	const mirrorViewMatrix = Matrix4.multiply(
		context.camera.viewMatrix,
		reflectionMatrix
	);
	const mirrorProjectionMatrix = context.camera.projectionMatrix.clone();
	const isCameraAbove = plane.distanceToPoint(originalPosition) > 0;
	const clipPlaneNormal = Matrix4.transformDirection(
		mirrorViewMatrix,
		plane.normal
	);
	let clipPlaneConstant = plane.distanceToPoint(mirroredPosition);
	if (!isCameraAbove) {
		clipPlaneNormal.x *= -1;
		clipPlaneNormal.y *= -1;
		clipPlaneNormal.z *= -1;
		clipPlaneConstant *= -1;
	}
	mirrorProjectionMatrix.applyObliqueClipping({
		normal: clipPlaneNormal,
		constant: clipPlaneConstant,
	});

	camera.position.set(
		mirroredPosition.x,
		mirroredPosition.y,
		mirroredPosition.z
	);
	camera.viewMatrix = mirrorViewMatrix;
	camera.projectionMatrix = mirrorProjectionMatrix;
	camera.viewProjectionMatrix = Matrix4.multiply(
		mirrorProjectionMatrix,
		mirrorViewMatrix
	);
	camera.frustum.setFromMatrix(camera.viewProjectionMatrix);

	const captureScene = createCaptureScene(
		context,
		camera,
		plane,
		planeKey,
		isCameraAbove
	);
	const capturePostProcess = context.postProcess.withPassDisabled("ssr");
	const transient = createTransientStore(context.transient);

	return {
		camera,
		attachments: { width, height },
		features: {
			...context.features,
			enableReflection: false,
		},
		postProcess: capturePostProcess,
		shadowMaps: captureScene.shadowMaps,
		scene: captureScene,
		shCoeffs: context.shCoeffs,
		shAmbientCoeffs: context.shAmbientCoeffs,
		worldMatrix: context.worldMatrix,
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width, height }],
			dirtyTileSize: Math.max(width, height),
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			postProcessStartPass: null,
			reasonMask: 0,
			temporalHistoryReset: true,
		},
		transient,
	};
}

function createCaptureScene(
	context: FrameContext,
	camera: FrameContext["camera"],
	plane: Plane,
	planeKey: string,
	isCameraAbove: boolean
): PreparedScene {
	const opaquePackets = filterCapturePackets(
		context.scene.opaquePackets,
		plane,
		planeKey,
		isCameraAbove
	);
	const transparentPackets = filterCapturePackets(
		context.scene.transparentPackets,
		plane,
		planeKey,
		isCameraAbove
	);
	return {
		...context.scene,
		camera,
		opaquePackets,
		transparentPackets,
		particleSystems: [],
		reflectivePackets: [],
		spatialIndex: null,
	};
}

function filterCapturePackets(
	packets: DrawPacket[],
	plane: Plane,
	planeKey: string,
	isCameraAbove: boolean
): DrawPacket[] {
	return packets.filter((packet) => {
		if (resolvePlaneKey(packet.material.mirrorPlane) === planeKey) {
			return false;
		}
		const distance = plane.distanceToPoint(packet.worldBounds.center);
		return isCameraAbove ? distance >= 0 : distance <= 0;
	});
}

function normalizePlane(
	planeLike: DrawPacket["material"]["mirrorPlane"]
): Plane | null {
	if (!planeLike) {
		return null;
	}
	const plane = new Plane(
		{
			x: planeLike.normal.x,
			y: planeLike.normal.y,
			z: planeLike.normal.z,
		},
		planeLike.constant
	);
	plane.normalize();
	const normalLength = Math.hypot(
		plane.normal.x,
		plane.normal.y,
		plane.normal.z
	);
	if (!Number.isFinite(normalLength) || normalLength <= 0) {
		return null;
	}
	return plane;
}

function resolvePlaneKey(
	planeLike: DrawPacket["material"]["mirrorPlane"]
): string {
	const plane = normalizePlane(planeLike);
	if (!plane) {
		return "";
	}
	return [
		plane.normal.x,
		plane.normal.y,
		plane.normal.z,
		plane.constant,
	]
		.map((value) => value.toFixed(6))
		.join(",");
}

function createTargets(
	backend: WebGPUBackend,
	key: string,
	width: number,
	height: number
): PlanarReflectionTargetSet {
	const safeKey = key.replace(/[^a-zA-Z0-9_-]/g, "_");
	return {
		sceneColor: backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
			label: `WebGPUPlanarReflectionColor_${safeKey}`,
		}),
		gAlbedoAlpha: backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA8Unorm,
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUPlanarReflectionAlbedo_${safeKey}`,
		}),
		gNormalRoughMetal: backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUPlanarReflectionNormal_${safeKey}`,
		}),
		gEmissiveOcclusion: backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUPlanarReflectionEmissive_${safeKey}`,
		}),
		gMotionDepth: backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUPlanarReflectionMotion_${safeKey}`,
		}),
		depth: backend.createTexture({
			width,
			height,
			format: TextureFormat.Depth32Float,
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUPlanarReflectionDepth_${safeKey}`,
		}),
		width,
		height,
	};
}

function destroyTargets(targets: PlanarReflectionTargetSet): void {
	targets.sceneColor.destroy();
	targets.gAlbedoAlpha.destroy();
	targets.gNormalRoughMetal.destroy();
	targets.gEmissiveOcclusion.destroy();
	targets.gMotionDepth.destroy();
	targets.depth.destroy();
}

function destroyBindingGroup(group: IBindingGroup | null): void {
	const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
	if (typeof destroyFn === "function") {
		destroyFn.call(group);
	}
}

function resolveDirtyRects(
	context: FrameContext,
	targetWidth: number,
	targetHeight: number
): Array<{ x: number; y: number; width: number; height: number }> {
	const width = Math.max(1, Math.floor(targetWidth));
	const height = Math.max(1, Math.floor(targetHeight));
	if (
		!context.incremental?.enabled ||
		context.incremental.forceFullFrame ||
		context.incremental.dirtyRects.length <= 0
	) {
		return [{ x: 0, y: 0, width, height }];
	}
	const sourceWidth = Math.max(1, Math.floor(context.attachments.width));
	const sourceHeight = Math.max(1, Math.floor(context.attachments.height));
	const scaleX = width / sourceWidth;
	const scaleY = height / sourceHeight;
	return context.incremental.dirtyRects
		.map((rect) => {
			const minX = Math.max(0, Math.floor(rect.x * scaleX));
			const minY = Math.max(0, Math.floor(rect.y * scaleY));
			const maxX = Math.min(width, Math.ceil((rect.x + rect.width) * scaleX));
			const maxY = Math.min(height, Math.ceil((rect.y + rect.height) * scaleY));
			return {
				x: minX,
				y: minY,
				width: maxX - minX,
				height: maxY - minY,
			};
		})
		.filter((rect) => rect.width > 0 && rect.height > 0);
}
