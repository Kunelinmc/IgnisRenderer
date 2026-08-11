import type { FrameContext } from "../../../pipeline/types";
import type { BackendPostProcessRuntime } from "../../../postprocess/BackendPostProcessRuntime";

import {
	WEBGL_FRAME_GRAPH_NODE_KINDS,
	type WebGLFrameGraphNode,
	type WebGLFrameGraphNodeKind,
} from "./types";

export interface WebGLFrameNodeServices {
	clearFrameTargets(context: FrameContext): void;
	renderEnvironmentNode(context: FrameContext): void;
	renderShadowNode(context: FrameContext): void;
	renderOpaqueDepthPrepass(context: FrameContext): Set<string>;
	renderOpaqueScene(
		context: FrameContext,
		earlyZPacketIds: ReadonlySet<string>,
	): void;
	renderTransparentLegacy(context: FrameContext): void;
	prepareTransmissionDepth(context: FrameContext): void;
	renderLegacyTransparentSegment(
		context: FrameContext,
		start: number,
		end: number,
	): void;
	copyTransmissionBackground(context: FrameContext): void;
	renderTransmissionPacket(context: FrameContext, index: number): void;
	prepareOITTransparent(context: FrameContext): void;
	renderOITTransparentAccum(context: FrameContext): void;
	renderOITTransparentReveal(context: FrameContext): void;
	copySceneColorForOIT(context: FrameContext): void;
	resolveOIT(context: FrameContext): void;
	renderOITLegacyTransparent(context: FrameContext): void;
	prepareOITParticles(): void;
	renderOITParticleAccum(context: FrameContext): void;
	renderOITParticleReveal(context: FrameContext): void;
	renderParticlesLegacy(context: FrameContext): void;
	renderOITAdditiveParticles(context: FrameContext): void;
	presentFrame(): void;
}

export interface WebGLFrameNodeExecutionState {
	earlyZPacketIds: ReadonlySet<string>;
}

export type WebGLFrameNodeExecutor = (
	node: WebGLFrameGraphNode,
	context: FrameContext,
) => void | Promise<void>;

/** Assigns every WebGL frame-graph node kind to exactly one executor. */
export class WebGLFrameNodeExecutorRegistry {
	private readonly _executors = new Map<
		WebGLFrameGraphNodeKind,
		WebGLFrameNodeExecutor
	>();

	public constructor(
		entries: readonly (readonly [WebGLFrameGraphNodeKind, WebGLFrameNodeExecutor])[],
	) {
		for (const [kind, executor] of entries) {
			if (this._executors.has(kind)) {
				throw new Error(
					`WebGL frame node kind "${kind}" has duplicate runtime owners.`,
				);
			}
			this._executors.set(kind, executor);
		}
		const missing = WEBGL_FRAME_GRAPH_NODE_KINDS.filter(
			(kind) => !this._executors.has(kind),
		);
		if (missing.length > 0) {
			throw new Error(
				`WebGL frame node runtimes are missing executors: ${missing.join(", ")}.`,
			);
		}
	}

	public static fromServices(
		services: WebGLFrameNodeServices,
		_postProcessRuntime: BackendPostProcessRuntime,
		state: WebGLFrameNodeExecutionState,
	): WebGLFrameNodeExecutorRegistry {
		return new WebGLFrameNodeExecutorRegistry([
			["frame-setup", () => {}],
			["opaque-external", () => {}],
			["scene-clear", (_node, context) => services.clearFrameTargets(context)],
			["environment", (_node, context) => services.renderEnvironmentNode(context)],
			["shadow", (_node, context) => services.renderShadowNode(context)],
			["opaque-depth-prepass", (_node, context) => {
				state.earlyZPacketIds = services.renderOpaqueDepthPrepass(context);
			}],
			["opaque-scene", (_node, context) =>
				services.renderOpaqueScene(context, state.earlyZPacketIds)],
			["transparent-legacy", (node, context) => {
				if (node.packetStart !== undefined && node.packetEnd !== undefined) {
					services.renderLegacyTransparentSegment(
						context,
						node.packetStart,
						node.packetEnd,
					);
					return;
				}
				if (node.scope === "transparent" || node.scope === "particles") {
					services.renderOITLegacyTransparent(context);
					return;
				}
				services.renderTransparentLegacy(context);
			}],
			["transmission-depth-copy", (_node, context) =>
				services.prepareTransmissionDepth(context)],
			["transmission-background-copy", (_node, context) =>
				services.copyTransmissionBackground(context)],
			["transmission-draw", (node, context) =>
				services.renderTransmissionPacket(context, node.packetIndex ?? -1)],
			["oit-clear", (node, context) => {
				if (node.scope === "particles") {
					services.prepareOITParticles();
					return;
				}
				services.prepareOITTransparent(context);
			}],
			["oit-accum", (node, context) => {
				if (node.scope === "particles") {
					services.renderOITParticleAccum(context);
					return;
				}
				services.renderOITTransparentAccum(context);
			}],
			["oit-reveal", (node, context) => {
				if (node.scope === "particles") {
					services.renderOITParticleReveal(context);
					return;
				}
				services.renderOITTransparentReveal(context);
			}],
			["oit-copy-scene-color", (_node, context) =>
				services.copySceneColorForOIT(context)],
			["oit-resolve", (_node, context) => services.resolveOIT(context)],
			["particles", (node, context) => {
				if (node.scope === "particles") {
					services.renderOITAdditiveParticles(context);
					return;
				}
				services.renderParticlesLegacy(context);
			}],
			["post-process-pass", () => {
				throw new Error("WebGL post-process pass nodes require the stage transaction coordinator.");
			}],
			["present", () => services.presentFrame()],
		]);
	}

	public execute(
		node: WebGLFrameGraphNode,
		context: FrameContext,
	): void | Promise<void> {
		const executor = this._executors.get(node.kind);
		if (typeof executor !== "function") {
			throw new Error(
				`WebGL frame graph node kind "${node.kind}" has no executor.`,
			);
		}
		return executor(node, context);
	}
}
