import type { WebGPUDeferredDecalPass } from "./WebGPUDeferredDecalPass";
import type { WebGPUDeferredLightingPass } from "./WebGPUDeferredLightingPass";
import {
	WEBGPU_DEFERRED_BASE_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_BASE_COLOR_TARGET_COUNT,
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_TARGET_COUNT,
	WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT,
	type WebGPUDeferredGBufferLayout,
} from "../constants";
import {
	materialRequiresExtendedWebGPUGBuffer,
	materialSupportsWebGPUDeferredLighting,
} from "../material";
import type { PreparedFramePacketSet } from "../../../pipeline/FramePackets";
import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import type {
	WebGPUFrameGraphModule,
	WebGPUFrameGraphContribution,
	WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import {
	defineWebGPUFrameMessage,
	type WebGPUFrameMessageHandler,
} from "./WebGPUFrameMessage";
import {
	WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE,
	WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE,
	WEBGPU_FRAME_FEATURE_STATES,
	WEBGPU_FRAME_CONTEXT_MESSAGE,
	WEBGPU_FRAME_PACKETS_MESSAGE,
} from "./WebGPUFrameMessages";
import {
	createWebGPUFrameGraphNode,
	readWebGPUFrameGraphResource,
	writeWebGPUFrameGraphResource,
} from "./WebGPUFrameGraphDsl";
import type {
	WebGPURecordingFrameSession as WebGPUFrameSession,
} from "./WebGPUFrameSession";
import type { WebGPUFrameGraphNode } from "./types";
import type { WebGPUDeferredOpaqueFrameState } from "./WebGPUScenePassRecorder";
import type { WebGPUFrameExecutionContext } from "./WebGPUFrameExecutionContext";

interface WebGPUDeferredScenePort {
	recordMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean,
	): Promise<void>;
}

/** @internal Deferred-owned handoff between opaque recording and lighting. */
export class WebGPUDeferredOpaqueStatePort {
	private _state: WebGPUDeferredOpaqueFrameState | null = null;

	public publish(state: WebGPUDeferredOpaqueFrameState): void {
		this._state = state;
	}

	public peek(): WebGPUDeferredOpaqueFrameState | null {
		return this._state;
	}

	public consume(): WebGPUDeferredOpaqueFrameState | null {
		const state = this._state;
		this._state = null;
		return state;
	}

	public clear(): void {
		this._state = null;
	}
}

const DEFERRED_GBUFFER_RENDER_RESOURCE_IDS = [
	"gbuffer:albedo-alpha",
	"gbuffer:normal-rough-metal",
	"gbuffer:emissive-occlusion",
	"gbuffer:motion-depth",
	"gbuffer:specular",
	"gbuffer:coat-sheen",
	"gbuffer:sheen-reflectance",
] as const;

const DEFERRED_GBUFFER_BASE_RESOURCE_IDS =
	DEFERRED_GBUFFER_RENDER_RESOURCE_IDS.slice(0, 4);

const DEFERRED_GBUFFER_STORAGE_RESOURCE_IDS = [
	"gbuffer:material-ext0",
	"gbuffer:material-ext3",
] as const;

const DEFERRED_GBUFFER_RESOURCE_IDS = [
	...DEFERRED_GBUFFER_RENDER_RESOURCE_IDS,
	...DEFERRED_GBUFFER_STORAGE_RESOURCE_IDS,
] as const;

export interface WebGPUDeferredFeatureAnalysis {
	readonly hasDeferredLightingWork: boolean;
	readonly deferredGBufferLayout: WebGPUDeferredGBufferLayout;
}

export const WEBGPU_DEFERRED_FEATURE_ANALYSIS =
	defineWebGPUFrameMessage<WebGPUDeferredFeatureAnalysis>({
		id: "webgpu:deferred-analysis",
		ownerId: "deferred",
		phase: "analysis",
	});

export function analyzeWebGPUDeferredFeatures(
	context: FrameContext,
	framePackets: PreparedFramePacketSet,
): WebGPUDeferredFeatureAnalysis {
	return {
		hasDeferredLightingWork: framePackets.opaque.some((packet) =>
			materialSupportsWebGPUDeferredLighting(packet.submission.material.effective)),
		deferredGBufferLayout:
			(context.scene.decalPackets?.length ?? 0) > 0 ||
			framePackets.opaque.some((packet) =>
				materialRequiresExtendedWebGPUGBuffer(packet.submission.material.effective))
				? "extended"
				: "base",
	};
}

export interface WebGPUDeferredConfigurationInput {
	readonly capabilities: {
		readonly maxColorAttachments: number;
		readonly maxColorAttachmentBytesPerSample: number;
		readonly maxStorageTexturesPerShaderStage: number;
	};
	readonly options: {
		readonly sampleCount: number;
		readonly enableDeferredLighting: boolean;
		readonly forceDeferredFallback: boolean;
	};
}

/** @internal Shared deferred capability policy used by frames and warmup. */
export function resolveWebGPUDeferredConfiguration(
	analysis: WebGPUDeferredFeatureAnalysis,
	input: WebGPUDeferredConfigurationInput,
) {
	const targetCount = analysis.deferredGBufferLayout === "base"
		? WEBGPU_DEFERRED_BASE_COLOR_TARGET_COUNT
		: WEBGPU_DEFERRED_COLOR_TARGET_COUNT;
	const byteCount = analysis.deferredGBufferLayout === "base"
		? WEBGPU_DEFERRED_BASE_COLOR_BYTES_PER_SAMPLE
		: WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE;
	const { capabilities, options } = input;
	const supported =
		options.sampleCount === 1 &&
		capabilities.maxColorAttachments >= targetCount &&
		capabilities.maxColorAttachmentBytesPerSample >= byteCount &&
		capabilities.maxStorageTexturesPerShaderStage >=
			WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT;
	const requested = options.enableDeferredLighting &&
		!options.forceDeferredFallback && analysis.hasDeferredLightingWork;
	const diagnostics = [];
	if (options.enableDeferredLighting && !options.forceDeferredFallback && !supported) {
		if (options.sampleCount !== 1) diagnostics.push({
			code: "webgpu-deferred-disabled-msaa",
			message: "WebGPU deferred lighting requires sampleCount=1; using the legacy MRT forward path.",
		});
		if (capabilities.maxColorAttachments < targetCount) diagnostics.push({
			code: "webgpu-deferred-disabled-attachments",
			message: `WebGPU deferred lighting requires ${targetCount} color attachments; device exposes ${capabilities.maxColorAttachments}.`,
		});
		if (capabilities.maxColorAttachmentBytesPerSample < byteCount) diagnostics.push({
			code: "webgpu-deferred-disabled-bytes",
			message: `WebGPU deferred lighting requires ${byteCount} color attachment bytes per sample; device exposes ${capabilities.maxColorAttachmentBytesPerSample}.`,
		});
		if (
			capabilities.maxStorageTexturesPerShaderStage <
			WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT
		) diagnostics.push({
			code: "webgpu-deferred-disabled-storage-textures",
			message: `WebGPU deferred lighting requires ${WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT} storage textures per shader stage; device exposes ${capabilities.maxStorageTexturesPerShaderStage}.`,
		});
	}
	return {
		supported,
		active: requested && supported,
		deferredGBufferLayout: analysis.deferredGBufferLayout,
		diagnostics,
	};
}

/** @internal Owns deferred decal, lighting, and fallback graph execution. */
export class WebGPUDeferredFrameModule implements WebGPUFrameGraphModule {
	public readonly id = "deferred";
	public readonly messageHandlers: readonly WebGPUFrameMessageHandler[] = [{
		id: "analyze",
		moduleId: this.id,
		phase: "analysis" as const,
		inputs: [
			{ descriptor: WEBGPU_FRAME_CONTEXT_MESSAGE },
			{ descriptor: WEBGPU_FRAME_PACKETS_MESSAGE },
		],
		outputs: [WEBGPU_DEFERRED_FEATURE_ANALYSIS],
		run: (messages, publisher) => publisher.publish(
			WEBGPU_DEFERRED_FEATURE_ANALYSIS,
			analyzeWebGPUDeferredFeatures(
				messages.get(WEBGPU_FRAME_CONTEXT_MESSAGE),
				messages.get(WEBGPU_FRAME_PACKETS_MESSAGE),
			),
		),
	}, {
		id: "configure",
		moduleId: this.id,
		phase: "configuration" as const,
		inputs: [
			{ descriptor: WEBGPU_DEFERRED_FEATURE_ANALYSIS },
			{ descriptor: WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE },
		],
		outputs: [WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE],
		run: (messages, publisher) => {
			const analysis = messages.get(WEBGPU_DEFERRED_FEATURE_ANALYSIS);
			const { capabilities, options } = messages.get(
				WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE,
			);
			const resolved = resolveWebGPUDeferredConfiguration(analysis, {
				capabilities,
				options: {
					sampleCount: options.samplePlan.sampleCount,
					enableDeferredLighting: options.enableDeferredLighting,
					forceDeferredFallback: options.forceDeferredFallback,
				},
			});
			publisher.publish(WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE, {
				source: this.id,
				targetClass: resolved.active ? "gbuffer" : "single",
				featureStates: {
					[WEBGPU_FRAME_FEATURE_STATES.deferredSupported]: resolved.supported,
					[WEBGPU_FRAME_FEATURE_STATES.deferredActive]: resolved.active,
					[WEBGPU_FRAME_FEATURE_STATES.deferredGBufferLayout]:
						resolved.deferredGBufferLayout,
				},
				diagnostics: resolved.diagnostics,
			});
		},
	}];
	public readonly executors = {
		"deferred-decal": async (_node: unknown, session: WebGPUFrameSession) => {
			if (this._opaqueState.peek()?.lightingEnabled) {
				await this._decalPass.recordDecalPass(session.context);
			}
		},
		"deferred-lighting": async (_node: unknown, session: WebGPUFrameSession) => {
			await this._recordLighting(session);
		},
	};
	public constructor(
		private readonly _lightingPass: WebGPUDeferredLightingPass,
		private readonly _decalPass: WebGPUDeferredDecalPass,
		private readonly _scene: WebGPUDeferredScenePort,
		private readonly _opaqueState: WebGPUDeferredOpaqueStatePort,
	) {}

	public beginFrame(): void {
		this._opaqueState.clear();
	}

	public activateFrame(context: WebGPUFrameExecutionContext): void {
		this._lightingPass.bindFrame(context);
		this._decalPass.bindFrame(context);
	}

	public closeFrame(): void {
		this._lightingPass.closeFrame();
		this._decalPass.closeFrame();
		this._opaqueState.clear();
	}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		if (
			input.pass.stage !== "main-opaque" ||
			!input.state.deferredActive ||
			input.state.sceneTargetMode !== "gbuffer"
		) return [];
		const nodes: WebGPUFrameGraphNode[] = [];
		if ((input.context.scene?.decalPackets.length ?? 0) > 0) {
			nodes.push(createWebGPUFrameGraphNode(
				input.pass,
				"deferred-decal",
				"WebGPUDeferredDecal",
				this._createDecalResources(input),
			));
		}
		nodes.push(createWebGPUFrameGraphNode(
			input.pass,
			"deferred-lighting",
			"WebGPUDeferredLighting",
			this._createLightingResources(input),
		));
		return nodes.length > 0 ? [{ lane: "lighting", nodes }] : [];
	}

	public invalidateFrameResources(): void {
		this._destroyBindings();
	}

	public onShaderRuntimeChanged(): void {
		this._destroyBindings();
	}

	public destroy(): void {
		this._destroyBindings();
	}

	private async _recordLighting(session: WebGPUFrameSession): Promise<void> {
		const state = this._opaqueState.consume();
		if (!state) return;
		if (state.lightingEnabled) {
			await this._lightingPass.recordLightingPass(
				session.context,
				state.clearSceneColor,
			);
		}
		if (state.fallbackPackets.length > 0) {
			await this._scene.recordMainPass(
				session.context,
				state.fallbackPackets,
				false,
				false,
			);
		}
	}

	private _destroyBindings(): void {
		this._lightingPass.destroyBindings();
		this._decalPass.destroyBindings();
	}

	private _createDecalResources(
		input: WebGPUFrameModulePlanningInput,
	): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		const resourceIds = input.state.deferredGBufferLayout === "base"
			? DEFERRED_GBUFFER_BASE_RESOURCE_IDS
			: DEFERRED_GBUFFER_RESOURCE_IDS;
		return {
			reads: resourceIds.map((id) =>
				readWebGPUFrameGraphResource(id, "copy-src")),
			writes: [
				...DEFERRED_GBUFFER_RENDER_RESOURCE_IDS.map((id) =>
					writeWebGPUFrameGraphResource(id, "render-attachment")),
				...DEFERRED_GBUFFER_STORAGE_RESOURCE_IDS.map((id) =>
					writeWebGPUFrameGraphResource(id, "storage-binding")),
			],
		};
	}

	private _createLightingResources(
		input: WebGPUFrameModulePlanningInput,
	): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		const resourceIds = input.state.deferredGBufferLayout === "base"
			? DEFERRED_GBUFFER_BASE_RESOURCE_IDS
			: DEFERRED_GBUFFER_RESOURCE_IDS;
		return {
			reads: [
				...resourceIds.map((id) =>
					readWebGPUFrameGraphResource(id, "texture-binding")),
				readWebGPUFrameGraphResource("shadow-atlas", "texture-binding", true),
				readWebGPUFrameGraphResource(
					"shadow-transmittance-atlas",
					"texture-binding",
					true,
				),
			],
			writes: [
				writeWebGPUFrameGraphResource(
					"frame:scene-color-main",
					"render-attachment",
				),
			],
		};
	}
}
