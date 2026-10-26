import type { WebGPUFrameServiceOwner } from "../WebGPUFrameServiceOwner";
import { WebGPUHiZBuilder } from "../WebGPUHiZBuilder";
import { WebGPUOcclusionCullingRuntime } from "../WebGPUOcclusionCullingRuntime";
import { WebGPUPlanarReflectionPass } from "../WebGPUPlanarReflectionPass";
import { WebGPUPostProcessRuntime } from "../WebGPUPostProcessRuntime";
import type { WebGPUSampleCountResolver } from "../WebGPUSampleCountResolver";
import { WebGPUCustomRenderTargetRuntime } from "./WebGPUCustomRenderTargetRuntime";
import { WebGPUFrameConfigurationModule } from "./WebGPUFrameConfigurationModule";
import { WebGPUDeferredDecalPass } from "./WebGPUDeferredDecalPass";
import {
	WebGPUDeferredFrameModule,
	WebGPUDeferredOpaqueStatePort,
} from "./WebGPUDeferredFrameModule";
import { WebGPUDeferredLightingPass } from "./WebGPUDeferredLightingPass";
import { WebGPUDepthDirtyClearPass } from "./WebGPUDepthDirtyClearPass";
import { WebGPUColorDirtyClearPass } from "./WebGPUColorDirtyClearPass";
import { WebGPUFrameGraphModuleRegistry } from "./WebGPUFrameGraphModuleRegistry";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import { WebGPUPostProcessBridge } from "./WebGPUPostProcessBridge";
import { WebGPUPostProcessFrameModule } from "./WebGPUPostProcessFrameModule";
import { WebGPUPresentationRuntime } from "./WebGPUPresentationRuntime";
import { WebGPUReflectionFrameModule } from "./WebGPUReflectionFrameModule";
import { WebGPUSceneFrameModule } from "./WebGPUSceneFrameModule";
import { WebGPUScenePassRecorder } from "./WebGPUScenePassRecorder";
import { WebGPUShadowFrameModule } from "./WebGPUShadowFrameModule";
import { WebGPUTransparencyRuntime } from "./WebGPUTransparencyRuntime";
import { WebGPUVisibilityFrameModule } from "./WebGPUVisibilityFrameModule";
import type { WebGPUShadowRenderProvider } from "../WebGPUResourceContracts";
import type { FrameContext } from "../../../pipeline/types";
import type { WarmupOptions, WarmupReport } from "../../IRenderBackend";
import {
	addWarmupPhase,
	buildWarmupPlan,
	createWarmupReport,
	finalizeWarmupReport,
	toShaderCompileError,
	type WarmupPhaseCounters,
} from "../../../pipeline/WarmupPlanner";
import { createBaselineFramePacketSet } from "../../../pipeline/FramePackets";
import { createWarmupYieldController } from "../../../pipeline/WarmupScheduler";

/** @internal Sealed backend-private frame runtime composition. */
export interface WebGPUFrameRuntimeComposition {
	readonly modules: WebGPUFrameGraphModuleRegistry;
	readonly lifecycle: Pick<
		WebGPUFrameGraphModuleRegistry,
		"invalidateFrameResources" | "onDisplayOutputChanged" |
		"onShaderRuntimeChanged" | "destroy"
	>;
	readonly postProcess: Pick<
		WebGPUPostProcessFrameModule,
		"getDebugState" | "invalidateFrameResources" |
		"createSessionPort" | "createGBufferBridge" |
		"createPassExecutionContext" | "completePass"
	>;
	readonly visibility: Pick<
		WebGPUVisibilityFrameModule,
		"reset" | "getVisibilityProvider"
	>;
	readonly renderTargets: Pick<
		WebGPUCustomRenderTargetRuntime,
		"readColor"
	>;
	/**
	 * Warms this device session's feature pipelines and reports completed phases.
	 * Planning failures reject; compilation failures are collected in the report.
	 *
	 * @internal WebGPU runtime composition; applications use Renderer.warmup().
	 * @param context Prepared scene and post-process inputs.
	 * @param options Scheduling, diagnostics, and progress options.
	 * @param sampleCount Requested main-scene sample count.
	 * @returns Aggregated feature phase reports without executing a frame.
	 */
	warmup(
		context: FrameContext,
		options: WarmupOptions,
		sampleCount: number,
	): Promise<WarmupReport>;
}

/** Creates one backend-owned, sealed frame runtime composition. */
export function createWebGPUFrameRuntimeComposition(options: {
	readonly host: WebGPUFrameHost;
	readonly frameServices: WebGPUFrameServiceOwner;
	readonly shadowRenderer: WebGPUShadowRenderProvider;
	readonly sampleCountResolver: WebGPUSampleCountResolver;
	warnOnce(code: string, message: string, cause?: unknown): void;
}): WebGPUFrameRuntimeComposition {
	const { host, frameServices } = options;
	const particleRenderer = frameServices.getParticleBillboardRenderer();
	const hiZBuilder = new WebGPUHiZBuilder(host.computeFacade);
	const postRuntime = new WebGPUPostProcessRuntime(
		host.computeFacade,
		(key, message) => options.warnOnce(key, message),
		frameServices.sceneFrameLayout,
		hiZBuilder,
		() => host.displayOutputState,
	);
	const planarReflectionPass = new WebGPUPlanarReflectionPass(
		host,
		frameServices,
		frameServices.createPlanarReflectionDrawResources(),
	);
	const renderTargets = new WebGPUCustomRenderTargetRuntime(
		host,
		options.sampleCountResolver,
		frameServices,
	);
	const modules = new WebGPUFrameGraphModuleRegistry();
	const presentation = new WebGPUPresentationRuntime(host, {
		getOutputColorDomain: () => modules.finalOutput.colorDomain,
	});
	const visibility = new WebGPUVisibilityFrameModule(
		hiZBuilder,
		new WebGPUOcclusionCullingRuntime(host),
	);
	const postBridge = new WebGPUPostProcessBridge(postRuntime, {
		isHiZReady: () => visibility.isHiZReady(),
	});
	const postProcessRuntime = host.postProcessRuntime;
	const postProcess = new WebGPUPostProcessFrameModule(
		postRuntime,
		postProcessRuntime,
		postBridge,
		presentation,
	);
	const depthDirtyClearPass = new WebGPUDepthDirtyClearPass(host);
	const colorDirtyClearPass = new WebGPUColorDirtyClearPass(host);
	const deferredOpaqueState = new WebGPUDeferredOpaqueStatePort();
	const deferredLightingPass = new WebGPUDeferredLightingPass(host, frameServices);
	const deferredDecalPass = new WebGPUDeferredDecalPass(host, frameServices);
	const sceneRecorder = new WebGPUScenePassRecorder(
		host,
		frameServices,
		particleRenderer,
		depthDirtyClearPass,
		colorDirtyClearPass,
		{
			preflightDeferredFrame: async (context) => {
				await deferredLightingPass.preflight();
				await deferredDecalPass.preflight(context);
			},
		},
	);
	const reflection = new WebGPUReflectionFrameModule(
		host,
		planarReflectionPass,
	);
	const configuration = new WebGPUFrameConfigurationModule();
	for (const module of [
		new WebGPUSceneFrameModule(
			sceneRecorder,
			depthDirtyClearPass,
			colorDirtyClearPass,
			deferredOpaqueState,
		),
		new WebGPUShadowFrameModule(options.shadowRenderer),
		new WebGPUDeferredFrameModule(
			deferredLightingPass,
			deferredDecalPass,
			sceneRecorder,
			deferredOpaqueState,
		),
		new WebGPUTransparencyRuntime(
			host,
			frameServices,
			particleRenderer,
			sceneRecorder,
			{ warnOnce: options.warnOnce },
		),
		reflection,
		visibility,
		postProcess,
		presentation,
		renderTargets,
	]) {
		modules.register(module);
	}
	for (const handler of configuration.messageHandlers) {
		modules.registerMessageHandler(handler);
	}
	modules.seal();
	return {
		modules,
		lifecycle: modules,
		postProcess,
		visibility,
		renderTargets,
		async warmup(context, warmupOptions, sampleCount) {
			const report = createWarmupReport("webgpu");
			const postProcessPlan = postProcessRuntime.planWarmup(context);
			const plan = buildWarmupPlan(context, warmupOptions, {
				passIds: postProcessPlan.orderedPasses.map((pass) => pass.id),
				descriptors: postProcessPlan.orderedPasses.map((pass) => pass.pass),
			});
			const framePackets = createBaselineFramePacketSet(context);
			const reportPhase = (phase: WarmupPhaseCounters): void => {
				addWarmupPhase(report, phase);
				warmupOptions.onProgress?.({
					phase: phase.phase,
					completed: phase.compiled + phase.skipped + phase.failed,
					total: phase.total,
				});
			};
			try {
				reportPhase(await postProcess.warmup(
					context,
					plan,
					warmupOptions,
					postProcessPlan,
				));
				const resourcePhases = await frameServices.warmup(
					context,
					plan,
					warmupOptions,
					framePackets,
					{
						enableEarlyZPrepass: host.enableEarlyZPrepass,
						enableDeferredLighting: host.enableDeferredLighting,
						sampleCount,
					},
				);
				for (const phase of resourcePhases) reportPhase(phase);
				const reflectionYield = createWarmupYieldController(warmupOptions);
				reportPhase(await reflection.warmup(
					context,
					framePackets,
					sampleCount,
					() => reflectionYield.yieldIfNeeded(),
				));
			} catch (error) {
				reportPhase({
					phase: "webgpu-warmup",
					total: 1,
					compiled: 0,
					skipped: 0,
					failed: 1,
					errors: [toShaderCompileError(error, "webgpu", "WebGPUWarmup")],
				});
			}
			return finalizeWarmupReport(report);
		},
	};
}
