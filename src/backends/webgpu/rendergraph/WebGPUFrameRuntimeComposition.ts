import type { FramePacketProvider } from "../../../pipeline/FramePacketContributorRegistry";
import type { WebGPUFrameServiceOwner } from "../WebGPUFrameServiceOwner";
import { WebGPUHiZBuilder } from "../WebGPUHiZBuilder";
import { WebGPUOcclusionCullingRuntime } from "../WebGPUOcclusionCullingRuntime";
import { WebGPUPlanarReflectionPass } from "../WebGPUPlanarReflectionPass";
import { WebGPUPostProcessRuntime } from "../WebGPUPostProcessRuntime";
import type {
	WebGPUParticleBillboardRenderer,
	WebGPUPreparedFrameResources,
} from "../WebGPUResourceContracts";
import type { WebGPUSampleCountResolver } from "../WebGPUSampleCountResolver";
import { WebGPUCustomRenderTargetRuntime } from "./WebGPUCustomRenderTargetRuntime";
import { WebGPUFrameConfigurationModule } from "./WebGPUFrameConfigurationModule";
import { WebGPUDeferredDecalPass } from "./WebGPUDeferredDecalPass";
import { WebGPUDeferredFrameModule } from "./WebGPUDeferredFrameModule";
import { WebGPUDeferredLightingPass } from "./WebGPUDeferredLightingPass";
import { WebGPUDepthDirtyClearPass } from "./WebGPUDepthDirtyClearPass";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";
import { WebGPUFrameGraphModuleRegistry } from "./WebGPUFrameGraphModuleRegistry";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import { WebGPUPostProcessBridge } from "./WebGPUPostProcessBridge";
import { WebGPUPostProcessFrameModule } from "./WebGPUPostProcessFrameModule";
import { WebGPUPresentationRuntime } from "./WebGPUPresentationRuntime";
import { WebGPUReflectionFrameModule } from "./WebGPUReflectionFrameModule";
import { WebGPUSceneFrameModule } from "./WebGPUSceneFrameModule";
import { WebGPUScenePassRecorder } from "./WebGPUScenePassRecorder";
import { WebGPUShadowFrameModule } from "./WebGPUShadowFrameModule";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import { WebGPUTransparencyRuntime } from "./WebGPUTransparencyRuntime";
import { WebGPUVisibilityFrameModule } from "./WebGPUVisibilityFrameModule";

/** @internal Narrow orchestrator capabilities supplied to the composition root. */
export interface WebGPUFrameRuntimeCompositionAccess {
	readonly recording: WebGPUFrameGraphRecordingContext;
	getSession(): WebGPUFrameSession | null;
	requireSession(): WebGPUFrameSession;
	requireFrameResources(): WebGPUPreparedFrameResources;
	warnOnce(code: string, message: string, cause?: unknown): void;
}

/** @internal Sealed backend-private frame runtime composition. */
export interface WebGPUFrameRuntimeComposition {
	readonly modules: WebGPUFrameGraphModuleRegistry;
	readonly postProcess: Pick<
		WebGPUPostProcessFrameModule,
		"describeFrame" | "buildGraphFrame" | "executeStage" | "getDebugState" |
		"warmup" | "invalidateFrameResources" | "createSessionPort"
	>;
	readonly visibility: Pick<
		WebGPUVisibilityFrameModule,
		"reset" | "getVisibilityProvider"
	>;
	readonly customRenderTargets: Pick<
		WebGPUCustomRenderTargetRuntime,
		"sync" | "readColor"
	>;
}

/** @internal Feature ports consumed by backend lifecycle integration. */
export type WebGPUFrameRuntimeCapabilities = Pick<
	WebGPUFrameRuntimeComposition,
	"postProcess" | "visibility" | "customRenderTargets"
>;

/** @internal Creates one sealed frame runtime composition per backend. */
export type WebGPUFrameRuntimeCompositionFactory = (
	access: WebGPUFrameRuntimeCompositionAccess,
) => WebGPUFrameRuntimeComposition;

/** Creates the backend-owned initialization-time frame-module factory. */
export function createWebGPUFrameRuntimeCompositionFactory(options: {
	readonly host: WebGPUFrameHost;
	readonly frameServices: WebGPUFrameServiceOwner;
	readonly framePackets: FramePacketProvider;
	readonly particleRenderer: WebGPUParticleBillboardRenderer;
	readonly sampleCountResolver: WebGPUSampleCountResolver;
}): WebGPUFrameRuntimeCompositionFactory {
	return (access) => {
		const { host, frameServices, framePackets, particleRenderer } = options;
		const hiZBuilder = new WebGPUHiZBuilder(host.computeFacade);
		const postRuntime = new WebGPUPostProcessRuntime(
			host.computeFacade,
			(key, message) => access.warnOnce(key, message),
			frameServices.sceneFrameLayout,
			hiZBuilder,
			() => host.displayOutputState,
		);
		const planarReflectionPass = new WebGPUPlanarReflectionPass(
			host,
			frameServices,
			framePackets,
		);
		const customRenderTargets = new WebGPUCustomRenderTargetRuntime(
			host,
			options.sampleCountResolver,
		);
		const modules = new WebGPUFrameGraphModuleRegistry();
		const presentation = new WebGPUPresentationRuntime(host, {
			recording: access.recording,
			getOutputColorDomain: () => modules.finalOutput.colorDomain,
		});
		const postBridge = new WebGPUPostProcessBridge(host, postRuntime, {
			getEncoder: () => access.recording.getEncoder(),
			getFrameTargets: () => access.recording.getFrameTargets(),
			isHiZReady: () => access.getSession()?.hiZStatus === "ready",
			requireFrameResources: () => access.requireFrameResources(),
			presentToCanvas: (source) => presentation.present(source, access.requireSession()),
			warmupPresent: () => presentation.warmup(),
			setMotionHistoryWriteTarget: (texture) => {
				const session = access.getSession();
				if (session) session.motionHistoryWriteTarget = texture;
			},
		});
		const postProcess = new WebGPUPostProcessFrameModule(
			postRuntime,
			host.postProcessRuntime,
			postBridge,
			presentation,
			access.recording,
		);
		const depthDirtyClearPass = new WebGPUDepthDirtyClearPass(host);
		const deferredLightingPass = new WebGPUDeferredLightingPass(host, frameServices, {
			recordingContext: access.recording,
		});
		const deferredDecalPass = new WebGPUDeferredDecalPass(host, frameServices, {
			recordingContext: access.recording,
		});
		const sceneRecorder = new WebGPUScenePassRecorder(
			host,
			frameServices,
			particleRenderer,
			access.recording,
			depthDirtyClearPass,
			{
				getGBufferWriteBinding: () => deferredLightingPass.getGBufferWriteBinding(),
				getDeferredGBufferLayout: () =>
					access.getSession()?.configuration?.deferredGBufferLayout ?? "extended",
				preflightDeferredFrame: async (context) => {
					await deferredLightingPass.preflight();
					await deferredDecalPass.preflight(context);
				},
			},
		);
		const reflection = new WebGPUReflectionFrameModule(
			host,
			planarReflectionPass,
			access.recording,
		);
		const visibility = new WebGPUVisibilityFrameModule(
			hiZBuilder,
			new WebGPUOcclusionCullingRuntime(host),
			access.recording,
		);
		const configuration = new WebGPUFrameConfigurationModule();
		for (const module of [
			new WebGPUSceneFrameModule(sceneRecorder, depthDirtyClearPass),
			new WebGPUShadowFrameModule(frameServices, access.recording),
			new WebGPUDeferredFrameModule(
				deferredLightingPass,
				deferredDecalPass,
				sceneRecorder,
			),
			new WebGPUTransparencyRuntime(
				host,
				frameServices,
				particleRenderer,
				access.recording,
				sceneRecorder,
				{ warnOnce: access.warnOnce },
			),
			reflection,
			visibility,
			postProcess,
			presentation,
			customRenderTargets,
		]) {
			modules.register(module);
		}
		for (const handler of configuration.messageHandlers) {
			modules.registerMessageHandler(handler);
		}
		modules.seal();
		return {
			modules,
			postProcess,
			visibility,
			customRenderTargets,
		};
	};
}
