import { ShaderMaterial } from "../materials/ShaderMaterial";
import type { Material } from "../materials/Material";
import { defineTransientKey, type FrameContext } from "./types";
import {
	getEnabledCustomPostProcessPassIds,
	isFogPostProcessEnabled,
} from "./PostProcessController";
import { ShaderCompileError } from "../shaders/runtime";
import type { ShaderCompilerBackend } from "../shaders/runtime/errorMapping";
import type {
	RenderBackendType,
	WarmupOptions,
	WarmupPhaseReport,
	WarmupReport,
} from "../renderers/IRenderBackend";
import type { PostProcessPass } from "../postprocess/PostProcessPass";

export type WarmupSceneTargetMode = "single" | "mrt";

export interface WarmupPlan {
	materials: Material[];
	shaderMaterials: ShaderMaterial[];
	enableEnvironment: boolean;
	enableShadows: boolean;
	enableParticles: boolean;
	includePostProcess: boolean;
	postProcessPasses: string[];
	sceneTargetMode: WarmupSceneTargetMode;
}

export const WARMUP_POST_PROCESS_ORDER_TRANSIENT_KEY =
	"pipeline:warmup-postprocess-order";

export const WARMUP_POST_PROCESS_DESCRIPTORS_TRANSIENT_KEY =
	defineTransientKey<readonly PostProcessPass[]>(
		"pipeline:warmup-postprocess-descriptors"
	);

export interface WarmupPhaseCounters extends WarmupPhaseReport {
	errors: ShaderCompileError[];
}

export function buildWarmupPlan(
	context: FrameContext,
	options: WarmupOptions = {}
): WarmupPlan {
	const includeCore = options.includeCorePasses !== false;
	const includeShadow = options.includeShadowPass !== false;
	const includeParticles = options.includeParticles !== false;
	const includePost = options.includePostProcess !== false;

	const materials = collectUniqueMaterials(context);
	const shaderMaterials = materials.filter(
		(material): material is ShaderMaterial => material instanceof ShaderMaterial
	);
	const postProcessPasses =
		includePost ? resolveEnabledPostProcessPasses(context) : [];
	const useMRT = postProcessPasses.length > 0;

	return {
		materials: includeCore ? materials : [],
		shaderMaterials: includeCore ? shaderMaterials : [],
		enableEnvironment:
			includeCore &&
			context.features.enableEnvironment &&
			context.scene.environment.backgroundEnabled &&
			!!context.scene.environment.backgroundTexture,
		enableShadows:
			includeShadow &&
			context.features.enableShadows &&
			context.scene.shadowCasterPackets.length > 0,
		enableParticles:
			includeParticles &&
			(context.scene.particleSystems?.length ?? 0) > 0,
		includePostProcess: includePost,
		postProcessPasses,
		sceneTargetMode: useMRT ? "mrt" : "single",
	};
}

export function createWarmupReport(backend: RenderBackendType): WarmupReport {
	const now = Date.now();
	return {
		backend,
		startedAt: now,
		finishedAt: now,
		durationMs: 0,
		total: 0,
		compiled: 0,
		skipped: 0,
		failed: 0,
		phases: [],
		errors: [],
	};
}

export function addWarmupPhase(
	report: WarmupReport,
	phase: WarmupPhaseCounters
): void {
	report.total += phase.total;
	report.compiled += phase.compiled;
	report.skipped += phase.skipped;
	report.failed += phase.failed;
	if (phase.errors.length > 0) {
		report.errors.push(...phase.errors);
	}
	report.phases.push({
		phase: phase.phase,
		total: phase.total,
		compiled: phase.compiled,
		skipped: phase.skipped,
		failed: phase.failed,
	});
}

export function finalizeWarmupReport(report: WarmupReport): WarmupReport {
	report.finishedAt = Date.now();
	report.durationMs = Math.max(0, report.finishedAt - report.startedAt);
	return report;
}

export function toShaderCompileError(
	error: unknown,
	backend: RenderBackendType,
	label: string
): ShaderCompileError {
	if (error instanceof ShaderCompileError) {
		return error;
	}
	const compilerBackend: ShaderCompilerBackend =
		(backend === "webgpu" || backend === "webgl" ? backend : "unknown") as
			ShaderCompilerBackend;
	return new ShaderCompileError({
		backend: compilerBackend,
		language: "wgsl",
		stage: "unknown",
		label,
		sourceKind: "unknown",
		code: "",
		sourceMap: null,
		messages: [
			{
				type: "error",
				message: String(error),
			},
		],
		cause: error,
	});
}

function collectUniqueMaterials(context: FrameContext): Material[] {
	const unique = new Set<Material>();
	const packets = [
		...context.scene.opaquePackets,
		...context.scene.transparentPackets,
		...context.scene.shadowCasterPackets,
		...context.scene.reflectivePackets,
	];
	for (const packet of packets) {
		if (packet.material) {
			unique.add(packet.material);
		}
	}
	return Array.from(unique);
}

function resolveEnabledPostProcessPasses(context: FrameContext): string[] {
	const orderedPasses = context.transient.get(
		WARMUP_POST_PROCESS_ORDER_TRANSIENT_KEY
	);
	if (
		Array.isArray(orderedPasses) &&
		orderedPasses.every((passId) => typeof passId === "string")
	) {
		return orderedPasses.slice();
	}

	const passes: string[] = [];
	const postProcess = context.postProcess;
	if (postProcess.isEnabled("ssao")) passes.push("ssao");
	if (postProcess.isEnabled("ssgi")) passes.push("ssgi");
	if (postProcess.isEnabled("taa")) passes.push("taa");
	if (postProcess.isEnabled("ssr")) passes.push("ssr");
	if (postProcess.isEnabled("volumetric")) passes.push("volumetric");
	if (isFogPostProcessEnabled(postProcess)) passes.push("fog");
	if (postProcess.isEnabled("motion-blur")) passes.push("motion-blur");
	if (postProcess.isEnabled("dof")) passes.push("dof");
	if (postProcess.isEnabled("bloom")) passes.push("bloom");
	if (postProcess.isEnabled("tonemap")) {
		passes.push("tonemap");
	}
	if (postProcess.isEnabled("color-filter")) passes.push("color-filter");
	if (postProcess.isEnabled("fxaa")) passes.push("fxaa");
	for (const id of getEnabledCustomPostProcessPassIds(postProcess)) {
		passes.push(id);
	}
	if (postProcess.isEnabled("gamma")) {
		passes.push("gamma");
	}
	return passes;
}
