import { ShaderMaterial } from "../materials/ShaderMaterial";
import type { Material } from "../materials/Material";
import type { FrameContext } from "./types";
import { ShaderCompileError } from "../shaders/runtime";
import type { ShaderCompilerBackend } from "../shaders/runtime/errorMapping";
import type {
	RenderBackendType,
	WarmupOptions,
	WarmupPhaseReport,
	WarmupReport,
} from "../backends/IRenderBackend";
import type { PostProcessPass } from "../postprocess/PostProcessPass";
import { resolvePostProcessExecutionOrder } from "../postprocess/PostProcessPlanner";

export type WarmupSceneTargetMode = "single" | "mrt";

export interface WarmupPlan {
	materials: Material[];
	shaderMaterials: ShaderMaterial[];
	enableEnvironment: boolean;
	enableShadows: boolean;
	enableParticles: boolean;
	postProcessPasses: string[];
	postProcessDescriptors: readonly PostProcessPass[];
	sceneTargetMode: WarmupSceneTargetMode;
}

export interface WarmupPhaseCounters extends WarmupPhaseReport {
	errors: ShaderCompileError[];
}

export interface WarmupPostProcessPlan {
	readonly passIds: readonly string[];
	readonly descriptors: readonly PostProcessPass[];
}

export function buildWarmupPlan(
	context: FrameContext,
	options: WarmupOptions = {},
	postProcessPlan?: WarmupPostProcessPlan
): WarmupPlan {
	const includeParticles = options.includeParticles !== false;

	const materials = collectUniqueMaterials(context);
	const shaderMaterials = materials.filter(
		(material): material is ShaderMaterial => material instanceof ShaderMaterial
	);
	const postProcessPasses = resolveEnabledPostProcessPasses(context, postProcessPlan);
	const postProcessDescriptors = resolvePostProcessDescriptors(context, postProcessPlan);
	const useMRT = postProcessPasses.length > 0;

	return {
		materials,
		shaderMaterials,
		enableEnvironment:
			context.features.enableEnvironment &&
			context.scene.environment.backgroundEnabled &&
			!!context.scene.environment.backgroundTexture,
		enableShadows:
			context.features.enableShadows &&
			context.scene.shadowCasterPackets.length > 0,
		enableParticles:
			includeParticles &&
			(context.scene.particleSystems?.length ?? 0) > 0,
		postProcessPasses,
		postProcessDescriptors,
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
		if (packet.submission.material.effective) {
			unique.add(packet.submission.material.effective);
		}
	}
	return Array.from(unique);
}

function resolveEnabledPostProcessPasses(
	context: FrameContext,
	postProcessPlan?: WarmupPostProcessPlan
): string[] {
	if (postProcessPlan) {
		return postProcessPlan.passIds.slice();
	}

	return resolvePostProcessExecutionOrder(context.postProcess).map(
		(pass) => pass.id
	);
}

function resolvePostProcessDescriptors(
	context: FrameContext,
	postProcessPlan?: WarmupPostProcessPlan
): readonly PostProcessPass[] {
	if (postProcessPlan) {
		return postProcessPlan.descriptors.slice();
	}
	return context.postProcess.getEnabledPasses().map((pass) => pass.pass);
}
