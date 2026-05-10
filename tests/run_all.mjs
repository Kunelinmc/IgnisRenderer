import { spawn } from "node:child_process";
import { availableParallelism } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = dirname(__dirname);
const BUN_EXECUTABLE = process.versions?.bun ? process.execPath : "bun";

const tests = [
	"test_lighting.mjs",
	"test_point_spot_lighting.mjs",
	"test_sh_lighting_regression.mjs",
	"test_lighting_shader_semantics.mjs",
	"test_software_shadow_sampling.mjs",
	"test_transparent_shadow_transmission.mjs",
	"test_shadow_strategy_csm.mjs",
	"test_shadow_metadata_stabilization.mjs",
	"test_shadow_manager.mjs",
	"test_backend_capabilities.mjs",
	"test_software_raster_modes.mjs",
	"test_software_early_z_prepass.mjs",
	"test_platform.mjs",
	"test_webgl_backend_stub.mjs",
	"test_webgl_backend_v2.mjs",
	"test_webgl_frame_executor_fxaa.mjs",
	"test_webgl_post_graph.mjs",
	"test_webgl_clustered_lighting_runtime.mjs",
	"test_webgl_sh_texture_upload.mjs",
	"test_webgl_global_uniform_binder_lights.mjs",
	"test_layer_boundaries.mjs",
	"test_frame_planner.mjs",
	"test_incremental_frame_planner.mjs",
	"test_incremental_postfx_grading.mjs",
	"test_dirty_rect_merger.mjs",
	"test_model_factory_winding.mjs",
	"test_render_list_builder.mjs",
	"test_shadow_metadata_bounds.mjs",
	"test_orthographic_camera_bounds.mjs",
	"test_scene_bounds_cache.mjs",
	"test_prepared_scene_cache.mjs",
	"test_prepared_scene_spatial_index_bvh.mjs",
	"test_spatial_hybrid_index.mjs",
	"test_scene_spatial_mode_hybrid.mjs",
	"test_geometry_registry_versioning.mjs",
	"test_csg_core.mjs",
	"test_csg_mesh_instance.mjs",
	"test_renderer_csg_stage.mjs",
	"test_lod_mesh_instance.mjs",
	"test_interaction_manager_selection.mjs",
	"test_interaction_outline_shape.mjs",
	"test_node_clone.mjs",
	"test_node_world_bounds_out.mjs",
	"test_ecs_world.mjs",
	"test_scene_ecs_sync.mjs",
	"test_animation_core.mjs",
	"test_animation_entity_binding.mjs",
	"test_animation_state_blendtree.mjs",
	"test_particle_simulation_stage.mjs",
	"test_physics_stepper.mjs",
	"test_physics_system_bindings.mjs",
	"test_physics_entity_target.mjs",
	"test_physics_system_optimizations.mjs",
	"test_physics_mesh_collision_v2.mjs",
	"test_physics_events.mjs",
	"test_physics_queries.mjs",
	"test_physics_adapter_contract.mjs",
	"test_physics_csg_sync.mjs",
	"test_physics_worker_adapter.mjs",
	"test_rapier_initial_transform_regression.mjs",
	"test_renderer_animation_stage.mjs",
	"test_renderer_particle_stage.mjs",
	"test_renderer_postanimation_hook.mjs",
	"test_renderer_warmup_lightprobe.mjs",
	"test_light_probe_runtime.mjs",
	"test_reflection_probe_capture_runtime.mjs",
	"test_camera_shake_plugin.mjs",
	"test_sobel_normal_mapper.mjs",
	"test_renderer_dynamic_texture_updates.mjs",
	"test_sparse_accessor.mjs",
	"test_gltf_primitive_modes.mjs",
	"test_gltf_material_extensions.mjs",
	"test_gltf_prefab_contract.mjs",
	"test_gltf_loader_security.mjs",
	"test_bvh_loader.mjs",
	"test_exr_loader.mjs",
	"test_pbr_textures.mjs",
	"test_shader_runtime.mjs",
	"test_logger.mjs",
	"test_shader_directive_pipeline_v2.mjs",
	"test_shader_directive_migration_guard.mjs",
	"test_shader_material.mjs",
	"test_canvas_texture.mjs",
	"test_video_texture.mjs",
	"test_texture_loader_cache.mjs",
	"test_webgpu_compute_facade.mjs",
	"test_webgpu_compute_runtime.mjs",
	"test_webgpu_particle_gpu_simulator.mjs",
	"test_webgpu_bridge.mjs",
	"test_webgpu_backend_cache_and_dependency.mjs",
	"test_webgpu_material_binding_cache.mjs",
	"test_webgpu_shadow_atlas_allocator.mjs",
	"test_webgpu_built_in_post_process_passes.mjs",
	"test_webgpu_post_graph.mjs",
	"test_webgpu_frame_executor_resilience.mjs",
	"test_renderer_stage_graph.mjs",
	"test_webgpu_postprocess_math.mjs",
	"test_webgpu_postprocess_runtime_spatial.mjs",
	"test_webgpu_postprocess_runtime_temporal.mjs",
	"test_webgpu_postprocess_runtime_screen.mjs",
	"test_worker_scheduler.mjs",
	"test_worker_transport_plugins.mjs",
	"test_light_probe_baker_async.mjs",
	"test_environment_ibl_update_runtime.mjs",
];

function getDefaultJobs() {
	const detected = (() => {
		try {
			return availableParallelism();
		} catch {
			return 4;
		}
	})();
	return Math.max(1, Math.min(tests.length, detected));
}

function parseOptions(argv) {
	let jobsFromCli = null;
	let failFast = false;

	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg === "--fail-fast") {
			failFast = true;
			continue;
		}
		if (arg === "--serial") {
			jobsFromCli = 1;
			continue;
		}
		if (arg === "-j" || arg === "--jobs") {
			const value = argv[i + 1];
			if (value) {
				jobsFromCli = Number.parseInt(value, 10);
				i++;
			}
			continue;
		}
		if (arg.startsWith("--jobs=")) {
			jobsFromCli = Number.parseInt(arg.slice("--jobs=".length), 10);
		}
	}

	const envJobs = Number.parseInt(process.env.TEST_JOBS ?? "", 10);
	const parsedJobs = Number.isInteger(jobsFromCli) ? jobsFromCli : envJobs;
	const jobs = Number.isInteger(parsedJobs) && parsedJobs > 0
		? parsedJobs
		: getDefaultJobs();

	return {
		jobs: Math.max(1, Math.min(tests.length, jobs)),
		failFast,
	};
}

function formatDuration(durationMs) {
	if (durationMs < 1000) {
		return `${durationMs}ms`;
	}
	return `${(durationMs / 1000).toFixed(2)}s`;
}

function runSingleTest(testName, index, total) {
	return new Promise((resolve) => {
		const startTime = Date.now();
		const absPath = join(__dirname, testName);
		console.log(`[start ${index + 1}/${total}] ${testName}`);

		const child = spawn(BUN_EXECUTABLE, [absPath], {
			cwd: PROJECT_ROOT,
			stdio: ["ignore", "pipe", "pipe"],
			env: process.env,
		});

		let stdout = "";
		let stderr = "";

		child.stdout.on("data", (chunk) => {
			stdout += String(chunk);
		});
		child.stderr.on("data", (chunk) => {
			stderr += String(chunk);
		});

		child.on("error", (error) => {
			resolve({
				testName,
				ok: false,
				code: 1,
				output: `${stdout}${stderr}\n${String(error)}`,
				durationMs: Date.now() - startTime,
			});
		});

		child.on("close", (code, signal) => {
			const finalCode = typeof code === "number" ? code : 1;
			const signalText = signal ? `\nterminated by signal: ${signal}` : "";
			resolve({
				testName,
				ok: finalCode === 0,
				code: finalCode,
				output: `${stdout}${stderr}${signalText}`,
				durationMs: Date.now() - startTime,
			});
		});
	});
}

async function runAll() {
	const options = parseOptions(process.argv.slice(2));
	const { jobs, failFast } = options;
	const total = tests.length;
	const startedAt = Date.now();

	console.log(
		`Running ${total} tests with concurrency=${jobs} (failFast=${failFast})\n`
	);

	const queue = tests.map((testName, index) => ({ testName, index }));
	const failures = [];
	let stopped = false;

	const workers = Array.from({ length: jobs }, async () => {
		while (queue.length > 0 && !stopped) {
			const next = queue.shift();
			if (!next) {
				return;
			}

			const result = await runSingleTest(next.testName, next.index, total);
			const title = `${result.ok ? "PASS" : "FAIL"} ${result.testName}`;
			const duration = formatDuration(result.durationMs);

			console.log("----------------------------------------");
			console.log(`${title} (${duration})`);
			if (result.output.trim().length > 0) {
				console.log(result.output.trimEnd());
			}

			if (!result.ok) {
				failures.push(result);
				if (failFast) {
					stopped = true;
				}
			}
		}
	});

	await Promise.all(workers);

	const totalDuration = formatDuration(Date.now() - startedAt);
	console.log("\n----------------------------------------");
	if (failures.length > 0) {
		console.log(
			`Some tests failed (${failures.length}/${total}) in ${totalDuration}.`
		);
		process.exit(1);
		return;
	}
	console.log(`All tests passed in ${totalDuration}.`);
}

await runAll();
