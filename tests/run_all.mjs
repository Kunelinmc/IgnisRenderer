import { spawnSync } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

const tests = [
	"test_lighting.mjs",
	"test_point_spot_lighting.mjs",
	"test_sh_lighting_regression.mjs",
	"test_lighting_shader_semantics.mjs",
	"test_software_shadow_sampling.mjs",
	"test_backend_capabilities.mjs",
	"test_software_raster_modes.mjs",
	"test_platform.mjs",
	"test_webgl_backend_stub.mjs",
	"test_webgl_backend_v1.mjs",
	"test_webgl_frame_executor_fxaa.mjs",
	"test_layer_boundaries.mjs",
	"test_frame_planner.mjs",
	"test_model_factory_winding.mjs",
	"test_render_list_builder.mjs",
	"test_shadow_metadata_bounds.mjs",
	"test_orthographic_camera_bounds.mjs",
	"test_scene_bounds_cache.mjs",
	"test_interaction_manager_selection.mjs",
	"test_interaction_outline_shape.mjs",
	"test_node_clone.mjs",
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
	"test_physics_events.mjs",
	"test_physics_queries.mjs",
	"test_physics_adapter_contract.mjs",
	"test_physics_worker_adapter.mjs",
	"test_rapier_initial_transform_regression.mjs",
	"test_renderer_animation_stage.mjs",
	"test_renderer_particle_stage.mjs",
	"test_renderer_postanimation_hook.mjs",
	"test_renderer_warmup_lightprobe.mjs",
	"test_camera_shake_plugin.mjs",
	"test_sobel_normal_mapper.mjs",
	"test_renderer_dynamic_texture_updates.mjs",
	"test_sparse_accessor.mjs",
	"test_gltf_primitive_modes.mjs",
	"test_gltf_material_extensions.mjs",
	"test_gltf_prefab_contract.mjs",
	"test_gltf_loader_security.mjs",
	"test_pbr_textures.mjs",
	"test_shader_runtime.mjs",
	"test_shader_directive_pipeline_v2.mjs",
	"test_shader_directive_migration_guard.mjs",
	"test_shader_material.mjs",
	"test_canvas_texture.mjs",
	"test_video_texture.mjs",
	"test_texture_loader_cache.mjs",
	"test_webgpu_compute_facade.mjs",
	"test_webgpu_bridge.mjs",
	"test_webgpu_backend_cache_and_dependency.mjs",
	"test_webgpu_material_binding_cache.mjs",
	"test_webgpu_post_graph.mjs",
	"test_webgpu_frame_executor_resilience.mjs",
	"test_renderer_stage_graph.mjs",
	"test_webgpu_postprocess_math.mjs",
	"test_webgpu_postprocess_runtime.mjs",
	"test_worker_scheduler.mjs",
	"test_worker_transport_plugins.mjs",
	"test_light_probe_baker_async.mjs",
];

let failed = false;

console.log("🚀 Running all tests...\n");

for (const test of tests) {
	console.log(`----------------------------------------`);
	console.log(`Running ${test}...`);
	const result = spawnSync("npx", ["tsx", join(__dirname, test)], {
		stdio: "inherit",
		shell: true,
	});

	if (result.status !== 0) {
		console.error(`❌ ${test} FAILED`);
		failed = true;
	} else {
		console.log(`✅ ${test} PASSED`);
	}
}

console.log(`\n----------------------------------------`);
if (failed) {
	console.log("❌ Some tests failed!");
	process.exit(1);
} else {
	console.log("✨ All tests passed!");
}
