import {
	resolveFogUniformParams,
	type FogOptions,
} from "../../postprocess/passes/FogPass";

/**
 * Owns the packed scene-fog uniform values shared by scene draws and the
 * particle pass. Values are packed in place; no per-frame allocation occurs.
 */
export class WebGLFogState {
	public readonly params0 = new Float32Array(4);
	public readonly params1 = new Float32Array(4);

	public update(options: FogOptions | undefined, enabled: boolean): void {
		resolveFogUniformParams(options, enabled, this.params0, this.params1);
	}
}
