import { FramePassPlanValidator } from "../../pipeline/FramePassPlanValidator";

export class WebGPUPassPlanner extends FramePassPlanValidator {
	/**
	 * Creates the WebGPU pass validator facade.
	 *
	 * @sideEffects None.
	 */
	public constructor() {
		super("WebGPU");
	}
}
