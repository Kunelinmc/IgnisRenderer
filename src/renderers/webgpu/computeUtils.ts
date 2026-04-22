import type { ICommandEncoder } from "../ICommandEncoder";
import type { IBindingGroup, IComputePipeline } from "../types";

export interface ComputePassBinding {
	index: number;
	group: IBindingGroup;
}

export interface ComputePassDispatch {
	x: number;
	y: number;
	z: number;
}

export function recordComputePass(
	encoder: ICommandEncoder,
	label: string,
	pipeline: IComputePipeline,
	bindings: ComputePassBinding[],
	dispatch: ComputePassDispatch
): void {
	const x = assertPositiveInteger(dispatch.x, "dispatch.x");
	const y = assertPositiveInteger(dispatch.y, "dispatch.y");
	const z = assertPositiveInteger(dispatch.z, "dispatch.z");

	encoder.beginComputePass({ label });
	encoder.setComputePipeline(pipeline);
	for (const binding of bindings) {
		encoder.setBindingGroup(binding.index, binding.group);
	}
	encoder.dispatchWorkgroups(x, y, z);
	encoder.endComputePass();
}

function assertPositiveInteger(value: number, name: string): number {
	if (!Number.isInteger(value) || value <= 0) {
		throw new Error(
			`recordComputePass() requires ${name} to be a positive integer, received ${value}.`
		);
	}
	return value;
}
