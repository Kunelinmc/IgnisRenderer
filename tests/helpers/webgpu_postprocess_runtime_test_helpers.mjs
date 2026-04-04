import assert from "node:assert/strict";

export class FakeBackend {
	constructor() {
		this.samplers = [];
		this.shaderModules = [];
		this.computePipelines = [];
		this.buffers = [];
		this.bindingGroups = [];
		this.bindGroupLayouts = [];
		this.pipelineLayouts = [];
		this.textures = [];
		this.textureViews = [];
		this.bindingGroupDestroyCalls = 0;
		this.bufferDestroyCalls = 0;
		this.textureDestroyCalls = 0;
		this.writeBufferCalls = 0;
	}

	createSampler(desc) {
		const sampler = { label: desc.label, desc };
		this.samplers.push(sampler);
		return sampler;
	}

	async createShaderModule(desc) {
		const module = { label: desc.label, desc };
		this.shaderModules.push(module);
		return module;
	}

	createComputePipeline(desc) {
		const pipeline = { label: desc.label, desc };
		this.computePipelines.push(pipeline);
		return pipeline;
	}

	createBuffer(desc) {
		const buffer = {
			size: desc.size,
			desc,
			destroyed: false,
			destroy: () => {
				if (buffer.destroyed) return;
				buffer.destroyed = true;
				this.bufferDestroyCalls++;
			},
		};
		this.buffers.push(buffer);
		return buffer;
	}

	createTexture(desc) {
		const texture = {
			width: desc.width,
			height: desc.height,
			label: desc.label,
			desc,
			destroyed: false,
			destroy: () => {
				if (texture.destroyed) return;
				texture.destroyed = true;
				this.textureDestroyCalls++;
			},
		};
		this.textures.push(texture);
		return texture;
	}

	writeBuffer(buffer, data) {
		this.writeBufferCalls++;
		buffer.lastWrite = Array.from(data);
	}

	createBindingGroup(desc) {
		const bindingGroup = {
			label: desc.label,
			desc,
			destroyed: false,
			destroy: () => {
				if (bindingGroup.destroyed) return;
				bindingGroup.destroyed = true;
				this.bindingGroupDestroyCalls++;
			},
		};
		this.bindingGroups.push(bindingGroup);
		return bindingGroup;
	}

	createBindGroupLayout(desc) {
		const layout = { label: desc.label, desc };
		this.bindGroupLayouts.push(layout);
		return layout;
	}

	createPipelineLayout(desc) {
		const layout = { label: desc.label, desc };
		this.pipelineLayouts.push(layout);
		return layout;
	}

	createTextureView(texture, desc = {}) {
		const view = { texture, desc };
		this.textureViews.push(view);
		return view;
	}
}

export class FakeEncoder {
	constructor() {
		this.calls = [];
	}

	beginComputePass(desc = {}) {
		this.calls.push(["beginComputePass", desc.label ?? null]);
	}

	setComputePipeline(pipeline) {
		this.calls.push(["setComputePipeline", pipeline.label]);
	}

	setBindingGroup(index, group) {
		this.calls.push(["setBindingGroup", index, group.label]);
	}

	dispatchWorkgroups(x, y = 1, z = 1) {
		this.calls.push(["dispatchWorkgroups", x, y, z]);
	}

	endComputePass() {
		this.calls.push(["endComputePass"]);
	}
}

export function createTexture(width, height, label) {
	return {
		width,
		height,
		label,
		destroy() {},
	};
}

export function assertClose(actual, expected, epsilon = 1e-6) {
	assert.ok(Math.abs(actual - expected) <= epsilon);
}
