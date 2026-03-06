export interface WebGPUPipelineLayouts {
	sceneFrameBindGroupLayout: GPUBindGroupLayout;
	skyboxFrameBindGroupLayout: GPUBindGroupLayout;
	modelBindGroupLayout: GPUBindGroupLayout;
	scenePipelineLayout: GPUPipelineLayout;
	skyboxPipelineLayout: GPUPipelineLayout;
}

export function createWebGPUPipelineLayouts(
	device: GPUDevice
): WebGPUPipelineLayouts {
	const sceneFrameBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUSceneFrameBindGroupLayout",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: { type: "uniform" },
			},
			{
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "depth" },
			},
			{
				binding: 2,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "float" },
			},
			{
				binding: 3,
				visibility: GPUShaderStage.FRAGMENT,
				sampler: { type: "filtering" },
			},
		],
	});
	const skyboxFrameBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUSkyboxFrameBindGroupLayout",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: { type: "uniform" },
			},
			{
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "float" },
			},
			{
				binding: 2,
				visibility: GPUShaderStage.FRAGMENT,
				sampler: { type: "filtering" },
			},
		],
	});

	const modelEntries: GPUBindGroupLayoutEntry[] = [
		{
			binding: 0,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
			buffer: { type: "uniform" },
		},
	];

	for (let i = 0; i < 14; i++) {
		modelEntries.push({
			binding: 1 + i * 2,
			visibility: GPUShaderStage.FRAGMENT,
			texture: { sampleType: "float" },
		});
		modelEntries.push({
			binding: 2 + i * 2,
			visibility: GPUShaderStage.FRAGMENT,
			sampler: { type: "filtering" },
		});
	}

	const modelBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUModelBindGroupLayout",
		entries: modelEntries,
	});

	const scenePipelineLayout = device.createPipelineLayout({
		label: "WebGPUScenePipelineLayout",
		bindGroupLayouts: [sceneFrameBindGroupLayout, modelBindGroupLayout],
	});
	const skyboxPipelineLayout = device.createPipelineLayout({
		label: "WebGPUSkyboxPipelineLayout",
		bindGroupLayouts: [skyboxFrameBindGroupLayout],
	});

	return {
		sceneFrameBindGroupLayout,
		skyboxFrameBindGroupLayout,
		modelBindGroupLayout,
		scenePipelineLayout,
		skyboxPipelineLayout,
	};
}
