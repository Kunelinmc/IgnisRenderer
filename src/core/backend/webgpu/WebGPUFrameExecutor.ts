import type { DrawPacket, FrameContext, FramePass } from '../../pipeline/types'
import type { ICommandEncoder } from '../ICommandEncoder'
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	TextureFormat,
	TextureUsage,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from '../types'
import type { WebGPUBackend } from '../WebGPUBackend'
import type { WebGPURenderResources } from './WebGPURenderResources'
import {
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
} from './constants'
import {
	WebGPUPostProcessGraph,
	type WebGPUFrameTargets,
	type WebGPUPostProcessPassContext,
	type WebGPUPostProcessPassPlugin,
} from './WebGPUPostProcessGraph'

const POST_PROCESS_STAGES = new Set<FramePass['stage']>([
	'ssao',
	'ssr',
	'volumetric',
	'fxaa',
	'gamma',
])

const WEBGPU_PRESENT_SHADER = /* wgsl */ `
struct PresentParams {
	gamma: f32,
	applyGamma: f32,
	_pad0: f32,
	_pad1: f32,
}

struct PresentVSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> presentParams: PresentParams;

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> PresentVSOut {
	var positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0)
	);

	let pos = positions[vertexIndex];
	var output: PresentVSOut;
	output.position = vec4<f32>(pos, 0.0, 1.0);
	// WebGPU texture V-axis is top-origin for sampling; flip Y from clip-space mapping.
	output.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
	return output;
}

@fragment
fn fsMain(input: PresentVSOut) -> @location(0) vec4<f32> {
	let sampled = textureSample(srcTexture, srcSampler, input.uv);
	let gamma = max(presentParams.gamma, 0.01);
	let linearColor = max(sampled.rgb, vec3<f32>(0.0));
	let gammaColor = pow(linearColor, vec3<f32>(1.0 / gamma));
	let outputColor = select(linearColor, gammaColor, presentParams.applyGamma > 0.5);
	return vec4<f32>(clamp(outputColor, vec3<f32>(0.0), vec3<f32>(1.0)), sampled.a);
}
`

export class WebGPUFrameExecutor {
	private _backend: WebGPUBackend
	private _resources: WebGPURenderResources
	private _encoder: ICommandEncoder | null = null
	private _frameContext: FrameContext | null = null
	private _frameTargets: WebGPUFrameTargets | null = null
	private _targetWidth = 0
	private _targetHeight = 0
	private _historyA: IRenderTexture | null = null
	private _historyB: IRenderTexture | null = null
	private _postGraphExecuted = false
	private _hasPresentedInFrame = false
	private _historyValid = false
	private _historyFlip = false
	private _mrtEnabled = true
	private _mrtSupportChecked = false
	private _warnedKeys = new Set<string>()
	private _postGraph: WebGPUPostProcessGraph
	private _presentShaderModule: IShaderModule | null = null
	private _presentPipeline: IRenderPipeline | null = null
	private _presentSampler: ISampler | null = null
	private _presentParamsBuffer: IRenderBuffer | null = null
	private _presentBinding: IBindingGroup | null = null
	private _presentBindingSource: IRenderTexture | null = null

	constructor(backend: WebGPUBackend, resources: WebGPURenderResources) {
		this._backend = backend
		this._resources = resources
		this._postGraph = new WebGPUPostProcessGraph(this._createDefaultPasses())
	}

	public beginFrame(context: FrameContext): void {
		this._frameContext = context
		this._encoder = this._backend.createCommandEncoder()
		this._postGraphExecuted = false
		this._hasPresentedInFrame = false

		this._ensureMRTSupport()
		if (this._mrtEnabled) {
			this._ensureFrameTargets(context.attachments.width, context.attachments.height)
			this._resources.setSceneTargetMode('mrt')
		} else {
			this._destroyFrameTargets()
			this._resources.setSceneTargetMode('single')
		}
	}

	public registerPostProcessPass(pass: WebGPUPostProcessPassPlugin): void {
		this._postGraph.registerPass(pass)
	}

	public unregisterPostProcessPass(id: string): void {
		this._postGraph.unregisterPass(id)
	}

	public async executePass(
		pass: FramePass,
		context: FrameContext
	): Promise<void> {
		if (!this._encoder) return

		switch (pass.stage) {
			case 'main-opaque':
				await this._recordMainPass(context.scene.opaquePackets, true)
				return
			case 'main-transparent':
				await this._recordMainPass(context.scene.transparentPackets, false)
				return
		}

		if (POST_PROCESS_STAGES.has(pass.stage) && !this._postGraphExecuted) {
			await this._runPostGraph(context)
		}
	}

	public async endFrame(): Promise<void> {
		if (!this._encoder) return

		if (this._mrtEnabled && this._frameTargets && !this._hasPresentedInFrame) {
			await this._presentToCanvas(
				this._frameTargets.sceneColor,
				this._frameContext?.features.enableGamma !== false
			)
		}

		const encoder = this._encoder
		const historySource = this._mrtEnabled ? this._frameTargets?.sceneColor : null
		const historyTarget = this._mrtEnabled ? this._frameTargets?.historyWrite : null
		const width = this._targetWidth
		const height = this._targetHeight

		this._backend.submit([encoder.finish()])
		this._encoder = null
		this._frameContext = null

		if (historySource && historyTarget && width > 0 && height > 0) {
			this._backend.copyTextureToTexture(
				{ texture: historySource },
				{ texture: historyTarget },
				{ width, height, depthOrArrayLayers: 1 }
			)
			this._historyValid = true
			this._historyFlip = !this._historyFlip
			if (this._frameTargets) {
				this._applyHistoryFlip(this._frameTargets)
			}
		}
	}

	private _createDefaultPasses(): WebGPUPostProcessPassPlugin[] {
		return [
			{
				id: 'ssao',
				kind: 'compute',
				dependsOn: [],
				isEnabled: (features) => features.enableSSAO,
				execute: async () => {},
			},
			{
				id: 'ssr',
				kind: 'compute',
				dependsOn: ['ssao'],
				isEnabled: (features) => features.enableSSR,
				execute: async () => {},
			},
			{
				id: 'volumetric',
				kind: 'compute',
				dependsOn: ['ssr'],
				isEnabled: (features) => features.enableVolumetric,
				execute: async () => {},
			},
			{
				id: 'fxaa',
				kind: 'compute',
				dependsOn: ['volumetric'],
				isEnabled: (features) => features.enableFXAA,
				execute: async () => {},
			},
			{
				id: 'gamma',
				kind: 'render',
				dependsOn: ['fxaa'],
				isEnabled: (features) => features.enableGamma,
				execute: async (ctx) => {
					await this._presentToCanvas(ctx.targets.sceneColor, true)
				},
			},
		]
	}

	private _ensureMRTSupport(): void {
		if (this._mrtSupportChecked) return
		this._mrtSupportChecked = true

		const maxColorAttachments =
			this._backend.device?.limits?.maxColorAttachments ?? 8
		const maxColorAttachmentBytesPerSample =
			this._backend.device?.limits?.maxColorAttachmentBytesPerSample ?? 32

		if (
			maxColorAttachments >= WEBGPU_MRT_COLOR_TARGET_COUNT &&
			maxColorAttachmentBytesPerSample >= WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE
		) {
			return
		}

		this._mrtEnabled = false
		if (maxColorAttachments < WEBGPU_MRT_COLOR_TARGET_COUNT) {
			this._warnOnce(
				'webgpu-mrt-disabled-attachments',
				`WebGPU device maxColorAttachments is ${maxColorAttachments}, requires ${WEBGPU_MRT_COLOR_TARGET_COUNT}; disabling MRT/GBuffer post-process pipeline`
			)
		}
		if (
			maxColorAttachmentBytesPerSample < WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE
		) {
			this._warnOnce(
				'webgpu-mrt-disabled-bytes',
				`WebGPU device maxColorAttachmentBytesPerSample is ${maxColorAttachmentBytesPerSample}, requires ${WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE}; disabling MRT/GBuffer post-process pipeline`
			)
		}
	}

	private _ensureFrameTargets(width: number, height: number): void {
		if (width <= 0 || height <= 0) {
			this._destroyFrameTargets()
			return
		}

		if (
			this._frameTargets &&
			this._targetWidth === width &&
			this._targetHeight === height
		) {
			this._applyHistoryFlip(this._frameTargets)
			return
		}

		this._destroyFrameTargets()
		this._targetWidth = width
		this._targetHeight = height
		this._historyValid = false

		const sceneColor = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage:
				TextureUsage.RenderAttachment |
				TextureUsage.TextureBinding |
				TextureUsage.CopySrc |
				TextureUsage.CopyDst,
			label: 'WebGPUSceneColor',
		})
		const gAlbedoAlpha = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA8Unorm,
			usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
			label: 'WebGPUGBuffer_AlbedoAlpha',
		})
		const gNormalRoughMetal = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
			label: 'WebGPUGBuffer_NormalRoughMetal',
		})
		const gEmissiveOcclusion = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
			label: 'WebGPUGBuffer_EmissiveOcclusion',
		})
		const gMotionDepth = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
			label: 'WebGPUGBuffer_MotionDepth',
		})
		const depth = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.Depth32Float,
			usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
			label: 'WebGPUDepthSampleable',
		})
		const historyA = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage:
				TextureUsage.RenderAttachment |
				TextureUsage.TextureBinding |
				TextureUsage.CopySrc |
				TextureUsage.CopyDst,
			label: 'WebGPUHistoryA',
		})
		const historyB = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage:
				TextureUsage.RenderAttachment |
				TextureUsage.TextureBinding |
				TextureUsage.CopySrc |
				TextureUsage.CopyDst,
			label: 'WebGPUHistoryB',
		})

		this._frameTargets = {
			sceneColor,
			gAlbedoAlpha,
			gNormalRoughMetal,
			gEmissiveOcclusion,
			gMotionDepth,
			depth,
			historyRead: historyA,
			historyWrite: historyB,
		}
		this._historyA = historyA
		this._historyB = historyB
		this._applyHistoryFlip(this._frameTargets)
	}

	private _applyHistoryFlip(targets: WebGPUFrameTargets): void {
		if (!this._historyA || !this._historyB) return
		targets.historyRead = this._historyFlip ? this._historyB : this._historyA
		targets.historyWrite = this._historyFlip ? this._historyA : this._historyB
	}

	private _destroyFrameTargets(): void {
		if (!this._frameTargets) return
		this._frameTargets.sceneColor.destroy()
		this._frameTargets.gAlbedoAlpha.destroy()
		this._frameTargets.gNormalRoughMetal.destroy()
		this._frameTargets.gEmissiveOcclusion.destroy()
		this._frameTargets.gMotionDepth.destroy()
		this._frameTargets.depth.destroy()
		this._frameTargets.historyRead.destroy()
		this._frameTargets.historyWrite.destroy()
		this._frameTargets = null
		this._historyA = null
		this._historyB = null
		this._presentBinding = null
		this._presentBindingSource = null
		this._targetWidth = 0
		this._targetHeight = 0
		this._historyValid = false
	}

	private _warnOnce(key: string, message: string): void {
		if (this._warnedKeys.has(key)) return
		this._warnedKeys.add(key)
		console.warn(message)
	}

	private async _runPostGraph(context: FrameContext): Promise<void> {
		this._postGraphExecuted = true
		if (!this._mrtEnabled || !this._frameTargets || !this._encoder) {
			return
		}

		context.transient.set('webgpu-history-valid', this._historyValid)
		const postContext: WebGPUPostProcessPassContext = {
			backend: this._backend,
			encoder: this._encoder,
			frameContext: context,
			targets: this._frameTargets,
		}
		const executed = await this._postGraph.execute(
			postContext,
			context.features,
			(key, message) => this._warnOnce(key, message)
		)
		context.transient.set('webgpu-post-order', executed)

		if (!executed.includes('gamma')) {
			await this._presentToCanvas(
				this._frameTargets.sceneColor,
				context.features.enableGamma !== false
			)
		}
	}

	private async _ensurePresentResources(): Promise<void> {
		if (!this._presentShaderModule) {
			this._presentShaderModule = await this._backend.createShaderModule({
				label: 'WebGPUPresentShader',
				code: WEBGPU_PRESENT_SHADER,
			})
		}

		if (!this._presentPipeline) {
			this._presentPipeline = this._backend.createPipeline({
				label: 'WebGPUPresentPipeline',
				vertex: {
					module: this._presentShaderModule,
					entryPoint: 'vsMain',
				},
				fragment: {
					module: this._presentShaderModule,
					entryPoint: 'fsMain',
					targets: [{ format: this._backend.canvasFormat as any }],
				},
				primitive: {
					topology: 'triangle-list' as any,
					cullMode: 'none',
					frontFace: 'ccw',
				},
			} as any)
		}

		if (!this._presentSampler) {
			this._presentSampler = this._backend.createSampler({
				label: 'WebGPUPresentSampler',
				magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear,
				mipmapFilter: FilterMode.Linear,
				addressModeU: AddressMode.ClampToEdge,
				addressModeV: AddressMode.ClampToEdge,
			})
		}

		if (!this._presentParamsBuffer) {
			this._presentParamsBuffer = this._backend.createBuffer({
				label: 'WebGPUPresentParams',
				size: 16,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			})
		}
	}

	private async _presentToCanvas(
		source: IRenderTexture,
		applyGamma: boolean
	): Promise<void> {
		if (!this._encoder) return
		await this._ensurePresentResources()
		if (
			!this._presentPipeline ||
			!this._presentSampler ||
			!this._presentParamsBuffer
		) {
			return
		}

		this._backend.writeBuffer(
			this._presentParamsBuffer,
			new Float32Array([2.2, applyGamma ? 1 : 0, 0, 0])
		)

		if (!this._presentBinding || this._presentBindingSource !== source) {
			this._presentBinding = this._backend.createBindingGroup({
				pipeline: this._presentPipeline,
				layoutIndex: 0,
				entries: [
					{ binding: 0, resource: source },
					{ binding: 1, resource: this._presentSampler },
					{ binding: 2, resource: this._presentParamsBuffer },
				],
				label: 'WebGPUPresentBinding',
			})
			this._presentBindingSource = source
		}

		this._encoder.beginRenderPass({
			label: 'WebGPUPresentPass',
			colorAttachments: [
				{
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: 'clear',
					storeOp: 'store',
				},
			],
		})
		this._encoder.setPipeline(this._presentPipeline)
		this._encoder.setBindingGroup(0, this._presentBinding)
		this._encoder.draw(3)
		this._encoder.endRenderPass()
		this._hasPresentedInFrame = true
	}

	private async _recordMainPass(
		packets: DrawPacket[],
		clearAttachments: boolean
	): Promise<void> {
		if (!this._encoder) return
		if (!this._mrtEnabled || !this._frameTargets) {
			await this._recordLegacyMainPass(packets, clearAttachments)
			return
		}

		this._encoder.beginRenderPass({
			label: clearAttachments ? 'WebGPUMainMRT_Clear' : 'WebGPUMainMRT_Load',
			colorAttachments: [
				{
					view: this._frameTargets.sceneColor,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: clearAttachments ? 'clear' : 'load',
					storeOp: 'store',
				},
				{
					view: this._frameTargets.gAlbedoAlpha,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: clearAttachments ? 'clear' : 'load',
					storeOp: 'store',
				},
				{
					view: this._frameTargets.gNormalRoughMetal,
					clearValue: { r: 0.5, g: 0.5, b: 1, a: 0 },
					loadOp: clearAttachments ? 'clear' : 'load',
					storeOp: 'store',
				},
				{
					view: this._frameTargets.gEmissiveOcclusion,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: clearAttachments ? 'clear' : 'load',
					storeOp: 'store',
				},
				{
					view: this._frameTargets.gMotionDepth,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: clearAttachments ? 'clear' : 'load',
					storeOp: 'store',
				},
			],
			depthStencilAttachment: {
				view: this._frameTargets.depth,
				depthClearValue: 1,
				depthLoadOp: clearAttachments ? 'clear' : 'load',
				depthStoreOp: 'store',
			},
		})

		for (const packet of packets) {
			const resources = await this._resources.getDrawResources(packet)
			if (!resources) continue

			this._encoder.setPipeline(resources.pipeline)
			this._encoder.setBindingGroup(0, resources.frameBinding)
			this._encoder.setBindingGroup(1, resources.modelBinding)
			this._encoder.setVertexBuffer(0, resources.vertexBuffer)
			this._encoder.setIndexBuffer(resources.indexBuffer, 'uint32')
			this._encoder.drawIndexed(resources.indexCount)
		}

		this._encoder.endRenderPass()
	}

	private async _recordLegacyMainPass(
		packets: DrawPacket[],
		clearAttachments: boolean
	): Promise<void> {
		if (!this._encoder) return
		const colorTexture = this._backend.getCanvasColorTexture()
		const depthTexture = this._backend.getCanvasDepthTexture()

		this._encoder.beginRenderPass({
			colorAttachments: [
				{
					view: colorTexture,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: clearAttachments ? 'clear' : 'load',
					storeOp: 'store',
				},
			],
			depthStencilAttachment: {
				view: depthTexture,
				depthClearValue: 1,
				depthLoadOp: clearAttachments ? 'clear' : 'load',
				depthStoreOp: 'store',
			},
		})

		if (clearAttachments) {
			const skyboxResources = await this._resources.getSkyboxResources()
			if (skyboxResources) {
				this._encoder.setPipeline(skyboxResources.pipeline)
				this._encoder.setBindingGroup(0, skyboxResources.frameBinding)
				this._encoder.draw(3)
			}
		}

		for (const packet of packets) {
			const resources = await this._resources.getDrawResources(packet)
			if (!resources) continue

			this._encoder.setPipeline(resources.pipeline)
			this._encoder.setBindingGroup(0, resources.frameBinding)
			this._encoder.setBindingGroup(1, resources.modelBinding)
			this._encoder.setVertexBuffer(0, resources.vertexBuffer)
			this._encoder.setIndexBuffer(resources.indexBuffer, 'uint32')
			this._encoder.drawIndexed(resources.indexCount)
		}

		this._encoder.endRenderPass()
	}
}
