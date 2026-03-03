import { IDevice } from "../ral/IDevice";
import { ICommandEncoder, RenderPassDesc } from "../ral/ICommandEncoder";
import {
	BufferDesc,
	IRenderBuffer,
	TextureDesc,
	IRenderTexture,
	PipelineDesc,
	IRenderPipeline,
	BindingGroupDesc,
	IBindingGroup,
	ShaderModuleDesc,
	IShaderModule,
	SamplerDesc,
	ISampler,
	IndexFormat,
	ComputePipelineDesc,
	IComputePipeline,
	TextureDataLayout,
} from "../ral/types";
import { Rasterizer } from "../software/Rasterizer";

/**
 * SoftwareBackend: Maps the RAL interface to the existing scanline Rasterizer.
 * Behaviorally equivalent to GPU backends, but executes on CPU.
 */
export class SoftwareBackend implements IDevice {
	private _rasterizer: Rasterizer;
	public readonly type = "software";
	public readonly canvasFormat = "rgba8unorm";

	constructor(renderer: any) {
		this._rasterizer = new Rasterizer(renderer);
	}
	async init(): Promise<void> {
		// Software rasterizer is always ready
	}

	public setRenderer(renderer: any) {
		(this._rasterizer as any)._renderer = renderer;
	}

	createBuffer(desc: BufferDesc): IRenderBuffer {
		return {
			size: desc.size,
			destroy: () => {},
			_cpuData: new ArrayBuffer(desc.size),
		} as any;
	}

	createTexture(desc: TextureDesc): IRenderTexture {
		return {
			width: desc.width,
			height: desc.height,
			destroy: () => {},
			_cpuPixels: new Uint8ClampedArray(desc.width * desc.height * 4),
		} as any;
	}

	createSampler(desc: SamplerDesc): ISampler {
		return { label: desc.label } as any;
	}

	async createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule> {
		return {
			label: desc.label,
			softwareDelegate: desc.softwareDelegate,
			code: desc.code,
		} as any;
	}

	createPipeline(desc: PipelineDesc): IRenderPipeline {
		return {
			label: desc.label,
			_config: desc,
		} as any;
	}

	createComputePipeline(desc: ComputePipelineDesc): IComputePipeline {
		return {
			label: desc.label,
			_config: desc,
		} as any;
	}

	createBindingGroup(desc: BindingGroupDesc): IBindingGroup {
		return {
			label: desc.label,
			_entries: desc.entries,
		} as any;
	}

	createCommandEncoder(): ICommandEncoder {
		return new SoftwareCommandEncoder(this._rasterizer);
	}

	writeBuffer(
		buffer: IRenderBuffer,
		data: ArrayBuffer,
		offset: number = 0
	): void {
		const target = (buffer as any)._cpuData;
		new Uint8Array(target).set(new Uint8Array(data), offset);
	}

	writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		desc: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number }
	): void {
		const target = (texture as any)._cpuPixels;
		const source = data instanceof ArrayBuffer ? data : (data as any).buffer;
		const offset = desc.offset || 0;
		const bytesPerRow = desc.bytesPerRow || size.width * 4;
		// Simple copy for now, assuming tightly packed if bytesPerRow not specified
		if (bytesPerRow === size.width * 4) {
			new Uint8Array(target).set(new Uint8Array(source, offset), 0);
		} else {
			const targetRows = new Uint8Array(target);
			const sourceRows = new Uint8Array(source, offset);
			for (let y = 0; y < size.height; y++) {
				const srcStart = y * bytesPerRow;
				const dstStart = y * size.width * 4;
				targetRows.set(
					sourceRows.subarray(srcStart, srcStart + size.width * 4),
					dstStart
				);
			}
		}
	}

	copyTextureToTexture(
		source: { texture: IRenderTexture; origin?: any; aspect?: any },
		destination: { texture: IRenderTexture; origin?: any; aspect?: any },
		copySize: { width: number; height: number; depthOrArrayLayers?: number }
	): void {
		const srcPixels = (source.texture as any)._cpuPixels;
		const dstPixels = (destination.texture as any)._cpuPixels;
		// Simplified full texture copy if no origin
		dstPixels.set(srcPixels);
	}

	submit(commands: any[]): void {
		// Execute the software commands immediately or in batches
		for (const cmd of commands) {
			cmd._execute();
		}

		// Present to canvas
		const renderer = (this._rasterizer as any)._renderer;
		if (renderer && renderer.canvas && renderer.pixels) {
			const ctx = renderer.canvas.getContext("2d");
			if (ctx) {
				const imgData = new ImageData(
					renderer.pixels,
					renderer.canvas.width,
					renderer.canvas.height
				);
				ctx.putImageData(imgData, 0, 0);
			}
		}
	}

	resize(width: number, height: number): void {
		this._rasterizer["_renderer"].canvas.width = width;
		this._rasterizer["_renderer"].canvas.height = height;
		// The depth buffer and color buffer are managed by the renderer's pixels array
	}
}

class SoftwareVertexFetcher {
	private static _cache = new WeakMap<any, any>();

	static getFetcher(layout: any): (vertices: Float32Array, idx: number) => any {
		let fetcher = this._cache.get(layout);
		if (!fetcher) {
			const stride = layout.arrayStride / 4;
			const attrs = layout.attributes.map((a: any) => ({
				loc: a.shaderLocation,
				off: a.offset / 4,
				fmt: a.format,
			}));

			fetcher = (vertices: Float32Array, idx: number) => {
				const res: any = {};
				const base = idx * stride;
				for (let i = 0; i < attrs.length; i++) {
					const a = attrs[i];
					const o = base + a.off;
					const loc = a.loc;
					if (a.fmt === "float32x3") {
						res[loc] = {
							x: vertices[o],
							y: vertices[o + 1],
							z: vertices[o + 2],
						};
					} else if (a.fmt === "float32x2") {
						res[loc] = { x: vertices[o], y: vertices[o + 1] };
					} else if (a.fmt === "float32x4") {
						res[loc] = {
							x: vertices[o],
							y: vertices[o + 1],
							z: vertices[o + 2],
							w: vertices[o + 3],
						};
					}
				}
				return res;
			};
			this._cache.set(layout, fetcher);
		}
		return fetcher;
	}
}

class SoftwareSampler {
	static sample2D(
		texture: IRenderTexture,
		u: number,
		v: number,
		sampler?: ISampler
	): { r: number; g: number; b: number; a: number } {
		const pixels = (texture as any)._cpuPixels;
		const width = texture.width;
		const height = texture.height;
		if (!pixels) return { r: 0, g: 0, b: 0, a: 1 };

		// Basic Repeat addressing
		let uu = u - Math.floor(u);
		let vv = v - Math.floor(v);

		let tx = Math.floor(uu * width);
		let ty = Math.floor(vv * height);
		tx = Math.max(0, Math.min(width - 1, tx));
		ty = Math.max(0, Math.min(height - 1, ty));

		const idx = (ty * width + tx) << 2;
		return {
			r: pixels[idx] / 255,
			g: pixels[idx + 1] / 255,
			b: pixels[idx + 2] / 255,
			a: pixels[idx + 3] / 255,
		};
	}
}

class SoftwareClipping {
	static clipTriangle(vertices: any[]): any[][] {
		// Sutherland-Hodgman clipping against near plane (w > 0.001)
		const near = 0.001;
		const inVertices = vertices;
		const outVertices: any[] = [];

		for (let i = 0; i < inVertices.length; i++) {
			const v1 = inVertices[i];
			const v2 = inVertices[(i + 1) % inVertices.length];

			const v1Inside = v1.position.w >= near;
			const v2Inside = v2.position.w >= near;

			if (v1Inside && v2Inside) {
				outVertices.push(v2);
			} else if (v1Inside && !v2Inside) {
				outVertices.push(this.intersect(v1, v2, near));
			} else if (!v1Inside && v2Inside) {
				outVertices.push(this.intersect(v1, v2, near));
				outVertices.push(v2);
			}
		}

		if (outVertices.length < 3) return [];
		// Triangulate the resulting polygon
		const tris: any[][] = [];
		for (let i = 1; i < outVertices.length - 1; i++) {
			tris.push([outVertices[0], outVertices[i], outVertices[i + 1]]);
		}
		return tris;
	}

	private static intersect(v1: any, v2: any, near: number): any {
		const t = (near - v1.position.w) / (v2.position.w - v1.position.w);
		const res: any = {
			position: {},
			world: {},
			normal: {},
			uv: {},
			tangent: {},
		};
		for (const key of ["x", "y", "z", "w"]) {
			res.position[key] =
				v1.position[key] + t * (v2.position[key] - v1.position[key]);
		}
		for (const key of ["x", "y", "z"]) {
			res.world[key] = v1.world[key] + t * (v2.world[key] - v1.world[key]);
			if (v1.normal)
				res.normal[key] =
					v1.normal[key] + t * (v2.normal[key] - v1.normal[key]);
		}
		if (v1.uv) {
			res.uv.x = v1.uv.x + t * (v2.uv.x - v1.uv.x);
			res.uv.y = v1.uv.y + t * (v2.uv.y - v1.uv.y);
		}
		if (v1.tangent && v2.tangent) {
			for (const key of ["x", "y", "z", "w"]) {
				res.tangent[key] =
					v1.tangent[key] + t * (v2.tangent[key] - v1.tangent[key]);
			}
		}
		if (v1.zView !== undefined && v2.zView !== undefined) {
			res.zView = v1.zView + t * (v2.zView - v1.zView);
		}
		return res;
	}
}

class SoftwareCommandEncoder implements ICommandEncoder {
	private _rasterizer: Rasterizer;
	private _commands: (() => void)[] = [];
	private _currentPipeline: any = null;
	private _currentVBuf: any = null;
	private _currentIBuf: any = null;
	private _currentBindingGroups: Map<number, IBindingGroup> = new Map();
	private _vertexCache: any[] = Array.from({ length: 3 }, () => ({})); // Minimal reuse

	constructor(rasterizer: Rasterizer) {
		this._rasterizer = rasterizer;
	}

	beginRenderPass(desc: RenderPassDesc): void {
		this._commands.push(() => {
			const colorAttachment = desc.colorAttachments[0];
			if (colorAttachment.loadOp === "clear") {
				const clear = colorAttachment.clearValue || { r: 0, g: 0, b: 0, a: 1 };
				const renderer = this._rasterizer["_renderer"];
				const size = renderer.canvas.width * renderer.canvas.height;
				const data = renderer.pixels;
				if (data) {
					for (let i = 0; i < size; i++) {
						const idx = i << 2;
						data[idx] = clear.r * 255;
						data[idx + 1] = clear.g * 255;
						data[idx + 2] = clear.b * 255;
						data[idx + 3] = clear.a * 255;
					}
				}

				const depthBuffer = renderer.depthBuffer;
				if (depthBuffer) {
					depthBuffer.fill(
						desc.depthStencilAttachment?.depthClearValue ?? Infinity
					);
				}

				const normalBuffer = renderer.normalBuffer;
				if (normalBuffer) {
					normalBuffer.fill(0);
				}
			}
		});
	}

	setPipeline(pipeline: IRenderPipeline): void {
		this._currentPipeline = pipeline as any;
	}

	setBindingGroup(index: number, group: IBindingGroup): void {
		this._currentBindingGroups.set(index, group);
	}

	beginComputePass(desc?: any): void {
		// Software compute pass start
	}

	setComputePipeline(pipeline: any): void {
		this._currentPipeline = pipeline;
	}

	dispatchWorkgroups(x: number, y?: number, z?: number): void {
		// Software compute dispatch - no-op for now unless logic provided
	}

	endComputePass(): void {
		// Software compute pass end
	}

	setVertexBuffer(slot: number, buffer: IRenderBuffer): void {
		this._currentVBuf = (buffer as any)._cpuData;
	}

	setIndexBuffer(buffer: IRenderBuffer, format: IndexFormat): void {
		this._currentIBuf = (buffer as any)._cpuData;
	}

	drawIndexed(
		indexCount: number,
		instanceCount: number = 1,
		firstIndex: number = 0
	): void {
		const pipeline = this._currentPipeline;
		if (!pipeline) return;
		const vBuf = this._currentVBuf;
		const iBuf = this._currentIBuf;
		const bindingState = this._currentBindingGroups.get(0) as any;

		this._commands.push(() => {
			const pipelineDesc = pipeline._config ?? pipeline;
			if (!vBuf || !iBuf) return;

			const indices = new Uint32Array(iBuf);
			const vertices = new Float32Array(vBuf);
			const renderer = this._rasterizer["_renderer"];
			if (!renderer || !renderer.camera) return;

			const width = renderer.canvas.width;
			const height = renderer.canvas.height;

			const vLayout = pipelineDesc.vertex.buffers[0];
			const vsDelegate = (pipelineDesc.vertex.module as any).softwareDelegate;
			const fetcher = SoftwareVertexFetcher.getFetcher(vLayout);

			for (let i = 0; i < indexCount; i += 3) {
				const triangleVertices: any[] = [];
				for (let j = 0; j < 3; j++) {
					const idx = indices[firstIndex + i + j];
					const attrs = fetcher(vertices, idx);
					triangleVertices.push(
						this._buildVertexOutput(attrs, renderer, bindingState, vsDelegate)
					);
				}

				// 2. Near-plane Clipping
				const clippedTris = SoftwareClipping.clipTriangle(triangleVertices);

				for (const tri of clippedTris) {
					const projectedVertices = tri.map((v) =>
						this._projectVertex(v, width, height)
					);

					// 3. Back-face Culling
					const v0 = projectedVertices[0];
					const v1 = projectedVertices[1];
					const v2 = projectedVertices[2];
					const area =
						(v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
					if (pipelineDesc.primitive?.cullMode !== "none") {
						if (
							pipelineDesc.primitive?.frontFace === "cw" ? area > 0 : area < 0
						)
							continue;
					}

					const d0 = -(v0.zView ?? 0);
					const d1 = -(v1.zView ?? 0);
					const d2 = -(v2.zView ?? 0);

					const face: any = {
						material: bindingState?._material ?? pipeline._material,
						vertices: projectedVertices,
						projected: projectedVertices,
						center: {
							x: (v0.world.x + v1.world.x + v2.world.x) / 3,
							y: (v0.world.y + v1.world.y + v2.world.y) / 3,
							z: (v0.world.z + v1.world.z + v2.world.z) / 3,
						},
						normal: v0.normal,
						depthInfo: {
							min: Math.min(d0, d1, d2),
							max: Math.max(d0, d1, d2),
							avg: (d0 + d1 + d2) / 3,
						},
					};

					this._rasterizer.drawTriangle(
						projectedVertices as any,
						face,
						renderer.pixels,
						false
					);
				}
			}
		});
	}

	private _buildVertexOutput(
		attrs: any,
		renderer: any,
		bindingState: any,
		vsDelegate: any
	): any {
		if (vsDelegate) {
			return vsDelegate(attrs, this._currentBindingGroups);
		}

		const modelMatrix = bindingState?._modelMatrix;
		const normalMatrix = bindingState?._normalMatrix;
		const pos = attrs[0] || { x: 0, y: 0, z: 0 };
		const normal = attrs[2] || { x: 0, y: 1, z: 0 };
		const tangent = attrs[3];
		const uv = attrs[1] || { x: 0, y: 0 };
		const worldPos = modelMatrix
			? this._transformPoint(modelMatrix, {
					x: pos.x,
					y: pos.y,
					z: pos.z,
					w: 1,
				})
			: { x: pos.x, y: pos.y, z: pos.z, w: 1 };
		const worldNormal = normalMatrix
			? this._normalize(this._transformDirection(normalMatrix, normal))
			: normal;

		let worldTangent = null;
		if (tangent) {
			const tangentDir = normalMatrix
				? this._normalize(this._transformDirection(normalMatrix, tangent))
				: tangent;
			worldTangent = {
				x: tangentDir.x,
				y: tangentDir.y,
				z: tangentDir.z,
				w: tangent.w ?? 1,
			};
		}

		const viewPos = this._transformPoint(renderer.camera.viewMatrix, {
			x: worldPos.x,
			y: worldPos.y,
			z: worldPos.z,
			w: 1,
		});
		const clipPos = this._transformPoint(renderer.camera.projectionMatrix, {
			x: viewPos.x,
			y: viewPos.y,
			z: viewPos.z,
			w: 1,
		});

		return {
			position: clipPos,
			world: { x: worldPos.x, y: worldPos.y, z: worldPos.z },
			normal: worldNormal,
			tangent: worldTangent,
			uv,
			zView: viewPos.z,
		};
	}

	private _projectVertex(vertex: any, width: number, height: number): any {
		const p = vertex.position;
		const clipW = Math.abs(p.w) > 1e-6 ? p.w : p.w >= 0 ? 1e-6 : -1e-6;
		const invW = 1 / clipW;

		return {
			x: (p.x * invW * 0.5 + 0.5) * width,
			y: (0.5 - p.y * invW * 0.5) * height,
			z: p.z * invW,
			w: invW,
			u: vertex.uv?.x ?? 0,
			v: vertex.uv?.y ?? 0,
			normal: vertex.normal,
			tangent: vertex.tangent,
			world: vertex.world,
			zView: vertex.zView,
		};
	}

	private _transformPoint(m: any, p: any): any {
		const e = m.elements;
		const x = p.x,
			y = p.y,
			z = p.z,
			w = p.w;
		return {
			x: e[0][0] * x + e[0][1] * y + e[0][2] * z + e[0][3] * w,
			y: e[1][0] * x + e[1][1] * y + e[1][2] * z + e[1][3] * w,
			z: e[2][0] * x + e[2][1] * y + e[2][2] * z + e[2][3] * w,
			w: e[3][0] * x + e[3][1] * y + e[3][2] * z + e[3][3] * w,
		};
	}

	private _transformDirection(m: any, direction: any): any {
		const e = m.elements ?? m;
		const x = direction.x;
		const y = direction.y;
		const z = direction.z;

		return {
			x: e[0][0] * x + e[0][1] * y + e[0][2] * z,
			y: e[1][0] * x + e[1][1] * y + e[1][2] * z,
			z: e[2][0] * x + e[2][1] * y + e[2][2] * z,
		};
	}

	private _normalize(direction: any): any {
		const len = Math.hypot(direction.x, direction.y, direction.z) || 1;
		return {
			x: direction.x / len,
			y: direction.y / len,
			z: direction.z / len,
		};
	}

	draw(
		vertexCount: number,
		instanceCount: number = 1,
		firstVertex: number = 0
	): void {
		const pipeline = this._currentPipeline;
		if (!pipeline) return;
		const vBuf = this._currentVBuf;
		const bindingState = this._currentBindingGroups.get(0) as any;

		this._commands.push(() => {
			const pipelineDesc = pipeline._config ?? pipeline;
			if (!vBuf) return;

			const vertices = new Float32Array(vBuf);
			const renderer = this._rasterizer["_renderer"];
			if (!renderer || !renderer.camera) return;

			const width = renderer.canvas.width;
			const height = renderer.canvas.height;

			const vLayout = pipelineDesc.vertex.buffers[0];
			const vsDelegate = (pipelineDesc.vertex.module as any).softwareDelegate;
			const fetcher = SoftwareVertexFetcher.getFetcher(vLayout);

			for (let i = 0; i < vertexCount; i += 3) {
				const projectedVertices: any[] = [];

				for (let j = 0; j < 3; j++) {
					const idx = firstVertex + i + j;
					const attrs = fetcher(vertices, idx);
					const vsOutput = this._buildVertexOutput(
						attrs,
						renderer,
						bindingState,
						vsDelegate
					);
					projectedVertices.push(this._projectVertex(vsOutput, width, height));
				}

				// 3. Back-face Culling
				const v0 = projectedVertices[0];
				const v1 = projectedVertices[1];
				const v2 = projectedVertices[2];
				const area =
					(v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
				if (pipelineDesc.primitive?.cullMode !== "none") {
					if (pipelineDesc.primitive?.frontFace === "cw" ? area > 0 : area < 0)
						continue;
				}

				const d0 = -(v0.zView ?? 0);
				const d1 = -(v1.zView ?? 0);
				const d2 = -(v2.zView ?? 0);

				const face: any = {
					material: bindingState?._material ?? pipeline._material,
					vertices: projectedVertices,
					projected: projectedVertices,
					center: {
						x: (v0.world.x + v1.world.x + v2.world.x) / 3,
						y: (v0.world.y + v1.world.y + v2.world.y) / 3,
						z: (v0.world.z + v1.world.z + v2.world.z) / 3,
					},
					normal: v0.normal, // Approximation
					depthInfo: {
						min: Math.min(d0, d1, d2),
						max: Math.max(d0, d1, d2),
						avg: (d0 + d1 + d2) / 3,
					},
				};

				this._rasterizer.drawTriangle(
					projectedVertices as any,
					face,
					renderer.pixels,
					false
				);
			}
		});
	}

	finish(): any {
		const cmds = [...this._commands];
		return {
			_execute: () => {
				for (const f of cmds) f();
			},
		};
	}

	endRenderPass(): void {}
}
