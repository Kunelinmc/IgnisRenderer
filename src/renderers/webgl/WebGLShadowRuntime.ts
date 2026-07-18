import { isShadowCastingLight } from "../../lights";
import type { ShadowMap } from "../../lights/shadows/ShadowMapping";
import {
	createParticleShadowVolumeGrid,
	hasParticleShadowCastingBatches,
	injectParticleBatchIntoShadowVolume,
	mergeParticleShadowBounds,
	resolveParticleShadowCasterBounds,
} from "../../pipeline/ParticleShadowVolume";
import {
	resolveShadowCasterBounds,
	selectCSMDirectionalLights,
	syncShadowMapRegistry,
	updateShadowMapMetadata,
} from "../../pipeline/ShadowMetadata";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type FrameContext,
	type ParticleRenderBatch,
} from "../../pipeline/types";
import { Logger } from "../../foundation/Logger";

import {
	WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH,
	WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES,
	WEBGL_SHADOW_CAPABILITIES,
	WEBGL_TEXTURE_UNIT_PARTICLE_SHADOW_VOLUME,
} from "./constants";
import type { WebGLFrameTargetManager } from "./WebGLFrameTargetManager";
import type { WebGLGeometryRegistry } from "./WebGLGeometryRegistry";
import type { WebGLLightState } from "./WebGLLightCollector";
import type { WebGLProgramLibrary } from "./WebGLProgramLibrary";
import { WebGLShadowPass } from "./WebGLShadowPass";

export interface WebGLShadowRuntimeHost {
	readonly gl: WebGL2RenderingContext;
	readonly programs: WebGLProgramLibrary;
	readonly geometry: WebGLGeometryRegistry;
	readonly targets: WebGLFrameTargetManager;
	readonly maxTextureSize: number;
	readonly maxTextureImageUnits: number;
	getLightState(): WebGLLightState | null;
	getWidth(): number;
	getHeight(): number;
}

/** Owns shadow rendering, metadata, and particle shadow-volume resources. */
export class WebGLShadowRuntime {
	public readonly pass: WebGLShadowPass;
	public particleVolumeTexture: WebGLTexture | null = null;
	public readonly particleVolumeAtlasSize = new Float32Array(2);
	public readonly particleVolumeGridSize = new Float32Array(4);
	public readonly particleVolumeSliceParams = new Float32Array(
		WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES * 4,
	);
	private readonly _host: WebGLShadowRuntimeHost;
	private _particleVolumeAtlasWidth = 0;
	private _particleVolumeAtlasHeight = 0;
	private _particleVolumePixels = new Float32Array(0);

	public constructor(host: WebGLShadowRuntimeHost) {
		this._host = host;
		this.pass = new WebGLShadowPass({
			gl: host.gl,
			programs: host.programs,
			geometry: host.geometry,
			getLightState: () => host.getLightState(),
			getSceneFramebuffer: () => host.targets._sceneFramebuffer,
			getViewportSize: () => ({ width: host.getWidth(), height: host.getHeight() }),
			getMaxTextureSize: () => host.maxTextureSize,
		});
	}

	public beginFrame(context: FrameContext): void {
		this.particleVolumeAtlasSize.fill(0);
		this.particleVolumeGridSize.fill(0);
		this.particleVolumeSliceParams.fill(0);
		this._syncMetadata(context);
	}

	public render(context: FrameContext): void {
		this._updateParticleVolumes(context);
		this.pass.render(context);
	}

	public get atlasTexture(): WebGLTexture | null {
		return this.pass.getTargets().atlasTexture;
	}

	public get transmittanceTexture(): WebGLTexture | null {
		return this.pass.getTargets().transmittanceTexture;
	}

	public get atlasTileSize(): number {
		return this.pass.getTargets().atlasTileSize;
	}

	public destroy(): void {
		this.pass.destroy();
		if (this.particleVolumeTexture) {
			this._host.gl.deleteTexture(this.particleVolumeTexture);
			this.particleVolumeTexture = null;
		}
		this._particleVolumeAtlasWidth = 0;
		this._particleVolumeAtlasHeight = 0;
		this._particleVolumePixels = new Float32Array(0);
	}

	private _syncMetadata(context: FrameContext): void {
		const shadowLights = context.scene.lights.filter(isShadowCastingLight);
		syncShadowMapRegistry(context.shadowMaps, shadowLights);
		if (!context.features.enableShadows) return;
		const bounds = resolveShadowCasterBounds(
			context.scene.shadowCasterPackets,
			context.scene.sceneBounds,
		);
		const combinedBounds = mergeParticleShadowBounds(
			bounds,
			resolveParticleShadowCasterBounds(context.scene.particleSystems),
		);
		const selectedCSMLights = selectCSMDirectionalLights(
			shadowLights,
			WEBGL_SHADOW_CAPABILITIES.maxCsmDirectionalLights,
		);
		for (const light of shadowLights) {
			const renderSet = context.shadowMaps.get(light);
			if (!renderSet) continue;
			updateShadowMapMetadata(renderSet, light, combinedBounds, {
				camera: context.scene.camera,
				backendCapabilities: WEBGL_SHADOW_CAPABILITIES,
				allowCSMDirectionalLights: selectedCSMLights,
				onWarning: (key, message) =>
					Logger.warn(`[${key}] ${message}`, {
						scope: "WebGLShadowRuntime",
						onceKey: key,
					}),
			});
		}
	}

	private _updateParticleVolumes(context: FrameContext): void {
		this.particleVolumeAtlasSize.fill(0);
		this.particleVolumeGridSize.fill(0);
		this.particleVolumeSliceParams.fill(0);
		const batches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) as
			| readonly ParticleRenderBatch[]
			| undefined;
		const shadow = this._host.getLightState()?.directionalShadows[0];
		if (
			!context.features.enableShadows ||
			!shadow?.enabled ||
			!hasParticleShadowCastingBatches(batches)
		) return;
		if (this._host.maxTextureImageUnits <= WEBGL_TEXTURE_UNIT_PARTICLE_SHADOW_VOLUME) {
			this._warn(
				"webgl-particle-shadow-volume-texture-units",
				"WebGL fragment texture unit budget is too small for particle shadow volumes; disabling particle volume shadows for this frame.",
			);
			return;
		}
		const atlasWidth =
			WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH *
			WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS;
		const atlasRows = Math.ceil(
			(WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH *
				WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES) /
				WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS,
		);
		const atlasHeight = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT * atlasRows;
		if (atlasWidth > this._host.maxTextureSize || atlasHeight > this._host.maxTextureSize) {
			this._warn(
				"webgl-particle-shadow-volume-atlas-limit",
				`WebGL particle shadow volume atlas ${atlasWidth}x${atlasHeight} exceeds MAX_TEXTURE_SIZE=${this._host.maxTextureSize}; disabling particle volume shadows for this frame.`,
			);
			return;
		}
		const texture = this._ensureParticleVolumeTexture(atlasWidth, atlasHeight);
		if (!texture) return;
		const requiredPixels = atlasWidth * atlasHeight;
		if (this._particleVolumePixels.length !== requiredPixels) {
			this._particleVolumePixels = new Float32Array(requiredPixels);
		}
		this._particleVolumePixels.fill(0);
		const matrices =
			shadow.strategyType === "csm" ?
				shadow.cascadeViewProjectionMatrices
			: [shadow.viewProjectionMatrix];
		const cascadeCount =
			shadow.strategyType === "csm" ?
				Math.max(1, Math.min(4, shadow.cascadeCount | 0))
			: 1;
		let activeSlices = 0;
		for (
			let sliceIndex = 0;
			sliceIndex < Math.min(WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES, cascadeCount);
			sliceIndex++
		) {
			const matrix = matrices[sliceIndex];
			if (!matrix) continue;
			const grid = createParticleShadowVolumeGrid({
				width: WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH,
				height: WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT,
				depth: WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH,
			});
			const shadowMap = { viewProjectionMatrix: matrix } as ShadowMap;
			for (const batch of batches ?? []) {
				injectParticleBatchIntoShadowVolume(grid, shadowMap, batch);
			}
			if (!grid.active) continue;
			this._packSlice(grid.density, sliceIndex, atlasWidth);
			const offset = sliceIndex * 4;
			this.particleVolumeSliceParams[offset] = 1;
			this.particleVolumeSliceParams[offset + 1] =
				sliceIndex * WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH;
			activeSlices++;
		}
		if (activeSlices === 0) {
			this.particleVolumeSliceParams.fill(0);
			return;
		}
		this.particleVolumeAtlasSize[0] = atlasWidth;
		this.particleVolumeAtlasSize[1] = atlasHeight;
		this.particleVolumeGridSize[0] = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH;
		this.particleVolumeGridSize[1] = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT;
		this.particleVolumeGridSize[2] = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH;
		this.particleVolumeGridSize[3] = WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS;
		const gl = this._host.gl;
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.R32F,
			atlasWidth,
			atlasHeight,
			0,
			gl.RED,
			gl.FLOAT,
			this._particleVolumePixels,
		);
	}

	private _ensureParticleVolumeTexture(
		width: number,
		height: number,
	): WebGLTexture | null {
		if (
			this.particleVolumeTexture &&
			this._particleVolumeAtlasWidth === width &&
			this._particleVolumeAtlasHeight === height
		) return this.particleVolumeTexture;
		const gl = this._host.gl;
		if (this.particleVolumeTexture) gl.deleteTexture(this.particleVolumeTexture);
		const texture = gl.createTexture();
		if (!texture) {
			this._warn(
				"webgl-particle-shadow-volume-create-failed",
				"Failed to create WebGL particle shadow volume atlas; disabling particle volume shadows for this frame.",
			);
			this.particleVolumeTexture = null;
			this._particleVolumeAtlasWidth = 0;
			this._particleVolumeAtlasHeight = 0;
			return null;
		}
		this.particleVolumeTexture = texture;
		this._particleVolumeAtlasWidth = width;
		this._particleVolumeAtlasHeight = height;
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		return texture;
	}

	private _packSlice(density: Float32Array, sliceIndex: number, atlasWidth: number): void {
		const width = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH;
		const height = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT;
		const depth = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH;
		for (let z = 0; z < depth; z++) {
			const tileIndex = sliceIndex * depth + z;
			const tileX = tileIndex % WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS;
			const tileY = Math.floor(tileIndex / WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS);
			for (let y = 0; y < height; y++) {
				const sourceOffset = z * width * height + y * width;
				const targetOffset = (tileY * height + y) * atlasWidth + tileX * width;
				this._particleVolumePixels.set(
					density.subarray(sourceOffset, sourceOffset + width),
					targetOffset,
				);
			}
		}
	}

	private _warn(key: string, message: string): void {
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGLShadowRuntime",
			onceKey: key,
		});
	}
}
