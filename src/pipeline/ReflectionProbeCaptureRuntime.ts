import type { Scene } from "../core/Scene";
import { Texture } from "../core/Texture";
import {
	LightType,
	type AmbientLight,
	type AreaLight,
	type DirectionalLight,
	type PointLight,
	type ReflectionProbe,
	type SceneLight,
	type SpotLight,
} from "../lights";
import { sRGBToLinear } from "../maths/Common";
import type { IVector3 } from "../maths/types";
import {
	bakeEnvironmentIBLFromEnvironmentMap,
	type BakedEnvironmentIBL,
	type EnvironmentIBLBakeOptions,
} from "./EnvironmentIBLBaker";
import {
	getDirectionalLightWorldDirection,
	getPointLightWorldPosition,
	getSpotLightInnerAngle,
	getSpotLightWorldDirection,
	getSpotLightWorldPosition,
} from "./LightTransforms";
import { directionToEquirectUV } from "./reflectionProbeRuntime";
import type { WebGPUComputeFacadeSource } from "../renderers/webgpu/ComputeFacade";

const DIRECTIONAL_LOBE_EXPONENT = 96;
const LOCAL_LIGHT_LOBE_EXPONENT = 64;
const AREA_LIGHT_LOBE_EXPONENT = 48;
const DEFAULT_MAX_BAKES_PER_FRAME = 1;
const MIN_LIGHT_DISTANCE = 1e-4;

interface CaptureTaskState {
	taskId: number;
	probeId: string;
	probeSignature: string;
	captureRequestToken: number;
	startedAtSeconds: number;
	scene: Scene;
}

interface RGBLinear {
	r: number;
	g: number;
	b: number;
}

interface CapturedDirectionalLight {
	direction: IVector3;
	color: RGBLinear;
}

interface CapturedLocalLight {
	direction: IVector3;
	color: RGBLinear;
}

interface CaptureLightingState {
	skybox: Texture | null;
	includeSkybox: boolean;
	ambient: RGBLinear;
	directionalLights: CapturedDirectionalLight[];
	pointLights: CapturedLocalLight[];
	spotLights: CapturedLocalLight[];
	areaLights: CapturedLocalLight[];
}

export interface ReflectionProbeCaptureRuntimeExecuteContext {
	scene: Scene;
	nowMs: number;
	webgpuSource?: WebGPUComputeFacadeSource | null;
}

export interface ReflectionProbeCaptureRuntimeOptions {
	maxBakesPerFrame?: number;
	bakeEnvironmentIBL?: (
		envMap: Texture,
		options: EnvironmentIBLBakeOptions
	) => Promise<BakedEnvironmentIBL>;
}

export class ReflectionProbeCaptureRuntime {
	private _inFlightTask: CaptureTaskState | null = null;
	private _nextTaskId = 0;
	private _nextProbeCursor = 0;
	private _maxBakesPerFrame: number;
	private _bakeEnvironmentIBL: (
		envMap: Texture,
		options: EnvironmentIBLBakeOptions
	) => Promise<BakedEnvironmentIBL>;
	private _lastCaptureSecondsByProbeId = new Map<string, number>();
	private _lastCaptureSceneVersionByProbeId = new Map<string, number>();
	private _lastHandledRequestTokenByProbeId = new Map<string, number>();

	constructor(options: ReflectionProbeCaptureRuntimeOptions = {}) {
		this._maxBakesPerFrame = Math.max(
			1,
			Math.floor(options.maxBakesPerFrame ?? DEFAULT_MAX_BAKES_PER_FRAME)
		);
		this._bakeEnvironmentIBL =
			options.bakeEnvironmentIBL ?? bakeEnvironmentIBLFromEnvironmentMap;
	}

	public execute(context: ReflectionProbeCaptureRuntimeExecuteContext): void {
		const probes = collectCapturedSceneProbes(context.scene.getLights());
		this._pruneProbeState(probes);
		if (probes.length === 0 || this._inFlightTask) {
			return;
		}

		const nowSeconds = Math.max(0, context.nowMs) / 1000;
		const sceneVersion = context.scene.version;
		let startedBakes = 0;

		for (let offset = 0; offset < probes.length; offset++) {
			if (startedBakes >= this._maxBakesPerFrame) break;
			const index = (this._nextProbeCursor + offset) % probes.length;
			const probe = probes[index];
			if (!this._shouldCaptureProbe(probe, sceneVersion, nowSeconds)) {
				continue;
			}
			this._nextProbeCursor = (index + 1) % probes.length;
			if (this._startCaptureForProbe(probe, context) === false) {
				continue;
			}
			startedBakes++;
		}
	}

	private _shouldCaptureProbe(
		probe: ReflectionProbe,
		sceneVersion: number,
		nowSeconds: number
	): boolean {
		const lastHandledToken =
			this._lastHandledRequestTokenByProbeId.get(probe.id) ?? 0;
		if (probe.captureRequestToken > lastHandledToken) {
			return true;
		}

		if (probe.captureUpdateMode === "manual") {
			return false;
		}

		if (probe.captureUpdateMode === "onSceneDirty") {
			const lastCapturedSceneVersion =
				this._lastCaptureSceneVersionByProbeId.get(probe.id);
			return lastCapturedSceneVersion !== sceneVersion;
		}

		const lastCaptureSeconds = this._lastCaptureSecondsByProbeId.get(probe.id);
		if (lastCaptureSeconds === undefined) {
			return true;
		}
		return nowSeconds - lastCaptureSeconds >= probe.captureIntervalSeconds;
	}

	private _startCaptureForProbe(
		probe: ReflectionProbe,
		context: ReflectionProbeCaptureRuntimeExecuteContext
	): boolean {
		const task: CaptureTaskState = {
			taskId: ++this._nextTaskId,
			probeId: probe.id,
			probeSignature: buildProbeCaptureSignature(probe),
			captureRequestToken: probe.captureRequestToken,
			startedAtSeconds: Math.max(0, context.nowMs) / 1000,
			scene: context.scene,
		};

		let environmentMap: Texture;
		try {
			environmentMap = captureProbeEnvironmentMap(task.scene, probe);
		} catch {
			return false;
		}

		this._inFlightTask = task;
		void this._runCaptureBake(task, environmentMap, context.webgpuSource ?? null);
		return true;
	}

	private async _runCaptureBake(
		task: CaptureTaskState,
		environmentMap: Texture,
		webgpuSource: WebGPUComputeFacadeSource | null
	): Promise<void> {
		const bakeOptions: EnvironmentIBLBakeOptions = {};
		if (webgpuSource) {
			bakeOptions.webgpuSource = webgpuSource;
		}

		try {
			const baked = await this._bakeEnvironmentIBL(environmentMap, bakeOptions);
			const probe = findCapturedSceneProbe(task.scene, task.probeId);
			if (!probe || !this._isTaskFresh(task, probe)) {
				return;
			}

			probe.prefilteredMap = baked.prefilteredMap;
			probe.markCaptureUpdated();
			probe.markRuntimeDirty();
			task.scene.invalidate("lighting");

			this._lastCaptureSecondsByProbeId.set(
				task.probeId,
				task.startedAtSeconds
			);
			this._lastCaptureSceneVersionByProbeId.set(
				task.probeId,
				task.scene.version
			);
			this._lastHandledRequestTokenByProbeId.set(
				task.probeId,
				task.captureRequestToken
			);
		} finally {
			if (this._inFlightTask?.taskId === task.taskId) {
				this._inFlightTask = null;
			}
		}
	}

	private _isTaskFresh(task: CaptureTaskState, probe: ReflectionProbe): boolean {
		if (this._inFlightTask?.taskId !== task.taskId) {
			return false;
		}
		if (probe.source !== "capturedScene") {
			return false;
		}
		if (probe.captureRequestToken !== task.captureRequestToken) {
			return false;
		}
		return buildProbeCaptureSignature(probe) === task.probeSignature;
	}

	private _pruneProbeState(probes: ReflectionProbe[]): void {
		const activeProbeIds = new Set(probes.map((probe) => probe.id));
		pruneProbeMap(this._lastCaptureSecondsByProbeId, activeProbeIds);
		pruneProbeMap(this._lastCaptureSceneVersionByProbeId, activeProbeIds);
		pruneProbeMap(this._lastHandledRequestTokenByProbeId, activeProbeIds);
		if (
			this._inFlightTask &&
			activeProbeIds.has(this._inFlightTask.probeId) === false
		) {
			this._inFlightTask = null;
		}
	}
}

function collectCapturedSceneProbes(lights: SceneLight[]): ReflectionProbe[] {
	const probes: ReflectionProbe[] = [];
	for (const light of lights) {
		if (light.type !== LightType.ReflectionProbe) continue;
		const probe = light as ReflectionProbe;
		if (probe.source !== "capturedScene") continue;
		probes.push(probe);
	}
	probes.sort((left, right) => left.id.localeCompare(right.id));
	return probes;
}

function findCapturedSceneProbe(
	scene: Scene,
	probeId: string
): ReflectionProbe | null {
	for (const light of scene.getLights()) {
		if (light.type !== LightType.ReflectionProbe) continue;
		const probe = light as ReflectionProbe;
		if (probe.source !== "capturedScene") continue;
		if (probe.id === probeId) {
			return probe;
		}
	}
	return null;
}

function pruneProbeMap<TValue>(
	map: Map<string, TValue>,
	activeProbeIds: Set<string>
): void {
	for (const probeId of map.keys()) {
		if (activeProbeIds.has(probeId)) continue;
		map.delete(probeId);
	}
}

function buildProbeCaptureSignature(probe: ReflectionProbe): string {
	const elements = probe.worldMatrix.elements;
	const matrix = new Array<string>(16);
	let cursor = 0;
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			matrix[cursor++] = elements[row][col].toFixed(6);
		}
	}
	return [
		probe.source,
		probe.captureUpdateMode,
		probe.captureResolution.width,
		probe.captureResolution.height,
		probe.captureFar.toFixed(6),
		probe.includeSkybox ? 1 : 0,
		probe.captureRequestToken,
		...matrix,
	].join("|");
}

function captureProbeEnvironmentMap(
	scene: Scene,
	probe: ReflectionProbe
): Texture {
	const width = Math.max(8, Math.floor(probe.captureResolution.width));
	const height = Math.max(4, Math.floor(probe.captureResolution.height));
	const faceSize = Math.max(4, Math.floor(Math.min(width / 4, height / 2)));
	const lightingState = buildCaptureLightingState(scene, probe);
	const cubeFaces = captureCubeFaces(faceSize, lightingState);
	const equirectData = convertCubeFacesToEquirect(
		cubeFaces,
		faceSize,
		width,
		height
	);

	const texture = new Texture(equirectData, width, height, "HDR");
	texture.wrapS = "Repeat";
	texture.wrapT = "Clamp";
	texture.minFilter = "Linear";
	texture.magFilter = "Linear";
	return texture;
}

function buildCaptureLightingState(
	scene: Scene,
	probe: ReflectionProbe
): CaptureLightingState {
	const lights = scene.getLights();
	const probePosition = probe.getWorldPosition({ x: 0, y: 0, z: 0 });
	const captureRange = Math.max(1, probe.captureFar);
	const ambient: RGBLinear = { r: 0, g: 0, b: 0 };
	const directionalLights: CapturedDirectionalLight[] = [];
	const pointLights: CapturedLocalLight[] = [];
	const spotLights: CapturedLocalLight[] = [];
	const areaLights: CapturedLocalLight[] = [];

	for (const light of lights) {
		switch (light.type) {
			case LightType.Ambient: {
				const ambientLight = light as AmbientLight;
				accumulateAmbient(ambient, ambientLight);
				break;
			}
			case LightType.Directional: {
				const directional = light as DirectionalLight;
				const worldDirection = getDirectionalLightWorldDirection(directional);
				const incoming = normalizeDirection({
					x: -worldDirection.x,
					y: -worldDirection.y,
					z: -worldDirection.z,
				});
				directionalLights.push({
					direction: incoming,
					color: toLinearColor(directional.color, directional.intensity ?? 1),
				});
				break;
			}
			case LightType.Point: {
				const point = light as PointLight;
				const worldPosition = getPointLightWorldPosition(point);
				const captured = buildCapturedLocalLight(
					probePosition,
					worldPosition,
					point.range,
					captureRange,
					toLinearColor(point.color, point.intensity ?? 1)
				);
				if (captured) {
					pointLights.push(captured);
				}
				break;
			}
			case LightType.Spot: {
				const spot = light as SpotLight;
				const worldPosition = getSpotLightWorldPosition(spot);
				const directionToProbe = normalizeDirection({
					x: probePosition.x - worldPosition.x,
					y: probePosition.y - worldPosition.y,
					z: probePosition.z - worldPosition.z,
				});
				const worldDirection = getSpotLightWorldDirection(spot);
				const coneWeight = computeSpotConeWeight(
					worldDirection,
					directionToProbe,
					spot.outerAngle,
					getSpotLightInnerAngle(spot)
				);
				if (coneWeight <= 0) {
					break;
				}
				const color = toLinearColor(spot.color, (spot.intensity ?? 1) * coneWeight);
				const captured = buildCapturedLocalLight(
					probePosition,
					worldPosition,
					spot.range,
					captureRange,
					color
				);
				if (captured) {
					spotLights.push(captured);
				}
				break;
			}
			case LightType.RectArea: {
				const area = light as AreaLight;
				const worldPosition = area.getWorldPosition({ x: 0, y: 0, z: 0 });
				const areaScale = Math.max(1, Math.sqrt(area.width * area.height)) * 0.01;
				const color = toLinearColor(
					area.color,
					(area.intensity ?? 1) * areaScale
				);
				const captured = buildCapturedLocalLight(
					probePosition,
					worldPosition,
					area.range,
					captureRange,
					color
				);
				if (captured) {
					areaLights.push(captured);
				}
				break;
			}
			default:
				break;
		}
	}

	return {
		skybox: probe.includeSkybox ? scene.skybox : null,
		includeSkybox: probe.includeSkybox,
		ambient,
		directionalLights,
		pointLights,
		spotLights,
		areaLights,
	};
}

function accumulateAmbient(
	target: RGBLinear,
	light: AmbientLight
): void {
	const linear = toLinearColor(light.color, light.intensity ?? 1);
	target.r += linear.r;
	target.g += linear.g;
	target.b += linear.b;
}

function buildCapturedLocalLight(
	probePosition: IVector3,
	lightPosition: IVector3,
	lightRange: number,
	captureFar: number,
	color: RGBLinear
): CapturedLocalLight | null {
	const toLight = {
		x: lightPosition.x - probePosition.x,
		y: lightPosition.y - probePosition.y,
		z: lightPosition.z - probePosition.z,
	};
	const distance = Math.hypot(toLight.x, toLight.y, toLight.z);
	const range = Math.max(
		MIN_LIGHT_DISTANCE,
		Math.min(
			Number.isFinite(lightRange) ? Math.max(lightRange, MIN_LIGHT_DISTANCE) : Infinity,
			captureFar
		)
	);
	if (distance > range) {
		return null;
	}
	const attenuation = computeDistanceAttenuation(distance, range);
	const direction = normalizeDirection(toLight);
	return {
		direction,
		color: {
			r: color.r * attenuation,
			g: color.g * attenuation,
			b: color.b * attenuation,
		},
	};
}

function computeSpotConeWeight(
	lightDirection: IVector3,
	directionToProbe: IVector3,
	outerAngle: number,
	innerAngle: number
): number {
	const cosTheta = dot(lightDirection, directionToProbe);
	const outerCos = Math.cos(Math.max(outerAngle, 0));
	const innerCos = Math.cos(Math.max(0, Math.min(innerAngle, outerAngle)));
	if (cosTheta <= outerCos) {
		return 0;
	}
	if (innerCos <= outerCos) {
		return 1;
	}
	return smoothstep(outerCos, innerCos, cosTheta);
}

function captureCubeFaces(
	faceSize: number,
	lightingState: CaptureLightingState
): Float32Array[] {
	const faces: Float32Array[] = [];
	for (let face = 0; face < 6; face++) {
		const data = new Float32Array(faceSize * faceSize * 4);
		for (let y = 0; y < faceSize; y++) {
			for (let x = 0; x < faceSize; x++) {
				const direction = directionFromCubeFace(face, x, y, faceSize);
				const radiance = sampleCapturedRadiance(direction, lightingState);
				const index = (y * faceSize + x) * 4;
				data[index] = radiance.r;
				data[index + 1] = radiance.g;
				data[index + 2] = radiance.b;
				data[index + 3] = 1;
			}
		}
		faces.push(data);
	}
	return faces;
}

function convertCubeFacesToEquirect(
	cubeFaces: Float32Array[],
	faceSize: number,
	width: number,
	height: number
): Float32Array {
	const data = new Float32Array(width * height * 4);
	for (let y = 0; y < height; y++) {
		const v = (y + 0.5) / height;
		for (let x = 0; x < width; x++) {
			const u = (x + 0.5) / width;
			const direction = directionFromEquirectUV(u, v);
			const sample = sampleCubeFaces(cubeFaces, faceSize, direction);
			const index = (y * width + x) * 4;
			data[index] = sample.r;
			data[index + 1] = sample.g;
			data[index + 2] = sample.b;
			data[index + 3] = 1;
		}
	}
	return data;
}

function sampleCapturedRadiance(
	direction: IVector3,
	lightingState: CaptureLightingState
): RGBLinear {
	const result: RGBLinear = {
		r: lightingState.ambient.r,
		g: lightingState.ambient.g,
		b: lightingState.ambient.b,
	};

	if (lightingState.includeSkybox && lightingState.skybox) {
		const sky = sampleSkyboxLinear(lightingState.skybox, direction);
		result.r += sky.r;
		result.g += sky.g;
		result.b += sky.b;
	}

	for (const directional of lightingState.directionalLights) {
		const lobe = Math.pow(
			Math.max(0, dot(direction, directional.direction)),
			DIRECTIONAL_LOBE_EXPONENT
		);
		result.r += directional.color.r * lobe;
		result.g += directional.color.g * lobe;
		result.b += directional.color.b * lobe;
	}

	for (const point of lightingState.pointLights) {
		const lobe = Math.pow(
			Math.max(0, dot(direction, point.direction)),
			LOCAL_LIGHT_LOBE_EXPONENT
		);
		result.r += point.color.r * lobe;
		result.g += point.color.g * lobe;
		result.b += point.color.b * lobe;
	}

	for (const spot of lightingState.spotLights) {
		const lobe = Math.pow(
			Math.max(0, dot(direction, spot.direction)),
			LOCAL_LIGHT_LOBE_EXPONENT
		);
		result.r += spot.color.r * lobe;
		result.g += spot.color.g * lobe;
		result.b += spot.color.b * lobe;
	}

	for (const area of lightingState.areaLights) {
		const lobe = Math.pow(
			Math.max(0, dot(direction, area.direction)),
			AREA_LIGHT_LOBE_EXPONENT
		);
		result.r += area.color.r * lobe;
		result.g += area.color.g * lobe;
		result.b += area.color.b * lobe;
	}

	return result;
}

function directionFromCubeFace(
	face: number,
	x: number,
	y: number,
	faceSize: number
): IVector3 {
	const u = (2 * (x + 0.5)) / faceSize - 1;
	const v = 1 - (2 * (y + 0.5)) / faceSize;
	switch (face) {
		case 0:
			return normalizeDirection({ x: 1, y: v, z: -u });
		case 1:
			return normalizeDirection({ x: -1, y: v, z: u });
		case 2:
			return normalizeDirection({ x: u, y: 1, z: -v });
		case 3:
			return normalizeDirection({ x: u, y: -1, z: v });
		case 4:
			return normalizeDirection({ x: u, y: v, z: 1 });
		default:
			return normalizeDirection({ x: -u, y: v, z: -1 });
	}
}

function directionFromEquirectUV(u: number, v: number): IVector3 {
	const phi = u * (2 * Math.PI) - Math.PI;
	const theta = v * Math.PI;
	const sinTheta = Math.sin(theta);
	return normalizeDirection({
		x: sinTheta * Math.sin(phi),
		y: Math.cos(theta),
		z: sinTheta * Math.cos(phi),
	});
}

function sampleCubeFaces(
	cubeFaces: Float32Array[],
	faceSize: number,
	direction: IVector3
): RGBLinear {
	const { face, uc, vc } = directionToCubeFaceUV(direction);
	const x = Math.max(0, Math.min(faceSize - 1, Math.floor((uc + 1) * 0.5 * faceSize)));
	const y = Math.max(0, Math.min(faceSize - 1, Math.floor((1 - (vc + 1) * 0.5) * faceSize)));
	const data = cubeFaces[face];
	const index = (y * faceSize + x) * 4;
	return {
		r: data[index],
		g: data[index + 1],
		b: data[index + 2],
	};
}

function directionToCubeFaceUV(direction: IVector3): {
	face: number;
	uc: number;
	vc: number;
} {
	const ax = Math.abs(direction.x);
	const ay = Math.abs(direction.y);
	const az = Math.abs(direction.z);
	if (ax >= ay && ax >= az) {
		if (direction.x > 0) {
			return {
				face: 0,
				uc: -direction.z / ax,
				vc: direction.y / ax,
			};
		}
		return {
			face: 1,
			uc: direction.z / ax,
			vc: direction.y / ax,
		};
	}
	if (ay >= ax && ay >= az) {
		if (direction.y > 0) {
			return {
				face: 2,
				uc: direction.x / ay,
				vc: -direction.z / ay,
			};
		}
		return {
			face: 3,
			uc: direction.x / ay,
			vc: direction.z / ay,
		};
	}
	if (direction.z > 0) {
		return {
			face: 4,
			uc: direction.x / az,
			vc: direction.y / az,
		};
	}
	return {
		face: 5,
		uc: -direction.x / az,
		vc: direction.y / az,
	};
}

function sampleSkyboxLinear(
	skybox: Texture,
	direction: IVector3
): RGBLinear {
	const uv = directionToEquirectUV(direction);
	const sample = skybox.sample(uv.u, uv.v);
	if (skybox.colorSpace === "sRGB") {
		return {
			r: sRGBToLinear(sample.r / 255),
			g: sRGBToLinear(sample.g / 255),
			b: sRGBToLinear(sample.b / 255),
		};
	}
	return {
		r: sample.r / 255,
		g: sample.g / 255,
		b: sample.b / 255,
	};
}

function toLinearColor(
	color: { r: number; g: number; b: number },
	intensity: number
): RGBLinear {
	const safeIntensity = Number.isFinite(intensity) ? Math.max(0, intensity) : 0;
	return {
		r: sRGBToLinear(clamp(color.r / 255, 0, 1)) * safeIntensity,
		g: sRGBToLinear(clamp(color.g / 255, 0, 1)) * safeIntensity,
		b: sRGBToLinear(clamp(color.b / 255, 0, 1)) * safeIntensity,
	};
}

function computeDistanceAttenuation(distance: number, range: number): number {
	if (distance <= MIN_LIGHT_DISTANCE) return 1;
	const normalized = clamp(distance / Math.max(range, MIN_LIGHT_DISTANCE), 0, 1);
	const inverseSquare = 1 / Math.max(distance * distance, MIN_LIGHT_DISTANCE);
	const rangeFade = 1 - normalized * normalized;
	return inverseSquare * rangeFade * rangeFade * 128;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
	if (edge1 <= edge0) {
		return x >= edge1 ? 1 : 0;
	}
	const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
	return t * t * (3 - 2 * t);
}

function dot(left: IVector3, right: IVector3): number {
	return left.x * right.x + left.y * right.y + left.z * right.z;
}

function normalizeDirection(direction: IVector3): IVector3 {
	const length = Math.hypot(direction.x, direction.y, direction.z);
	if (length <= MIN_LIGHT_DISTANCE) {
		return { x: 0, y: 0, z: 1 };
	}
	const inv = 1 / length;
	return {
		x: direction.x * inv,
		y: direction.y * inv,
		z: direction.z * inv,
	};
}

function clamp(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(max, value));
}
