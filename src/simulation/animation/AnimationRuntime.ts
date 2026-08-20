import { Quaternion } from "../../maths/Quaternion";
import type { AnimationSystem } from "../../animation/AnimationSystem";
import type { AnimationMixer } from "../../animation/AnimationMixer";
import type { AnimationLayer } from "../../animation/AnimationLayer";
import type { AnimationStateMachine } from "../../animation/AnimationStateMachine";
import type { AnimationMotionDefinition } from "../../animation/types";
import { sampleTrack } from "./interpolation";
import {
	ANIMATION_DEFORMATION_STATES_KEY,
	ANIMATION_RUNTIME_POSE_KEY,
	ANIMATION_SOFTWARE_DEFORMED_GEOMETRY_KEY,
	ANIMATION_JOINT_MATRICES_KEY,
	ANIMATION_MORPH_WEIGHTS_KEY,
	type AnimationPoseState,
	type DeformedGeometryMap,
	type JointMatrixMap,
	type MorphWeightMap,
	type PrimitiveDeformationMap,
} from "./types";
import { deformPrimitiveGeometry } from "./SoftwareAnimationDeformer";
import type { MeshInstance } from "../../meshes";
import type { KeyframeTrack } from "../../animation/KeyframeTrack";
import type { Scene } from "../../core/Scene";
import type { TransientStore } from "../../foundation/TransientStore";
import type { BoundingSphere, IPrimitiveGeometry } from "../../core/types";
import type { Skeleton } from "../../animation/Skeleton";

interface TrackAccumulator {
	track: KeyframeTrack;
	overrideValues: number[];
	overrideWeight: number;
	additiveValues: number[];
	quaternionOverrides: Array<{ value: Quaternion; weight: number }>;
	quaternionAdditives: Array<{ value: Quaternion; weight: number }>;
}

interface JointRevisionState {
	skeleton: Skeleton;
	matrices: Float32Array;
	revision: number;
}

interface PrimitiveRevisionState {
	geometry: IPrimitiveGeometry;
	geometryVersion: number;
	jointRevision: number;
	morphWeights: Float32Array;
	revision: number;
}

const EMPTY_MORPH_WEIGHTS = new Float32Array(0);

function isDeformableInstance(instance: MeshInstance): boolean {
	if (instance.skeleton) return true;
	return instance.mesh.primitives.some(
		(primitive) => (primitive.geometry.morphTargets?.length ?? 0) > 0,
	);
}

export class AnimationRuntime {
	private _defaultsByMixer = new WeakMap<
		AnimationMixer,
		Map<string, number[]>
	>();
	private _transitionStateByMachine = new WeakMap<
		AnimationStateMachine,
		number | null
	>();
	private _jointRevisionStateByInstance = new Map<string, JointRevisionState>();
	private _primitiveRevisionStateByPacket = new Map<
		string,
		PrimitiveRevisionState
	>();
	private _nextDeformationRevision = 1;

	public update(
		system: AnimationSystem,
		deltaSeconds: number,
		transient: TransientStore,
		scene?: Scene
	): void {
		this.updatePose(system, deltaSeconds, transient, scene);
		this.resolveDeformations(system, transient, scene);
	}

	/**
	 * Samples animation state into scene and morph authoring values.
	 *
	 * @internal Owned by the renderer animation stage. Deformation payloads are
	 * resolved separately after world transforms are current.
	 */
	public updatePose(
		system: AnimationSystem,
		deltaSeconds: number,
		transient: TransientStore,
		scene?: Scene
	): void {
		const dt = Math.max(0, deltaSeconds);
		const poseStates: AnimationPoseState[] = [];

		for (const mixer of system.mixers) {
			this._updateMixer(
				mixer,
				dt,
				poseStates,
				scene
			);
		}

		transient.set(ANIMATION_RUNTIME_POSE_KEY, poseStates);
	}

	/**
	 * Resolves current deformation payloads after world transforms are updated.
	 *
	 * @internal Owned by the renderer deformation stage.
	 */
	public resolveDeformations(
		system: AnimationSystem,
		transient: TransientStore,
		scene?: Scene,
	): void {
		const deformedGeometry: DeformedGeometryMap = new Map();
		const jointMatrices: JointMatrixMap = new Map();
		const morphWeights: MorphWeightMap = new Map();
		const deformationStates: PrimitiveDeformationMap = new Map();
		const nextJointStates = new Map<string, JointRevisionState>();
		const nextPrimitiveStates = new Map<string, PrimitiveRevisionState>();
		const instances = new Set<MeshInstance>();
		for (const instance of scene?.getMeshInstances() ?? []) {
			if (isDeformableInstance(instance)) instances.add(instance);
		}

		for (const mixer of system.mixers) {
			for (const binding of mixer.morphBindings.values()) {
				instances.add(binding as MeshInstance);
			}
		}

		for (const instance of instances) {
			let jointRevision = 0;
			if (instance.skeleton) {
				instance.skeleton.updateJointMatrices(instance.worldMatrix);
				const matrices = instance.skeleton.toFloat32Array();
				const previous = this._jointRevisionStateByInstance.get(instance.id);
				const changed =
					!previous ||
					previous.skeleton !== instance.skeleton ||
					!floatArraysEqual(previous.matrices, matrices);
				jointRevision = changed
					? this._allocateDeformationRevision()
					: previous.revision;
				nextJointStates.set(instance.id, {
					skeleton: instance.skeleton,
					matrices,
					revision: jointRevision,
				});
				jointMatrices.set(instance.id, {
					skeleton: instance.skeleton,
					matrices,
				});
			}

			for (let index = 0; index < instance.mesh.primitives.length; index++) {
				const primitive = instance.mesh.primitives[index];
				const weights = instance.morphWeights[index] ?? EMPTY_MORPH_WEIGHTS;
				const hasSkinning =
					!!instance.skeleton &&
					!!primitive.geometry.joints0 &&
					!!primitive.geometry.weights0;
				const hasMorphTargets =
					(primitive.geometry.morphTargets?.length ?? 0) > 0 &&
					weights.length > 0;
				if (!hasSkinning && !hasMorphTargets) continue;
				const packetId = `${instance.id}:${primitive.id}`;
				const weightsSnapshot = new Float32Array(weights);
				if (hasMorphTargets) {
					morphWeights.set(packetId, {
						packetId,
						weights: weightsSnapshot,
						targetCount: weightsSnapshot.length,
					});
				}

				const override = deformPrimitiveGeometry({
					geometry: primitive.geometry,
					morphWeights: weights,
					skeleton: hasSkinning ? instance.skeleton : null,
					meshWorldMatrix: instance.worldMatrix,
					jointMatricesCurrent: true,
				});
				const localBounds = computePositionBounds(
					override.positions ?? primitive.geometry.positions
				);
				const geometryVersion = primitive.geometryVersion ?? 0;
				const previous = this._primitiveRevisionStateByPacket.get(packetId);
				const changed =
					!previous ||
					previous.geometry !== primitive.geometry ||
					previous.geometryVersion !== geometryVersion ||
					previous.jointRevision !== (hasSkinning ? jointRevision : 0) ||
					!floatArraysEqual(previous.morphWeights, weightsSnapshot);
				const revision = changed
					? this._allocateDeformationRevision()
					: previous.revision;

				deformedGeometry.set(packetId, override);
				deformationStates.set(packetId, {
					packetId,
					revision,
					localBounds,
				});
				nextPrimitiveStates.set(packetId, {
					geometry: primitive.geometry,
					geometryVersion,
					jointRevision: hasSkinning ? jointRevision : 0,
					morphWeights: weightsSnapshot,
					revision,
				});
			}
		}

		transient.set(ANIMATION_SOFTWARE_DEFORMED_GEOMETRY_KEY, deformedGeometry);
		transient.set(ANIMATION_JOINT_MATRICES_KEY, jointMatrices);
		transient.set(ANIMATION_MORPH_WEIGHTS_KEY, morphWeights);
		transient.set(ANIMATION_DEFORMATION_STATES_KEY, deformationStates);
		this._jointRevisionStateByInstance = nextJointStates;
		this._primitiveRevisionStateByPacket = nextPrimitiveStates;
	}

	private _updateMixer(
		mixer: AnimationMixer,
		deltaSeconds: number,
		poseStates: AnimationPoseState[],
		scene?: Scene
	): void {
		this._updateStateMachines(mixer, deltaSeconds);

		for (const layer of mixer.layers) {
			for (const action of layer.actions) {
				action.update(deltaSeconds);
			}
		}

		const defaults = this._getDefaultsForMixer(mixer);
		const accumulators = new Map<string, TrackAccumulator>();

		for (const layer of mixer.layers) {
			const layerWeight = Math.max(0, layer.weight);
			if (layerWeight <= 1e-6) continue;

			for (const action of layer.actions) {
				if (!action.enabled || action.finished) continue;
				const actionWeight = Math.max(0, action.weight * layerWeight);
				if (actionWeight <= 1e-6) continue;

				const duration = Math.max(1e-6, action.clip.duration);
				const localTime = normalizeTime(action.time, duration);
				for (const track of action.clip.tracks) {
					if (!layer.allowsPath(track.binding.targetPath)) continue;
					if (
						!mixer.rootMotion.enabled &&
						track.binding.targetType === "node" &&
						track.binding.property === "translation" &&
						track.binding.targetPath === mixer.rootMotion.trackPath
					) {
						continue;
					}
					const isRotation = track.binding.property === "rotation";
					const sampled = sampleTrack(track, localTime, {
						isQuaternion: isRotation,
					});
					if (sampled.length === 0) continue;
					const key = bindingKey(track);
					let accumulator = accumulators.get(key);
					if (!accumulator) {
						accumulator = {
							track,
							overrideValues: new Array(sampled.length).fill(0),
							overrideWeight: 0,
							additiveValues: new Array(sampled.length).fill(0),
							quaternionOverrides: [],
							quaternionAdditives: [],
						};
						accumulators.set(key, accumulator);
					}

					if (isRotation) {
						const q = new Quaternion(
							sampled[0],
							sampled[1],
							sampled[2],
							sampled[3]
						).normalize();
						if (layer.blendMode === "additive" || action.additive) {
							accumulator.quaternionAdditives.push({
								value: q,
								weight: actionWeight,
							});
						} else {
							accumulator.quaternionOverrides.push({
								value: q,
								weight: actionWeight,
							});
						}
						continue;
					}

					if (layer.blendMode === "additive" || action.additive) {
						for (let i = 0; i < sampled.length; i++) {
							accumulator.additiveValues[i] += sampled[i] * actionWeight;
						}
						continue;
					}

					for (let i = 0; i < sampled.length; i++) {
						accumulator.overrideValues[i] += sampled[i] * actionWeight;
					}
					accumulator.overrideWeight += actionWeight;
				}
			}
		}

		for (const accumulator of accumulators.values()) {
			const track = accumulator.track;
			const defaultValue = this._getDefaultValue(mixer, defaults, track, scene);
			let finalValue = defaultValue.slice(0, accumulator.overrideValues.length);

			if (track.binding.property === "rotation") {
				const baseQuaternion =
					accumulator.quaternionOverrides.length > 0
						? blendQuaternions(accumulator.quaternionOverrides)
						: new Quaternion(
								defaultValue[0],
								defaultValue[1],
								defaultValue[2],
								defaultValue[3]
							).normalize();

				let quaternion = baseQuaternion;
				if (accumulator.quaternionAdditives.length > 0) {
					const additive = blendAdditiveQuaternion(
						accumulator.quaternionAdditives
					);
					quaternion = Quaternion.multiply(quaternion, additive).normalize();
				}
				finalValue = [quaternion.x, quaternion.y, quaternion.z, quaternion.w];
			} else {
				if (accumulator.overrideWeight > 1e-6) {
					for (let i = 0; i < accumulator.overrideValues.length; i++) {
						finalValue[i] =
							accumulator.overrideValues[i] / accumulator.overrideWeight;
					}
				}
				for (let i = 0; i < accumulator.additiveValues.length; i++) {
					finalValue[i] += accumulator.additiveValues[i];
				}
			}

			if (
				this._applyTrackValue(mixer, track, finalValue, poseStates, scene)
			) {
				defaults.set(bindingKey(track), finalValue.slice());
			}
		}

	}

	private _allocateDeformationRevision(): number {
		return this._nextDeformationRevision++;
	}

	private _updateStateMachines(
		mixer: AnimationMixer,
		deltaSeconds: number
	): void {
		for (const [layerName, stateMachine] of mixer.stateMachines.entries()) {
			const layer = mixer.getOrCreateLayer(layerName);
			const primary = layer.actions.find((action) => action.enabled) ?? null;
			const normalized = primary
				? normalizeTime(primary.time, Math.max(primary.clip.duration, 1e-6)) /
					Math.max(primary.clip.duration, 1e-6)
				: 0;
			stateMachine.update(normalized, deltaSeconds);

			const transition = stateMachine.transitionState;
			const transitionKey = transition ? transition.id : null;
			const previousTransitionKey =
				this._transitionStateByMachine.get(stateMachine);
			if (transitionKey && transitionKey !== previousTransitionKey) {
				this._applyTransitionCrossFade(mixer, layer, stateMachine);
			}
			this._transitionStateByMachine.set(stateMachine, transitionKey);

			this._applyStateMotion(mixer, layer, stateMachine, deltaSeconds);
		}
	}

	private _applyTransitionCrossFade(
		mixer: AnimationMixer,
		layer: AnimationLayer,
		stateMachine: AnimationStateMachine
	): void {
		const transition = stateMachine.transitionState;
		if (!transition) return;
		const fromState = stateMachine.getStateDefinition(transition.from);
		const toState = stateMachine.getStateDefinition(transition.to);
		if (!fromState || !toState) return;
		for (const clipName of this._collectMotionClipNames(
			mixer,
			fromState.motion
		)) {
			const action = mixer.clipAction(clipName, layer.name);
			action.play().fadeOut(transition.duration);
		}
		for (const clipName of this._collectMotionClipNames(
			mixer,
			toState.motion
		)) {
			const action = mixer.clipAction(clipName, layer.name);
			action.play().fadeIn(transition.duration);
		}
	}

	private _applyStateMotion(
		mixer: AnimationMixer,
		layer: AnimationLayer,
		stateMachine: AnimationStateMachine,
		deltaSeconds: number
	): void {
		const state = stateMachine.currentState;
		if (!state) return;
		const desired = this._resolveMotionWeights(
			mixer,
			state.motion,
			stateMachine,
			deltaSeconds
		);
		for (const [clipName, weight] of desired.entries()) {
			const action = mixer.clipAction(clipName, layer.name);
			action.play();
			action.loop = state.loop ?? true;
			action.setEffectiveTimeScale(state.speed ?? 1);
			action.setEffectiveWeight(weight);
		}
		for (const action of layer.actions) {
			if (desired.has(action.clip.name)) continue;
			action.setEffectiveWeight(0);
		}
	}

	private _resolveMotionWeights(
		mixer: AnimationMixer,
		motion: AnimationMotionDefinition,
		stateMachine: AnimationStateMachine,
		deltaSeconds: number
	): Map<string, number> {
		if (motion.type === "clip") {
			return new Map([[motion.clipName, 1]]);
		}
		if (motion.type === "blendtree1d") {
			const tree = mixer.blendTrees1D.get(motion.treeName);
			if (!tree) return new Map();
			const value = Number(stateMachine.getParameter(tree.parameter) ?? 0);
			const map = new Map<string, number>();
			for (const child of tree.evaluate(value)) {
				map.set(child.clipName, child.weight);
			}
			return map;
		}
		if (motion.type === "blendtree2d") {
			const tree = mixer.blendTrees2D.get(motion.treeName);
			if (!tree) return new Map();
			const valueX = Number(stateMachine.getParameter(tree.parameterX) ?? 0);
			const valueY = Number(stateMachine.getParameter(tree.parameterY) ?? 0);
			const map = new Map<string, number>();
			for (
				const child of tree.evaluate(valueX, valueY, tree.blendMode, {
					deltaTimeSeconds: deltaSeconds,
				})
			) {
				map.set(child.clipName, child.weight);
			}
			return map;
		}
		const tree = mixer.blendTreesDirect.get(motion.treeName);
		if (!tree) return new Map();
		const map = new Map<string, number>();
		for (const child of tree.evaluate(stateMachine.parameterValues)) {
			map.set(child.clipName, child.weight);
		}
		return map;
	}

	private _collectMotionClipNames(
		mixer: AnimationMixer,
		motion: AnimationMotionDefinition
	): string[] {
		if (motion.type === "clip") return [motion.clipName];
		if (motion.type === "blendtree1d") {
			const tree = mixer.blendTrees1D.get(motion.treeName);
			return tree ? tree.children.map((child) => child.clipName) : [];
		}
		if (motion.type === "blendtree2d") {
			const tree = mixer.blendTrees2D.get(motion.treeName);
			return tree ? tree.children.map((child) => child.clipName) : [];
		}
		const tree = mixer.blendTreesDirect.get(motion.treeName);
		return tree ? tree.children.map((child) => child.clipName) : [];
	}

	private _getDefaultsForMixer(mixer: AnimationMixer): Map<string, number[]> {
		let defaults = this._defaultsByMixer.get(mixer);
		if (!defaults) {
			defaults = new Map();
			this._defaultsByMixer.set(mixer, defaults);
		}
		return defaults;
	}

	private _getDefaultValue(
		mixer: AnimationMixer,
		defaults: Map<string, number[]>,
		track: KeyframeTrack,
		scene?: Scene
	): number[] {
		const key = bindingKey(track);
		const existing = defaults.get(key);
		if (existing) return existing;

		const binding = track.binding;
		let value: number[] = [];
		if (binding.targetType === "node") {
			const entityValue = this._getEntityDefaultValue(
				mixer,
				binding.targetPath,
				binding.property,
				scene
			);
			if (entityValue) {
				value = entityValue;
			}

			const node = mixer.nodeBindings.get(binding.targetPath);
			if (node && value.length === 0) {
				switch (binding.property) {
					case "translation":
						value = [node.position.x, node.position.y, node.position.z];
						break;
					case "scale":
						value = [node.scale.x, node.scale.y, node.scale.z];
						break;
					case "rotation":
						value = [
							node.quaternion.x,
							node.quaternion.y,
							node.quaternion.z,
							node.quaternion.w,
						];
						break;
				}
			}
		}
		if (binding.targetType === "material") {
			const material = mixer.materialBindings.get(binding.targetPath);
			if (material) {
				switch (binding.property) {
					case "opacity":
						value = [Number(material.opacity ?? 1)];
						break;
					case "emissiveIntensity":
						value = [Number(material.emissiveIntensity ?? 1)];
						break;
					case "baseColor":
						if ("albedo" in material) {
							value = [
								material.albedo.r / 255,
								material.albedo.g / 255,
								material.albedo.b / 255,
							];
						}
						break;
					case "emissive":
						if ("emissive" in material) {
							value = [
								material.emissive.r / 255,
								material.emissive.g / 255,
								material.emissive.b / 255,
							];
						}
						break;
				}
			}
		}
		if (binding.targetType === "morph") {
			const instance = mixer.morphBindings.get(binding.targetPath) as
				| MeshInstance
				| undefined;
			if (instance) {
				if (binding.morphTargetIndex !== undefined) {
					const weight =
						instance.morphWeights[0]?.[binding.morphTargetIndex] ?? 0;
					value = [weight];
				} else {
					value = Array.from(instance.morphWeights[0] ?? []);
				}
			}
		}

		if (value.length === 0) {
			value = new Array(track.valueSize).fill(0);
		}
		defaults.set(key, value);
		return value;
	}

	private _applyTrackValue(
		mixer: AnimationMixer,
		track: KeyframeTrack,
		value: number[],
		poseStates: AnimationPoseState[],
		scene?: Scene
	): boolean {
		const binding = track.binding;
		if (binding.targetType === "node") {
			const appliedToEntity = this._applyEntityTrackValue(
				mixer,
				binding.targetPath,
				binding.property,
				value,
				scene
			);
			const node = mixer.nodeBindings.get(binding.targetPath);
			if (!node) return appliedToEntity;
			if (binding.property === "translation" && value.length >= 3) {
				node.position.set(value[0], value[1], value[2]);
				node.updateLocalMatrix();
				poseStates.push({
					path: binding.targetPath,
					translation: [value[0], value[1], value[2]],
				});
				return true;
			}
			if (binding.property === "scale" && value.length >= 3) {
				node.scale.set(value[0], value[1], value[2]);
				node.updateLocalMatrix();
				poseStates.push({
					path: binding.targetPath,
					scale: [value[0], value[1], value[2]],
				});
				return true;
			}
			if (binding.property === "rotation" && value.length >= 4) {
				node.quaternion = new Quaternion(
					value[0],
					value[1],
					value[2],
					value[3]
				).normalize();
				node.updateLocalMatrix();
				poseStates.push({
					path: binding.targetPath,
					rotation: [value[0], value[1], value[2], value[3]],
				});
				return true;
			}
			return appliedToEntity;
		}
		if (binding.targetType === "material") {
			const material = mixer.materialBindings.get(binding.targetPath);
			if (!material) return false;
			switch (binding.property) {
				case "opacity":
					material.opacity = value[0] ?? material.opacity;
					return true;
				case "emissiveIntensity":
					material.emissiveIntensity = value[0] ?? material.emissiveIntensity;
					return true;
				case "baseColor":
					if ("albedo" in material) {
						material.albedo.r = Math.max(
							0,
							Math.min(255, (value[0] ?? 0) * 255)
						);
						material.albedo.g = Math.max(
							0,
							Math.min(255, (value[1] ?? 0) * 255)
						);
						material.albedo.b = Math.max(
							0,
							Math.min(255, (value[2] ?? 0) * 255)
						);
						return true;
					}
					return false;
				case "emissive":
					if ("emissive" in material) {
						material.emissive.r = Math.max(
							0,
							Math.min(255, (value[0] ?? 0) * 255)
						);
						material.emissive.g = Math.max(
							0,
							Math.min(255, (value[1] ?? 0) * 255)
						);
						material.emissive.b = Math.max(
							0,
							Math.min(255, (value[2] ?? 0) * 255)
						);
						return true;
					}
					return false;
			}
			return false;
		}
		if (binding.targetType === "morph") {
			const instance = mixer.morphBindings.get(binding.targetPath) as
				| MeshInstance
				| undefined;
			if (!instance) return false;
			if (binding.morphTargetIndex !== undefined) {
				for (const weights of instance.morphWeights) {
					if (binding.morphTargetIndex >= weights.length) continue;
					weights[binding.morphTargetIndex] =
						value[0] ?? weights[binding.morphTargetIndex];
				}
				return true;
			}
			for (const weights of instance.morphWeights) {
				const count = Math.min(weights.length, value.length);
				for (let i = 0; i < count; i++) {
					weights[i] = value[i];
				}
			}
			return true;
		}
		return false;
	}

	private _getEntityDefaultValue(
		mixer: AnimationMixer,
		path: string,
		property: string,
		scene: Scene | undefined
	): number[] | null {
		if (!scene) return null;
		const entityId = mixer.entityBindings.get(path);
		if (entityId === undefined) return null;
		const local = scene.ecs.getComponent(entityId, "LocalTransform");
		if (!local) return null;

		switch (property) {
			case "translation":
				return [local.positionX, local.positionY, local.positionZ];
			case "scale":
				return [local.scaleX, local.scaleY, local.scaleZ];
			case "rotation":
				return [
					local.rotationX,
					local.rotationY,
					local.rotationZ,
					local.rotationW,
				];
			default:
				return null;
		}
	}

	private _applyEntityTrackValue(
		mixer: AnimationMixer,
		path: string,
		property: string,
		value: number[],
		scene: Scene | undefined
	): boolean {
		if (!scene) return false;
		const entityId = mixer.entityBindings.get(path);
		if (entityId === undefined) return false;
		const local = scene.ecs.getComponent(entityId, "LocalTransform");
		if (!local) return false;

		if (property === "translation" && value.length >= 3) {
			local.positionX = value[0];
			local.positionY = value[1];
			local.positionZ = value[2];
		} else if (property === "scale" && value.length >= 3) {
			local.scaleX = value[0];
			local.scaleY = value[1];
			local.scaleZ = value[2];
		} else if (property === "rotation" && value.length >= 4) {
			local.rotationX = value[0];
			local.rotationY = value[1];
			local.rotationZ = value[2];
			local.rotationW = value[3];
		} else {
			return false;
		}

		scene.ecs.setComponent(entityId, "LocalTransform", local);
		const node = scene.ecs.getNodeByEntity(entityId);
		if (node) {
			node.position.set(local.positionX, local.positionY, local.positionZ);
			node.quaternion.x = local.rotationX;
			node.quaternion.y = local.rotationY;
			node.quaternion.z = local.rotationZ;
			node.quaternion.w = local.rotationW;
			node.scale.set(local.scaleX, local.scaleY, local.scaleZ);
			node.updateLocalMatrix();
		}
		return true;
	}
}

function bindingKey(track: KeyframeTrack): string {
	const binding = track.binding;
	return [
		binding.targetType,
		binding.targetPath,
		binding.property,
		binding.morphTargetIndex ?? -1,
	].join("|");
}

function floatArraysEqual(left: Float32Array, right: Float32Array): boolean {
	if (left.length !== right.length) return false;
	for (let index = 0; index < left.length; index++) {
		if (left[index] !== right[index]) return false;
	}
	return true;
}

function computePositionBounds(positions: ArrayLike<number>): BoundingSphere {
	if (positions.length < 3) {
		return {
			center: { x: 0, y: 0, z: 0 },
			radius: 0,
		};
	}

	let minX = Number.POSITIVE_INFINITY;
	let minY = Number.POSITIVE_INFINITY;
	let minZ = Number.POSITIVE_INFINITY;
	let maxX = Number.NEGATIVE_INFINITY;
	let maxY = Number.NEGATIVE_INFINITY;
	let maxZ = Number.NEGATIVE_INFINITY;
	for (let index = 0; index + 2 < positions.length; index += 3) {
		const x = Number(positions[index]);
		const y = Number(positions[index + 1]);
		const z = Number(positions[index + 2]);
		minX = Math.min(minX, x);
		minY = Math.min(minY, y);
		minZ = Math.min(minZ, z);
		maxX = Math.max(maxX, x);
		maxY = Math.max(maxY, y);
		maxZ = Math.max(maxZ, z);
	}
	const center = {
		x: (minX + maxX) * 0.5,
		y: (minY + maxY) * 0.5,
		z: (minZ + maxZ) * 0.5,
	};
	let radiusSquared = 0;
	for (let index = 0; index + 2 < positions.length; index += 3) {
		const dx = Number(positions[index]) - center.x;
		const dy = Number(positions[index + 1]) - center.y;
		const dz = Number(positions[index + 2]) - center.z;
		radiusSquared = Math.max(radiusSquared, dx * dx + dy * dy + dz * dz);
	}
	return {
		center,
		radius: Math.sqrt(radiusSquared),
	};
}

function normalizeTime(time: number, duration: number): number {
	if (duration <= 0) return 0;
	let t = time;
	while (t < 0) t += duration;
	while (t > duration) t -= duration;
	return t;
}

function blendQuaternions(
	samples: Array<{ value: Quaternion; weight: number }>
): Quaternion {
	let x = 0;
	let y = 0;
	let z = 0;
	let w = 0;
	let total = 0;
	let reference: Quaternion | null = null;
	for (const sample of samples) {
		if (sample.weight <= 1e-6) continue;
		if (!reference) {
			reference = sample.value;
		}
		let sx = sample.value.x;
		let sy = sample.value.y;
		let sz = sample.value.z;
		let sw = sample.value.w;
		if (
			reference.x * sx +
				reference.y * sy +
				reference.z * sz +
				reference.w * sw <
			0
		) {
			sx = -sx;
			sy = -sy;
			sz = -sz;
			sw = -sw;
		}
		x += sx * sample.weight;
		y += sy * sample.weight;
		z += sz * sample.weight;
		w += sw * sample.weight;
		total += sample.weight;
	}
	if (total <= 1e-6) return new Quaternion();
	return new Quaternion(x / total, y / total, z / total, w / total).normalize();
}

function blendAdditiveQuaternion(
	samples: Array<{ value: Quaternion; weight: number }>
): Quaternion {
	let result = new Quaternion(0, 0, 0, 1);
	for (const sample of samples) {
		if (sample.weight <= 1e-6) continue;
		const w = Math.max(0, Math.min(1, sample.weight));
		const additive = Quaternion.slerp(
			new Quaternion(0, 0, 0, 1),
			sample.value,
			w
		);
		result = Quaternion.multiply(result, additive).normalize();
	}
	return result;
}
