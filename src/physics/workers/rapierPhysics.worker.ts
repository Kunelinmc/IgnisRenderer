import { RapierPhysicsAdapter } from "../adapters/RapierPhysicsAdapter";
import {
	postMessageWorkerTransportPlugin,
	sharedArrayBufferWorkerTransportPlugin,
} from "../../workers/transports";
import type { WorkerTransportPlugin } from "../../workers/types";
import type { RapierWorkerCommand, RapierWorkerTaskPayload } from "./rapierWorkerProtocol";

interface WorkerEnvelope<TPayload> {
	id: number;
	payload: TPayload;
}

interface WorkerResultEnvelope<TResult> {
	id: number;
	result?: TResult;
	error?: string;
}

const transportPlugins: WorkerTransportPlugin[] = [
	sharedArrayBufferWorkerTransportPlugin,
	postMessageWorkerTransportPlugin,
];

type WorkerScopeLike = typeof globalThis & {
	postMessage(message: unknown, transfer?: Transferable[]): void;
	onmessage: ((event: { data: unknown }) => void) | null;
};

const workerScope = globalThis as WorkerScopeLike;

let adapter: RapierPhysicsAdapter | null = null;
let initialized = false;

function decodeEnvelope(
	data: unknown
): {
	plugin: WorkerTransportPlugin;
	envelope: WorkerEnvelope<RapierWorkerTaskPayload>;
} | null {
	for (const plugin of transportPlugins) {
		const envelope = plugin.decodeTask(data);
		if (!envelope) continue;
		return {
			plugin,
			envelope: envelope as WorkerEnvelope<RapierWorkerTaskPayload>,
		};
	}
	return null;
}

function encodeResult(
	plugin: WorkerTransportPlugin,
	envelope: WorkerResultEnvelope<unknown>
): {
	message: unknown;
	transfer?: Transferable[];
} {
	return plugin.encodeResult(envelope);
}

async function ensureAdapter(strict: boolean): Promise<RapierPhysicsAdapter> {
	if (!adapter) {
		adapter = new RapierPhysicsAdapter({
			strict,
		});
	}
	if (!initialized) {
		await adapter.init();
		initialized = true;
	}
	return adapter;
}

function applyCommand(
	rapierAdapter: RapierPhysicsAdapter,
	command: RapierWorkerCommand
): void {
	switch (command.type) {
		case "createWorld":
			rapierAdapter.createWorld(command.config);
			return;
		case "destroyWorld":
			rapierAdapter.destroyWorld(command.worldId);
			return;
		case "createBody":
			rapierAdapter.createBody(
				command.worldId,
				command.bodyId,
				command.descriptor,
				command.initialTransform
			);
			return;
		case "destroyBody":
			rapierAdapter.destroyBody(command.worldId, command.bodyId);
			return;
		case "setBodyTransform":
			rapierAdapter.setBodyTransform(
				command.worldId,
				command.bodyId,
				command.transform
			);
			return;
		case "setBodyLinearVelocity":
			rapierAdapter.setBodyLinearVelocity(
				command.worldId,
				command.bodyId,
				command.velocity
			);
			return;
		case "setAngularVelocity":
			rapierAdapter.setAngularVelocity(
				command.worldId,
				command.bodyId,
				command.velocity
			);
			return;
		case "setBodyType":
			rapierAdapter.setBodyType(
				command.worldId,
				command.bodyId,
				command.bodyType
			);
			return;
		case "setBodyMass":
			rapierAdapter.setBodyMass(command.worldId, command.bodyId, command.mass);
			return;
		case "setBodyGravityScale":
			rapierAdapter.setBodyGravityScale(
				command.worldId,
				command.bodyId,
				command.scale
			);
			return;
		case "setBodyLinearDamping":
			rapierAdapter.setBodyLinearDamping(
				command.worldId,
				command.bodyId,
				command.value
			);
			return;
		case "setBodyAngularDamping":
			rapierAdapter.setBodyAngularDamping(
				command.worldId,
				command.bodyId,
				command.value
			);
			return;
		case "wakeUpBody":
			rapierAdapter.wakeUpBody(command.worldId, command.bodyId);
			return;
		case "applyForce":
			rapierAdapter.applyForce(command.worldId, command.bodyId, command.force);
			return;
		case "applyTorque":
			rapierAdapter.applyTorque(command.worldId, command.bodyId, command.torque);
			return;
		case "applyImpulse":
			rapierAdapter.applyImpulse(
				command.worldId,
				command.bodyId,
				command.impulse
			);
			return;
		case "addCollider":
			rapierAdapter.addCollider(
				command.worldId,
				command.bodyId,
				command.colliderId,
				command.descriptor,
				command.shape
			);
			return;
		case "destroyCollider":
			rapierAdapter.destroyCollider(command.worldId, command.colliderId);
			return;
		case "setColliderSensor":
			rapierAdapter.setColliderSensor(
				command.worldId,
				command.colliderId,
				command.isSensor
			);
			return;
		case "setColliderCollisionFilter":
			rapierAdapter.setColliderCollisionFilter(
				command.worldId,
				command.colliderId,
				command.filter
			);
			return;
		case "setColliderMaterial":
			rapierAdapter.setColliderMaterial(
				command.worldId,
				command.colliderId,
				command.material
			);
			return;
		case "createJoint":
			rapierAdapter.createJoint(
				command.worldId,
				command.jointId,
				command.descriptor
			);
			return;
		case "destroyJoint":
			rapierAdapter.destroyJoint(command.worldId, command.jointId);
			return;
		case "createCharacterController":
			rapierAdapter.createCharacterController(
				command.worldId,
				command.controllerId,
				command.descriptor
			);
			return;
		case "moveCharacterController":
			rapierAdapter.moveCharacterController(
				command.worldId,
				command.controllerId,
				command.direction,
				command.deltaSeconds
			);
			return;
		case "destroyCharacterController":
			rapierAdapter.destroyCharacterController(
				command.worldId,
				command.controllerId
			);
			return;
		case "jumpCharacterController":
			rapierAdapter.jumpCharacterController(
				command.worldId,
				command.controllerId,
				command.speed
			);
			return;
		case "setCharacterControllerMaxSlope":
			rapierAdapter.setCharacterControllerMaxSlope(
				command.worldId,
				command.controllerId,
				command.value
			);
			return;
		case "setCharacterControllerStepHeight":
			rapierAdapter.setCharacterControllerStepHeight(
				command.worldId,
				command.controllerId,
				command.value
			);
			return;
		default: {
			const exhaustiveCheck: never = command;
			throw new Error(
				`Unhandled Rapier worker command: ${String(exhaustiveCheck)}`
			);
		}
	}
}

async function executeTask(payload: RapierWorkerTaskPayload): Promise<unknown> {
	if (payload.type === "init") {
		await ensureAdapter(payload.strict);
		return { initialized: true };
	}

	const rapierAdapter = await ensureAdapter(false);
	for (const command of payload.commands) {
		applyCommand(rapierAdapter, command);
	}

	const request = payload.request;
	if (!request) return null;

	switch (request.type) {
		case "stepWorld":
			return rapierAdapter.stepWorld(request.worldId, request.deltaSeconds);
		case "moveCharacterController":
			return rapierAdapter.moveCharacterController(
				request.worldId,
				request.controllerId,
				request.direction,
				request.deltaSeconds
			);
		case "isCharacterControllerGrounded":
			return rapierAdapter.isCharacterControllerGrounded(
				request.worldId,
				request.controllerId
			);
		case "raycast":
			return rapierAdapter.raycast(request.worldId, request.query);
		case "raycastAll":
			return rapierAdapter.raycastAll(request.worldId, request.query);
		case "sphereCast":
			return rapierAdapter.sphereCast(request.worldId, request.query);
		case "boxCast":
			return rapierAdapter.boxCast(request.worldId, request.query);
		case "overlapSphere":
			return rapierAdapter.overlapSphere(request.worldId, request.query);
		case "overlapBox":
			return rapierAdapter.overlapBox(request.worldId, request.query);
		default: {
			const exhaustiveCheck: never = request;
			throw new Error(
				`Unhandled Rapier worker request: ${String(exhaustiveCheck)}`
			);
		}
	}
}

workerScope.onmessage = (event) => {
	void (async () => {
		const decoded = decodeEnvelope(event.data);
		if (!decoded) return;

		const { plugin, envelope } = decoded;
		try {
			const result = await executeTask(envelope.payload);
			const encoded = encodeResult(plugin, {
				id: envelope.id,
				result,
			});
			workerScope.postMessage(
				encoded.message,
				encoded.transfer && encoded.transfer.length > 0 ?
					encoded.transfer
				:	undefined
			);
		} catch (error) {
			const encoded = encodeResult(plugin, {
				id: envelope.id,
				error: error instanceof Error ? error.message : String(error),
			});
			workerScope.postMessage(
				encoded.message,
				encoded.transfer && encoded.transfer.length > 0 ?
					encoded.transfer
				:	undefined
			);
		}
	})();
};
