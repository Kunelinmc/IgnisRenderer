import { AmmoPhysicsAdapter } from "../adapters/AmmoPhysicsAdapter";
import {
	postMessageWorkerTransportPlugin,
	sharedArrayBufferWorkerTransportPlugin,
} from "../../workers/transports";
import type { WorkerTransportPlugin } from "../../workers/types";
import type { AmmoWorkerCommand, AmmoWorkerTaskPayload } from "./ammoWorkerProtocol";

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

let adapter: AmmoPhysicsAdapter | null = null;
let initialized = false;

function decodeEnvelope(
	data: unknown
): {
	plugin: WorkerTransportPlugin;
	envelope: WorkerEnvelope<AmmoWorkerTaskPayload>;
} | null {
	for (const plugin of transportPlugins) {
		const envelope = plugin.decodeTask(data);
		if (!envelope) continue;
		return {
			plugin,
			envelope: envelope as WorkerEnvelope<AmmoWorkerTaskPayload>,
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

async function ensureAdapter(strict: boolean): Promise<AmmoPhysicsAdapter> {
	if (!adapter) {
		adapter = new AmmoPhysicsAdapter({
			strict,
		});
	}
	if (!initialized) {
		await adapter.init();
		initialized = true;
	}
	return adapter;
}

function applyCommand(ammoAdapter: AmmoPhysicsAdapter, command: AmmoWorkerCommand): void {
	switch (command.type) {
		case "createWorld":
			ammoAdapter.createWorld(command.config);
			return;
		case "destroyWorld":
			ammoAdapter.destroyWorld(command.worldId);
			return;
		case "createBody":
			ammoAdapter.createBody(
				command.worldId,
				command.bodyId,
				command.descriptor,
				command.initialTransform
			);
			return;
		case "destroyBody":
			ammoAdapter.destroyBody(command.worldId, command.bodyId);
			return;
		case "setBodyTransform":
			ammoAdapter.setBodyTransform(
				command.worldId,
				command.bodyId,
				command.transform
			);
			return;
		case "setBodyLinearVelocity":
			ammoAdapter.setBodyLinearVelocity(
				command.worldId,
				command.bodyId,
				command.velocity
			);
			return;
		case "setAngularVelocity":
			ammoAdapter.setAngularVelocity(
				command.worldId,
				command.bodyId,
				command.velocity
			);
			return;
		case "applyForce":
			ammoAdapter.applyForce(command.worldId, command.bodyId, command.force);
			return;
		case "applyTorque":
			ammoAdapter.applyTorque(command.worldId, command.bodyId, command.torque);
			return;
		case "applyImpulse":
			ammoAdapter.applyImpulse(command.worldId, command.bodyId, command.impulse);
			return;
		case "addCollider":
			ammoAdapter.addCollider(
				command.worldId,
				command.bodyId,
				command.colliderId,
				command.descriptor,
				command.shape
			);
			return;
		case "destroyCollider":
			ammoAdapter.destroyCollider(command.worldId, command.colliderId);
			return;
		case "setColliderSensor":
			ammoAdapter.setColliderSensor(
				command.worldId,
				command.colliderId,
				command.isSensor
			);
			return;
		case "setCollisionMask":
			ammoAdapter.setCollisionMask(
				command.worldId,
				command.colliderId,
				command.mask
			);
			return;
		case "setColliderMaterial":
			ammoAdapter.setColliderMaterial(
				command.worldId,
				command.colliderId,
				command.material
			);
			return;
		case "createJoint":
			ammoAdapter.createJoint(command.worldId, command.jointId, command.descriptor);
			return;
		case "destroyJoint":
			ammoAdapter.destroyJoint(command.worldId, command.jointId);
			return;
		case "createCharacterController":
			ammoAdapter.createCharacterController(
				command.worldId,
				command.controllerId,
				command.descriptor
			);
			return;
		case "destroyCharacterController":
			ammoAdapter.destroyCharacterController(command.worldId, command.controllerId);
			return;
		case "jumpCharacterController":
			ammoAdapter.jumpCharacterController(
				command.worldId,
				command.controllerId,
				command.speed
			);
			return;
		case "setCharacterControllerMaxSlope":
			ammoAdapter.setCharacterControllerMaxSlope(
				command.worldId,
				command.controllerId,
				command.value
			);
			return;
		case "setCharacterControllerStepHeight":
			ammoAdapter.setCharacterControllerStepHeight(
				command.worldId,
				command.controllerId,
				command.value
			);
			return;
		default: {
			const exhaustiveCheck: never = command;
			throw new Error(`Unhandled Ammo worker command: ${String(exhaustiveCheck)}`);
		}
	}
}

async function executeTask(payload: AmmoWorkerTaskPayload): Promise<unknown> {
	if (payload.type === "init") {
		await ensureAdapter(payload.strict);
		return { initialized: true };
	}

	const ammoAdapter = await ensureAdapter(false);
	for (const command of payload.commands) {
		applyCommand(ammoAdapter, command);
	}

	const request = payload.request;
	if (!request) return null;

	switch (request.type) {
		case "stepWorld":
			return ammoAdapter.stepWorld(request.worldId, request.deltaSeconds);
		case "moveCharacterController":
			return ammoAdapter.moveCharacterController(
				request.worldId,
				request.controllerId,
				request.direction,
				request.deltaSeconds
			);
		case "isCharacterControllerGrounded":
			return ammoAdapter.isCharacterControllerGrounded(
				request.worldId,
				request.controllerId
			);
		case "raycast":
			return ammoAdapter.raycast(request.worldId, request.query);
		case "raycastAll":
			return ammoAdapter.raycastAll(request.worldId, request.query);
		case "sphereCast":
			return ammoAdapter.sphereCast(request.worldId, request.query);
		case "boxCast":
			return ammoAdapter.boxCast(request.worldId, request.query);
		case "overlapSphere":
			return ammoAdapter.overlapSphere(request.worldId, request.query);
		case "overlapBox":
			return ammoAdapter.overlapBox(request.worldId, request.query);
		default: {
			const exhaustiveCheck: never = request;
			throw new Error(`Unhandled Ammo worker request: ${String(exhaustiveCheck)}`);
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
