import {
	DRAW_PACKET_FLAG_REFLECTIVE,
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	defineTransientKey,
	type DrawPacket,
	type FrameContext,
} from "./types";

/** @internal Identifies the camera view for which frame packets are prepared. */
export type FramePacketViewPurpose =
	| "main"
	| "probe-capture"
	| "planar-reflection";

/** @internal Context supplied to one frame-packet contributor. */
export interface FramePacketContributorContext {
	readonly frameContext: FrameContext;
	readonly purpose: FramePacketViewPurpose;
}

/** @internal Accepts packets produced by a frame-packet contributor. */
export interface FramePacketSink {
	add(packet: DrawPacket): void;
}

/** @internal Produces supplemental logical draw packets for one frame view. */
export interface FramePacketContributor {
	readonly id: string;
	supports(context: FramePacketContributorContext): boolean;
	contribute(context: FramePacketContributorContext, sink: FramePacketSink): void;
}

/** @internal Complete draw-packet lists for one prepared frame view. */
export interface PreparedFramePacketSet {
	readonly all: readonly DrawPacket[];
	readonly opaque: readonly DrawPacket[];
	readonly transparent: readonly DrawPacket[];
	readonly shadowCasters: readonly DrawPacket[];
	readonly shadowTransmitters: readonly DrawPacket[];
	readonly reflective: readonly DrawPacket[];
}

/** @internal Narrow frame-packet preparation capability for backend runtimes. */
export interface FramePacketProvider {
	createBaseline(context: FrameContext): PreparedFramePacketSet;
	prepare(
		context: FrameContext,
		purpose: FramePacketViewPurpose,
	): PreparedFramePacketSet;
}

/** @internal Creates a packet set containing only a prepared scene's packets. */
export function createBaselineFramePacketSet(
	context: FrameContext,
): PreparedFramePacketSet {
	return {
		all: [
			...context.scene.opaquePackets,
			...context.scene.transparentPackets,
		],
		opaque: context.scene.opaquePackets.slice(),
		transparent: context.scene.transparentPackets.slice(),
		shadowCasters: context.scene.shadowCasterPackets.slice(),
		shadowTransmitters: context.scene.shadowTransmitterPackets.slice(),
		reflective: context.scene.reflectivePackets.slice(),
	};
}

interface PreparedFramePacketCacheRecord {
	readonly registry: FramePacketContributorRegistry;
	readonly scene: FrameContext["scene"];
	readonly viewCamera: FrameContext["viewCamera"];
	readonly purpose: FramePacketViewPurpose;
	readonly packets: PreparedFramePacketSet;
}

const PREPARED_FRAME_PACKET_SET_KEY =
	defineTransientKey<PreparedFramePacketCacheRecord>(
		"pipeline:prepared-frame-packet-set",
	);

/**
 * Composes baseline prepared-scene packets with registered contributors.
 *
 * @internal Backend runtimes own one registry instance and must register every
 * contributor before the first call to `prepare()`.
 */
export class FramePacketContributorRegistry implements FramePacketProvider {
	private readonly _contributors: FramePacketContributor[] = [];
	private _sealed = false;

	/** Registers one contributor before packet preparation begins. */
	public register(contributor: FramePacketContributor): void {
		if (this._sealed) {
			throw new Error(
				"Frame packet contributors cannot be registered after preparation begins.",
			);
		}
		if (this._contributors.some((existing) => existing.id === contributor.id)) {
			throw new Error(`Frame packet contributor "${contributor.id}" is already registered.`);
		}
		this._contributors.push(contributor);
	}

	/** Creates an uncached baseline set for warmup or scene-only preparation. */
	public createBaseline(context: FrameContext): PreparedFramePacketSet {
		return createBaselineFramePacketSet(context);
	}

	/** Prepares and caches the effective packet lists for one frame view. */
	public prepare(
		context: FrameContext,
		purpose: FramePacketViewPurpose,
	): PreparedFramePacketSet {
		this._sealed = true;
		const cached = context.transient.get(PREPARED_FRAME_PACKET_SET_KEY);
		if (
			cached &&
			cached.registry === this &&
			cached.scene === context.scene &&
			cached.viewCamera === context.viewCamera &&
			cached.purpose === purpose
		) {
			return cached.packets;
		}

		const all: DrawPacket[] = [
			...context.scene.opaquePackets,
			...context.scene.transparentPackets,
		];
		const opaque = context.scene.opaquePackets.slice();
		const transparent = context.scene.transparentPackets.slice();
		const shadowCasters = context.scene.shadowCasterPackets.slice();
		const shadowTransmitters = context.scene.shadowTransmitterPackets.slice();
		const reflective = context.scene.reflectivePackets.slice();
		const contributorContext: FramePacketContributorContext = {
			frameContext: context,
			purpose,
		};
		const sink: FramePacketSink = {
			add: (packet) => {
				all.push(packet);
				const flags = packet.passFlags;
				if ((flags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0) {
					transparent.push(packet);
				} else {
					opaque.push(packet);
				}
				if ((flags & DRAW_PACKET_FLAG_SHADOW_CASTER) !== 0) {
					shadowCasters.push(packet);
				}
				if ((flags & DRAW_PACKET_FLAG_SHADOW_TRANSMITTER) !== 0) {
					shadowTransmitters.push(packet);
				}
				if ((flags & DRAW_PACKET_FLAG_REFLECTIVE) !== 0) {
					reflective.push(packet);
				}
			},
		};

		for (const contributor of this._contributors) {
			if (contributor.supports(contributorContext)) {
				contributor.contribute(contributorContext, sink);
			}
		}

		const packets: PreparedFramePacketSet = {
			all,
			opaque,
			transparent,
			shadowCasters,
			shadowTransmitters,
			reflective,
		};
		context.transient.set(PREPARED_FRAME_PACKET_SET_KEY, {
			registry: this,
			scene: context.scene,
			viewCamera: context.viewCamera,
			purpose,
			packets,
		});
		return packets;
	}
}
