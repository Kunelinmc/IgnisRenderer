import { AnimationClip, type GLTFAnimationBundle, KeyframeTrack } from "../animation";
import { Node } from "../core/Node";
import { NodeEntityPrefab } from "../ecs";
import type { EntityPrefab } from "../ecs";
import { Quaternion } from "../maths/Quaternion";
import type { IVector3 } from "../maths/types";
import { Loader, type LoaderEvents, type ParseProgressEvent } from "./Loader";

const DEGREE_TO_RADIAN = Math.PI / 180;
const DEFAULT_ROOT_NAME = "bvhRoot";
const DEFAULT_CLIP_NAME = "bvhClip";

type BVHChannelAxis = "X" | "Y" | "Z";

interface BVHChannel {
	type: "position" | "rotation";
	axis: BVHChannelAxis;
}

interface BVHEndSite {
	name: string;
	offset: IVector3;
}

interface BVHJoint {
	index: number;
	name: string;
	parentIndex: number | null;
	offset: IVector3;
	channels: BVHChannel[];
	channelStart: number;
	children: number[];
	endSites: BVHEndSite[];
}

interface ParsedBVHData {
	rootJointIndex: number;
	joints: BVHJoint[];
	totalChannelCount: number;
	frameCount: number;
	frameTimeSeconds: number;
	motionValues: Float32Array;
}

interface BVHHierarchyBuildResult {
	root: Node;
	nodeByJointIndex: Map<number, Node>;
	pathByJointIndex: Map<number, string>;
}

export interface BVHLoaderEvents extends LoaderEvents {
	load: [Node];
	parsestart: [];
	parseend: [Node];
	loadprefab: [EntityPrefab];
	loadanimation: [GLTFAnimationBundle];
}

export interface BVHParseOptions {
	rootName?: string;
	clipName?: string;
}

/**
 * BVHLoader parses Biovision Hierarchy (.bvh) files and builds a Node hierarchy
 * with an AnimationClip compatible with the existing animation runtime.
 */
export class BVHLoader extends Loader<BVHLoaderEvents> {
	private _lastAnimationBundle: GLTFAnimationBundle | null = null;

	constructor() {
		super();
	}

	/**
	 * Returns the latest parsed animation bundle.
	 */
	public getLastAnimationBundle(): GLTFAnimationBundle | null {
		return this._lastAnimationBundle;
	}

	/**
	 * Clears the latest parsed animation bundle cache.
	 */
	public clearLastAnimationBundle(): void {
		this._lastAnimationBundle = null;
	}

	/**
	 * Loads a BVH file from URL and parses it into a Node hierarchy.
	 */
	public async load(url: string, options: BVHParseOptions = {}): Promise<Node> {
		try {
			const buffer = await this._fetchWithProgress(url);
			const text = decodeTextBuffer(buffer);
			const root = this.parse(text, {
				...options,
				clipName: options.clipName ?? extractClipNameFromURL(url) ?? DEFAULT_CLIP_NAME,
			});
			this.emit("load", root);
			return root;
		} catch (error) {
			this.emit("error", error);
			throw error;
		}
	}

	/**
	 * Loads a BVH file from URL and returns an EntityPrefab.
	 */
	public async loadPrefab(url: string, options: BVHParseOptions = {}): Promise<EntityPrefab> {
		try {
			const buffer = await this._fetchWithProgress(url);
			const text = decodeTextBuffer(buffer);
			const prefab = this.parsePrefab(text, {
				...options,
				clipName: options.clipName ?? extractClipNameFromURL(url) ?? DEFAULT_CLIP_NAME,
			});
			this.emit("loadprefab", prefab);
			return prefab;
		} catch (error) {
			this.emit("error", error);
			throw error;
		}
	}

	/**
	 * Parses BVH text/buffer into a Node hierarchy and animation bundle.
	 */
	public parse(data: string | ArrayBuffer, options: BVHParseOptions = {}): Node {
		this.emit("parsestart");
		this.emit("parseprogress", {
			current: 0,
			total: 3,
			message: "Parsing BVH hierarchy",
		} as ParseProgressEvent);

		const parsed = this._parseBVH(data);
		this.emit("parseprogress", {
			current: 1,
			total: 3,
			message: "Building BVH node hierarchy",
		} as ParseProgressEvent);

		const hierarchy = this._buildHierarchy(parsed, options.rootName ?? DEFAULT_ROOT_NAME);
		this.emit("parseprogress", {
			current: 2,
			total: 3,
			message: "Building BVH animation tracks",
		} as ParseProgressEvent);

		const clip = this._buildClip(
			parsed,
			hierarchy.pathByJointIndex,
			options.clipName ??
				sanitizePathSegment(
					parsed.joints[parsed.rootJointIndex]?.name ?? DEFAULT_CLIP_NAME,
				),
		);

		this._lastAnimationBundle = {
			clips: [clip],
			skeletons: [],
			morphBindings: [],
			nodePathMap: Object.fromEntries(
				Array.from(hierarchy.pathByJointIndex.entries()).map(([jointIndex, path]) => [
					path,
					hierarchy.nodeByJointIndex.get(jointIndex)?.id ?? "",
				]),
			),
		};
		this.emit("loadanimation", this._lastAnimationBundle);
		this.emit("parseend", hierarchy.root);
		return hierarchy.root;
	}

	/**
	 * Parses BVH input into an EntityPrefab.
	 */
	public parsePrefab(data: string | ArrayBuffer, options: BVHParseOptions = {}): EntityPrefab {
		const root = this.parse(data, options);
		return new NodeEntityPrefab(root, this._lastAnimationBundle);
	}

	private _parseBVH(data: string | ArrayBuffer): ParsedBVHData {
		const text = typeof data === "string" ? data : decodeTextBuffer(data);
		const reader = new BVHLineReader(text);
		const hierarchyLabel = reader.readRequiredLine("Expected BVH HIERARCHY section");
		if (hierarchyLabel.toUpperCase() !== "HIERARCHY") {
			throw new Error(`Invalid BVH: expected "HIERARCHY", got "${hierarchyLabel}"`);
		}

		let channelCursor = 0;
		const joints: BVHJoint[] = [];
		const rootLine = reader.readRequiredLine("Expected BVH ROOT declaration");
		const rootJointIndex = this._parseJoint(
			rootLine,
			null,
			joints,
			reader,
			() => channelCursor,
			(nextValue) => {
				channelCursor = nextValue;
			},
		);

		const motionLabel = reader.readRequiredLine("Expected BVH MOTION section");
		if (motionLabel.toUpperCase() !== "MOTION") {
			throw new Error(`Invalid BVH: expected "MOTION", got "${motionLabel}"`);
		}

		const framesLine = reader.readRequiredLine("Expected BVH Frames line");
		const frameCount = parseFramesLine(framesLine);
		const frameTimeLine = reader.readRequiredLine("Expected BVH Frame Time line");
		const frameTimeSeconds = parseFrameTimeLine(frameTimeLine);
		const motionValues = parseMotionValues(reader, frameCount, channelCursor);

		return {
			rootJointIndex,
			joints,
			totalChannelCount: channelCursor,
			frameCount,
			frameTimeSeconds,
			motionValues,
		};
	}

	private _parseJoint(
		line: string,
		parentIndex: number | null,
		joints: BVHJoint[],
		reader: BVHLineReader,
		getChannelCursor: () => number,
		setChannelCursor: (next: number) => void,
	): number {
		const declaration = parseJointDeclaration(line);
		const jointIndex = joints.length;
		const joint: BVHJoint = {
			index: jointIndex,
			name: declaration.name,
			parentIndex,
			offset: { x: 0, y: 0, z: 0 },
			channels: [],
			channelStart: getChannelCursor(),
			children: [],
			endSites: [],
		};
		joints.push(joint);

		this._ensureOpeningBrace(line, reader);
		let seenOffset = false;
		let seenChannels = false;
		let endSiteCount = 0;

		while (true) {
			const nextLine = reader.readRequiredLine(
				`Unexpected EOF while parsing joint "${joint.name}"`,
			);
			if (nextLine === "}") {
				break;
			}

			const upper = nextLine.toUpperCase();
			if (upper.startsWith("OFFSET")) {
				joint.offset = parseOffsetLine(nextLine);
				seenOffset = true;
				continue;
			}

			if (upper.startsWith("CHANNELS")) {
				if (seenChannels) {
					throw new Error(
						`Invalid BVH: duplicate CHANNELS line in joint "${joint.name}"`,
					);
				}
				joint.channels = parseChannelsLine(nextLine);
				joint.channelStart = getChannelCursor();
				setChannelCursor(getChannelCursor() + joint.channels.length);
				seenChannels = true;
				continue;
			}

			if (/^JOINT(\s|$)/i.test(nextLine)) {
				const childIndex = this._parseJoint(
					nextLine,
					jointIndex,
					joints,
					reader,
					getChannelCursor,
					setChannelCursor,
				);
				joint.children.push(childIndex);
				continue;
			}

			if (/^END\s+SITE(\s|$)/i.test(nextLine)) {
				endSiteCount++;
				const endSite = this._parseEndSite(
					nextLine,
					reader,
					`${joint.name}_end${endSiteCount > 1 ? `_${endSiteCount}` : ""}`,
				);
				joint.endSites.push(endSite);
				continue;
			}

			throw new Error(
				`Invalid BVH line in joint "${joint.name}": "${nextLine}" (line ${reader.lineNumber})`,
			);
		}

		if (!seenOffset) {
			throw new Error(`Invalid BVH: joint "${joint.name}" is missing OFFSET declaration`);
		}
		if (declaration.type === "ROOT" && !seenChannels) {
			throw new Error(
				`Invalid BVH: root joint "${joint.name}" is missing CHANNELS declaration`,
			);
		}

		return jointIndex;
	}

	private _parseEndSite(line: string, reader: BVHLineReader, name: string): BVHEndSite {
		this._ensureOpeningBrace(line, reader);
		let offset: IVector3 | null = null;
		while (true) {
			const nextLine = reader.readRequiredLine(
				`Unexpected EOF while parsing End Site "${name}"`,
			);
			if (nextLine === "}") break;
			const upper = nextLine.toUpperCase();
			if (upper.startsWith("OFFSET")) {
				offset = parseOffsetLine(nextLine);
				continue;
			}
			throw new Error(
				`Invalid BVH line in End Site "${name}": "${nextLine}" (line ${reader.lineNumber})`,
			);
		}
		if (!offset) {
			throw new Error(`Invalid BVH: End Site "${name}" is missing OFFSET`);
		}
		return { name, offset };
	}

	private _ensureOpeningBrace(line: string, reader: BVHLineReader): void {
		if (line.endsWith("{")) {
			return;
		}
		const braceLine = reader.readRequiredLine('Expected "{" in BVH hierarchy');
		if (braceLine !== "{") {
			throw new Error(
				`Invalid BVH: expected "{", got "${braceLine}" (line ${reader.lineNumber})`,
			);
		}
	}

	private _buildHierarchy(parsed: ParsedBVHData, rootName: string): BVHHierarchyBuildResult {
		const root = new Node({
			idPrefix: "node",
			name: rootName,
		});

		const nodeByJointIndex = new Map<number, Node>();
		const pathByJointIndex = new Map<number, string>();
		const siblingNameCounter = new Map<string, number>();

		const rootPath = `/${sanitizePathSegment(root.name)}`;
		const attachJoint = (jointIndex: number, parentNode: Node, parentPath: string): void => {
			const joint = parsed.joints[jointIndex];
			const node = new Node({
				idPrefix: "node",
				name: joint.name,
				position: {
					x: joint.offset.x,
					y: joint.offset.y,
					z: joint.offset.z,
				},
			});
			node.updateLocalMatrix();
			parentNode.addChild(node);

			const path = createUniquePath(
				parentPath,
				sanitizePathSegment(joint.name),
				siblingNameCounter,
			);
			nodeByJointIndex.set(jointIndex, node);
			pathByJointIndex.set(jointIndex, path);

			for (const childJointIndex of joint.children) {
				attachJoint(childJointIndex, node, path);
			}
			for (const endSite of joint.endSites) {
				const endNode = new Node({
					idPrefix: "node",
					name: endSite.name,
					position: {
						x: endSite.offset.x,
						y: endSite.offset.y,
						z: endSite.offset.z,
					},
				});
				endNode.updateLocalMatrix();
				node.addChild(endNode);
			}
		};

		attachJoint(parsed.rootJointIndex, root, rootPath);
		return {
			root,
			nodeByJointIndex,
			pathByJointIndex,
		};
	}

	private _buildClip(
		parsed: ParsedBVHData,
		pathByJointIndex: Map<number, string>,
		clipName: string,
	): AnimationClip {
		const frameCount = parsed.frameCount;
		const times = new Float32Array(frameCount);
		for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
			times[frameIndex] = frameIndex * parsed.frameTimeSeconds;
		}

		const tracks: KeyframeTrack[] = [];
		for (const joint of parsed.joints) {
			const path = pathByJointIndex.get(joint.index);
			if (!path) continue;

			const hasTranslation = joint.channels.some((channel) => channel.type === "position");
			const hasRotation = joint.channels.some((channel) => channel.type === "rotation");

			let translationValues: Float32Array | null = null;
			let rotationValues: Float32Array | null = null;

			if (hasTranslation) {
				translationValues = new Float32Array(frameCount * 3);
			}
			if (hasRotation) {
				rotationValues = new Float32Array(frameCount * 4);
			}

			for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
				const channelStart = frameIndex * parsed.totalChannelCount + joint.channelStart;
				let tx = joint.offset.x;
				let ty = joint.offset.y;
				let tz = joint.offset.z;
				let rotation = new Quaternion(0, 0, 0, 1);

				for (let channelIndex = 0; channelIndex < joint.channels.length; channelIndex++) {
					const channel = joint.channels[channelIndex];
					const value = parsed.motionValues[channelStart + channelIndex];
					if (channel.type === "position") {
						if (channel.axis === "X") tx = joint.offset.x + value;
						else if (channel.axis === "Y") ty = joint.offset.y + value;
						else tz = joint.offset.z + value;
						continue;
					}

					rotation = Quaternion.multiply(
						rotation,
						createAxisQuaternion(channel.axis, value * DEGREE_TO_RADIAN),
					);
				}

				if (translationValues) {
					const offset = frameIndex * 3;
					translationValues[offset] = tx;
					translationValues[offset + 1] = ty;
					translationValues[offset + 2] = tz;
				}

				if (rotationValues) {
					rotation.normalize();
					const offset = frameIndex * 4;
					rotationValues[offset] = rotation.x;
					rotationValues[offset + 1] = rotation.y;
					rotationValues[offset + 2] = rotation.z;
					rotationValues[offset + 3] = rotation.w;
				}
			}

			if (translationValues) {
				tracks.push(
					new KeyframeTrack({
						name: `${joint.name}:translation`,
						binding: {
							targetType: "node",
							targetPath: path,
							property: "translation",
						},
						times,
						values: translationValues,
						valueSize: 3,
						interpolation: "linear",
					}),
				);
			}

			if (rotationValues) {
				tracks.push(
					new KeyframeTrack({
						name: `${joint.name}:rotation`,
						binding: {
							targetType: "node",
							targetPath: path,
							property: "rotation",
						},
						times,
						values: rotationValues,
						valueSize: 4,
						interpolation: "linear",
					}),
				);
			}
		}

		return new AnimationClip({
			name: clipName.length > 0 ? clipName : DEFAULT_CLIP_NAME,
			duration: frameCount > 0 ? times[frameCount - 1] : 0,
			tracks,
		});
	}
}

class BVHLineReader {
	private readonly _lines: string[];
	private _index = 0;

	constructor(text: string) {
		this._lines = text.split(/\r?\n/);
	}

	public get lineNumber(): number {
		return Math.max(1, this._index);
	}

	public readRequiredLine(errorMessage: string): string {
		const line = this.readLine();
		if (line === null) {
			throw new Error(`${errorMessage} (line ${this.lineNumber})`);
		}
		return line;
	}

	public readLine(): string | null {
		while (this._index < this._lines.length) {
			const line = this._lines[this._index++].trim().replace(/^\uFEFF/, "");
			if (line.length === 0) {
				continue;
			}
			return line;
		}
		return null;
	}
}

function parseJointDeclaration(line: string): { type: "ROOT" | "JOINT"; name: string } {
	const normalized = line.endsWith("{") ? line.slice(0, -1).trim() : line;
	const match = /^(ROOT|JOINT)\s+(.+)$/i.exec(normalized);
	if (!match) {
		throw new Error(`Invalid BVH joint declaration: "${line}"`);
	}
	return {
		type: match[1].toUpperCase() as "ROOT" | "JOINT",
		name: match[2].trim(),
	};
}

function parseOffsetLine(line: string): IVector3 {
	const parts = line.split(/\s+/);
	if (parts.length < 4) {
		throw new Error(`Invalid BVH OFFSET line: "${line}"`);
	}
	return {
		x: parseFiniteNumber(parts[1], "OFFSET X"),
		y: parseFiniteNumber(parts[2], "OFFSET Y"),
		z: parseFiniteNumber(parts[3], "OFFSET Z"),
	};
}

function parseChannelsLine(line: string): BVHChannel[] {
	const parts = line.split(/\s+/);
	if (parts.length < 2) {
		throw new Error(`Invalid BVH CHANNELS line: "${line}"`);
	}
	const count = parseFiniteNumber(parts[1], "CHANNELS count");
	if (!Number.isInteger(count) || count < 0) {
		throw new Error(`Invalid BVH CHANNELS count: "${parts[1]}"`);
	}
	const values = parts.slice(2);
	if (values.length < count) {
		throw new Error(
			`Invalid BVH CHANNELS line: expected ${count} channels, got ${values.length}`,
		);
	}

	const channels: BVHChannel[] = [];
	for (let i = 0; i < count; i++) {
		const raw = values[i];
		const parsed = parseChannelToken(raw);
		channels.push(parsed);
	}
	return channels;
}

function parseChannelToken(token: string): BVHChannel {
	const normalized = token.trim().toUpperCase();
	const positionMatch = /^(X|Y|Z)POSITION$/.exec(normalized);
	if (positionMatch) {
		return {
			type: "position",
			axis: positionMatch[1] as BVHChannelAxis,
		};
	}
	const rotationMatch = /^(X|Y|Z)ROTATION$/.exec(normalized);
	if (rotationMatch) {
		return {
			type: "rotation",
			axis: rotationMatch[1] as BVHChannelAxis,
		};
	}
	throw new Error(`Unsupported BVH channel token: "${token}"`);
}

function parseFramesLine(line: string): number {
	const match = /^FRAMES:\s*(\d+)\s*$/i.exec(line);
	if (!match) {
		throw new Error(`Invalid BVH Frames line: "${line}"`);
	}
	const value = Number.parseInt(match[1], 10);
	if (!Number.isSafeInteger(value) || value < 0) {
		throw new Error(`Invalid BVH frame count: "${match[1]}"`);
	}
	return value;
}

function parseFrameTimeLine(line: string): number {
	const match = /^FRAME\s+TIME:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$/i.exec(line);
	if (!match) {
		throw new Error(`Invalid BVH Frame Time line: "${line}"`);
	}
	const value = Number.parseFloat(match[1]);
	if (!Number.isFinite(value) || value < 0) {
		throw new Error(`Invalid BVH frame time: "${match[1]}"`);
	}
	return value;
}

function parseMotionValues(
	reader: BVHLineReader,
	frameCount: number,
	channelCount: number,
): Float32Array {
	const expectedValueCount = frameCount * channelCount;
	if (expectedValueCount === 0) {
		return new Float32Array(0);
	}

	const values = new Float32Array(expectedValueCount);
	let cursor = 0;
	while (cursor < expectedValueCount) {
		const line = reader.readLine();
		if (line === null) break;
		const parts = line.split(/\s+/);
		for (const part of parts) {
			if (part.length === 0) continue;
			if (cursor >= expectedValueCount) {
				break;
			}
			values[cursor++] = parseFiniteNumber(part, "motion value");
		}
	}

	if (cursor < expectedValueCount) {
		throw new Error(
			`Invalid BVH motion data: expected ${expectedValueCount} values, got ${cursor}`,
		);
	}
	return values;
}

function parseFiniteNumber(token: string, label: string): number {
	const value = Number(token);
	if (!Number.isFinite(value)) {
		throw new Error(`Invalid ${label}: "${token}"`);
	}
	return value;
}

function sanitizePathSegment(value: string): string {
	return value.replace(/[^\w\-]+/g, "_");
}

function createUniquePath(
	parentPath: string,
	baseSegment: string,
	counter: Map<string, number>,
): string {
	const safeBase = baseSegment.length > 0 ? baseSegment : "joint";
	const key = `${parentPath}|${safeBase}`;
	const current = counter.get(key) ?? 0;
	counter.set(key, current + 1);
	const suffix = current === 0 ? "" : `_${current + 1}`;
	return `${parentPath}/${safeBase}${suffix}`;
}

function createAxisQuaternion(axis: BVHChannelAxis, angle: number): Quaternion {
	if (axis === "X") {
		return Quaternion.fromAxisAngle({ x: 1, y: 0, z: 0 }, angle);
	}
	if (axis === "Y") {
		return Quaternion.fromAxisAngle({ x: 0, y: 1, z: 0 }, angle);
	}
	return Quaternion.fromAxisAngle({ x: 0, y: 0, z: 1 }, angle);
}

function decodeTextBuffer(buffer: ArrayBuffer): string {
	return new TextDecoder("utf-8").decode(new Uint8Array(buffer));
}

function extractClipNameFromURL(url: string): string | null {
	const trimmed = url.trim();
	if (!trimmed) {
		return null;
	}
	const withoutQuery = trimmed.split(/[?#]/, 1)[0];
	const lastSlash = withoutQuery.lastIndexOf("/");
	const filename = lastSlash >= 0 ? withoutQuery.slice(lastSlash + 1) : withoutQuery;
	if (!filename) {
		return null;
	}
	const withoutExtension = filename.replace(/\.[^.]+$/, "");
	if (!withoutExtension) {
		return null;
	}
	return sanitizePathSegment(withoutExtension);
}
