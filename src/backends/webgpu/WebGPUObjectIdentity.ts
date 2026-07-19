import { getWebGPUObjectResourceHandle } from "./WebGPUResourceHandle";

const HASH64_OFFSET_BASIS = 0xcbf29ce484222325n;
const HASH64_PRIME = 0x100000001b3n;
const HASH64_MASK = 0xffffffffffffffffn;

export const WEBGPU_RESOURCE_ID_REBASE_THRESHOLD = 0x40000000;

export class WebGPUObjectIdentity {
	private _resourceIds = new WeakMap<object, number>();
	private _nextResourceId = 1;

	public constructor(private readonly _onRebase?: () => void) {}

	public reset(): void {
		this._resourceIds = new WeakMap<object, number>();
		this._nextResourceId = 1;
	}

	public getObjectId(value: object): number {
		let id = this._resourceIds.get(value);
		if (id !== undefined) {
			return id;
		}
		if (this._nextResourceId >= WEBGPU_RESOURCE_ID_REBASE_THRESHOLD) {
			this.rebase();
		}
		id = this._nextResourceId++;
		this._resourceIds.set(value, id);
		return id;
	}

	public getCacheToken(value: unknown): string {
		if (value === null) {
			return "null";
		}
		if (value === undefined) {
			return "undefined";
		}

		const type = typeof value;
		if (type === "string" || type === "number" || type === "boolean") {
			return `${type}:${String(value)}`;
		}
		if (type === "bigint") {
			return `bigint:${String(value)}`;
		}
		if (type === "symbol") {
			return `symbol:${String(value)}`;
		}
		if (type === "function" || type === "object") {
			const backendHandle = getWebGPUObjectResourceHandle(value);
			if (backendHandle) {
				return `obj:${this.getObjectId(backendHandle)}`;
			}
			return `obj:${this.getObjectId(value as object)}`;
		}
		return `${type}:${String(value)}`;
	}

	public rebase(): void {
		this.reset();
		this._onRebase?.();
	}
}

export function hashString64(value: string): bigint {
	let hash = HASH64_OFFSET_BASIS;
	for (let i = 0; i < value.length; i++) {
		hash = hash64Combine(hash, value.charCodeAt(i));
	}
	return hash;
}

export function hash64Combine(hash: bigint, value: number): bigint {
	const normalized = BigInt(value >>> 0);
	return ((hash ^ normalized) * HASH64_PRIME) & HASH64_MASK;
}

export function createHash64(): bigint {
	return HASH64_OFFSET_BASIS;
}
