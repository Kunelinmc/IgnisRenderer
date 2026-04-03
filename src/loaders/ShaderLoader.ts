import { Platform } from "../foundation/Platform";
import {
	createInlineCompositeShaderSource,
	type CompositeShaderSource,
	type ShaderSourceSegmentKind,
} from "../shaders/runtime";

export interface RawShaderModule {
	default: string;
}

export interface ShaderLoadRequest {
	key: string;
	nodeRelativePath: string;
	nodeBaseUrl: string;
	browserLoader: () => Promise<string>;
	segmentKind?: ShaderSourceSegmentKind;
}

type NodeFsModule = {
	readFile: (
		path: string | URL,
		options?: string | { encoding?: string }
	) => Promise<string>;
};

export class ShaderLoader {
	private _rawCache = new Map<string, Promise<string>>();
	private _compositeCache = new Map<string, Promise<CompositeShaderSource>>();

	public loadSource(request: Readonly<ShaderLoadRequest>): Promise<string> {
		let cached = this._rawCache.get(request.key);
		if (!cached) {
			cached = this.loadComposite(request).then((composite) => composite.code);
			this._rawCache.set(request.key, cached);
		}
		return cached;
	}

	public loadComposite(
		request: Readonly<ShaderLoadRequest>
	): Promise<CompositeShaderSource> {
		let cached = this._compositeCache.get(request.key);
		if (!cached) {
			cached = this._readShaderCode(request).then((code) =>
				createInlineCompositeShaderSource(
					code,
					request.nodeRelativePath,
					request.segmentKind ?? "source"
				)
			);
			this._compositeCache.set(request.key, cached);
		}
		return cached;
	}

	private async _readShaderCode(
		request: Readonly<ShaderLoadRequest>
	): Promise<string> {
		if (!Platform.isNodeRuntime()) {
			return request.browserLoader();
		}

		const fsSpecifier = ["node", "fs/promises"].join(":");
		const fsModule = (await import(/* @vite-ignore */ fsSpecifier)) as NodeFsModule;
		return fsModule.readFile(
			new URL(request.nodeRelativePath, request.nodeBaseUrl),
			"utf8"
		);
	}
}

export function fromRawShaderModuleLoader(
	loader: () => Promise<RawShaderModule>
): () => Promise<string> {
	return () => loader().then((module) => module.default);
}
