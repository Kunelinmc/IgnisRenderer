import type { CubeTexture } from "../../core/CubeTexture";
import type { Texture } from "../../core/Texture";

/**
 * Runtime output bindings for captured-scene probe textures.
 */
export class ProbeCaptureController {
	protected _rawTexture: Texture | null = null;
	protected _cubeTexture: CubeTexture | null = null;

	/**
	 * Equirectangular HDR texture updated after successful probe capture.
	 */
	public get rawTexture(): Texture | null {
		return this._rawTexture;
	}

	/**
	 * Cubemap HDR texture updated from captured probe faces.
	 */
	public get cubeTexture(): CubeTexture | null {
		return this._cubeTexture;
	}

	/**
	 * Binds a user-owned texture for raw equirectangular capture output.
	 */
	public bindRawTexture(texture: Texture | null): this {
		this._rawTexture = texture;
		return this;
	}

	/**
	 * Binds a user-owned cubemap texture for captured face output.
	 */
	public bindCubeTexture(texture: CubeTexture | null): this {
		this._cubeTexture = texture;
		return this;
	}

	/**
	 * Removes all output bindings without mutating previously written textures.
	 */
	public clearOutputs(): this {
		this._rawTexture = null;
		this._cubeTexture = null;
		return this;
	}
}

/**
 * Runtime output bindings for captured reflection probes.
 */
export class ReflectionProbeCaptureController extends ProbeCaptureController {
	private _prefilteredTexture: Texture | null = null;

	/**
	 * Prefiltered HDR texture updated after reflection probe capture.
	 */
	public get prefilteredTexture(): Texture | null {
		return this._prefilteredTexture;
	}

	/**
	 * Binds a user-owned texture for prefiltered reflection output.
	 */
	public bindPrefilteredTexture(texture: Texture | null): this {
		this._prefilteredTexture = texture;
		return this;
	}

	public override clearOutputs(): this {
		super.clearOutputs();
		this._prefilteredTexture = null;
		return this;
	}
}
