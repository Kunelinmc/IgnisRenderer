import type { ProjectedFace, ProjectedVertex } from "../../core/types";
import type { IVector3, IVector4 } from "../../maths/types";
import type { FragmentInput } from "../../shaders";
import { CoreConstants } from "./constants";

/**
 * @internal Software rasterizer cache for perspective-correct vertex attributes.
 */
export interface SoftwareInterpolatedVertex {
	x: number;
	y: number;
	z: number;
	iz: number;
	worldO: IVector3;
	previousWorldO: IVector3;
	normalO: IVector3;
	tangentO: IVector4;
	uO: number;
	vO: number;
	u2O: number;
	v2O: number;
	u3O: number;
	v3O: number;
	u4O: number;
	v4O: number;
	zCamO: number;
}

/**
 * @internal Software rasterizer edge sample for scanline span setup.
 */
export interface SoftwareEdgeSample {
	x: number;
	iz: number;
	worldO: IVector3;
	previousWorldO: IVector3;
	normalO: IVector3;
	tangentO: IVector4;
	uO: number;
	vO: number;
	u2O: number;
	v2O: number;
	u3O: number;
	v3O: number;
	u4O: number;
	v4O: number;
	zCamO: number;
}

/**
 * @internal Mutable depth-only scanline span used by the software rasterizer.
 */
export class SoftwareDepthSpan {
	public iz = 0;
	public zCamO = 0;
	public zCam = 0;
	public zCamValue = 0;

	private _diz = 0;
	private _dzCamO = 0;

	public setup(
		left: SoftwareEdgeSample,
		right: SoftwareEdgeSample,
		startX: number
	): void {
		const spanWidth = right.x - left.x;
		const spanInv = 1.0 / (spanWidth || CoreConstants.EPSILON);
		this._diz = (right.iz - left.iz) * spanInv;
		this._dzCamO = (right.zCamO - left.zCamO) * spanInv;

		const dx = startX + 0.5 - left.x;
		this.iz = left.iz + dx * this._diz;
		this.zCamO = left.zCamO + dx * this._dzCamO;
	}

	public computeDepth(): boolean {
		const safeIz =
			Math.abs(this.iz) > CoreConstants.EPSILON ? this.iz
			: this.iz >= 0 ? CoreConstants.EPSILON
			: -CoreConstants.EPSILON;
		this.zCam = 1 / safeIz;
		this.zCamValue = this.zCamO * this.zCam;
		return this.zCam > 0;
	}

	public advance(): void {
		this.iz += this._diz;
		this.zCamO += this._dzCamO;
	}
}

/**
 * @internal Mutable fragment scanline span used by the software rasterizer.
 */
export class SoftwareFragmentSpan {
	public iz = 0;
	public worldOx = 0;
	public worldOy = 0;
	public worldOz = 0;
	public previousWorldOx = 0;
	public previousWorldOy = 0;
	public previousWorldOz = 0;
	public normalOx = 0;
	public normalOy = 0;
	public normalOz = 0;
	public tangentOx = 0;
	public tangentOy = 0;
	public tangentOz = 0;
	public tangentOw = 0;
	public uO = 0;
	public vO = 0;
	public u2O = 0;
	public v2O = 0;
	public u3O = 0;
	public v3O = 0;
	public u4O = 0;
	public v4O = 0;
	public zCamO = 0;
	public zCam = 0;
	public zCamValue = 0;

	private _diz = 0;
	private _dWorldOx = 0;
	private _dWorldOy = 0;
	private _dWorldOz = 0;
	private _dPreviousWorldOx = 0;
	private _dPreviousWorldOy = 0;
	private _dPreviousWorldOz = 0;
	private _dNormalOx = 0;
	private _dNormalOy = 0;
	private _dNormalOz = 0;
	private _dTangentOx = 0;
	private _dTangentOy = 0;
	private _dTangentOz = 0;
	private _dTangentOw = 0;
	private _duO = 0;
	private _dvO = 0;
	private _du2O = 0;
	private _dv2O = 0;
	private _du3O = 0;
	private _dv3O = 0;
	private _du4O = 0;
	private _dv4O = 0;
	private _dzCamO = 0;

	public setup(
		left: SoftwareEdgeSample,
		right: SoftwareEdgeSample,
		startX: number
	): void {
		const spanWidth = right.x - left.x;
		const spanInv = 1.0 / (spanWidth || CoreConstants.EPSILON);

		this._diz = (right.iz - left.iz) * spanInv;
		this._dWorldOx = (right.worldO.x - left.worldO.x) * spanInv;
		this._dWorldOy = (right.worldO.y - left.worldO.y) * spanInv;
		this._dWorldOz = (right.worldO.z - left.worldO.z) * spanInv;
		this._dPreviousWorldOx =
			(right.previousWorldO.x - left.previousWorldO.x) * spanInv;
		this._dPreviousWorldOy =
			(right.previousWorldO.y - left.previousWorldO.y) * spanInv;
		this._dPreviousWorldOz =
			(right.previousWorldO.z - left.previousWorldO.z) * spanInv;
		this._dNormalOx = (right.normalO.x - left.normalO.x) * spanInv;
		this._dNormalOy = (right.normalO.y - left.normalO.y) * spanInv;
		this._dNormalOz = (right.normalO.z - left.normalO.z) * spanInv;
		this._dTangentOx = (right.tangentO.x - left.tangentO.x) * spanInv;
		this._dTangentOy = (right.tangentO.y - left.tangentO.y) * spanInv;
		this._dTangentOz = (right.tangentO.z - left.tangentO.z) * spanInv;
		this._dTangentOw = (right.tangentO.w - left.tangentO.w) * spanInv;
		this._duO = (right.uO - left.uO) * spanInv;
		this._dvO = (right.vO - left.vO) * spanInv;
		this._du2O = (right.u2O - left.u2O) * spanInv;
		this._dv2O = (right.v2O - left.v2O) * spanInv;
		this._du3O = (right.u3O - left.u3O) * spanInv;
		this._dv3O = (right.v3O - left.v3O) * spanInv;
		this._du4O = (right.u4O - left.u4O) * spanInv;
		this._dv4O = (right.v4O - left.v4O) * spanInv;
		this._dzCamO = (right.zCamO - left.zCamO) * spanInv;

		const dx = startX + 0.5 - left.x;
		this.iz = left.iz + dx * this._diz;
		this.worldOx = left.worldO.x + dx * this._dWorldOx;
		this.worldOy = left.worldO.y + dx * this._dWorldOy;
		this.worldOz = left.worldO.z + dx * this._dWorldOz;
		this.previousWorldOx =
			left.previousWorldO.x + dx * this._dPreviousWorldOx;
		this.previousWorldOy =
			left.previousWorldO.y + dx * this._dPreviousWorldOy;
		this.previousWorldOz =
			left.previousWorldO.z + dx * this._dPreviousWorldOz;
		this.normalOx = left.normalO.x + dx * this._dNormalOx;
		this.normalOy = left.normalO.y + dx * this._dNormalOy;
		this.normalOz = left.normalO.z + dx * this._dNormalOz;
		this.tangentOx = left.tangentO.x + dx * this._dTangentOx;
		this.tangentOy = left.tangentO.y + dx * this._dTangentOy;
		this.tangentOz = left.tangentO.z + dx * this._dTangentOz;
		this.tangentOw = left.tangentO.w + dx * this._dTangentOw;
		this.uO = left.uO + dx * this._duO;
		this.vO = left.vO + dx * this._dvO;
		this.u2O = left.u2O + dx * this._du2O;
		this.v2O = left.v2O + dx * this._dv2O;
		this.u3O = left.u3O + dx * this._du3O;
		this.v3O = left.v3O + dx * this._dv3O;
		this.u4O = left.u4O + dx * this._du4O;
		this.v4O = left.v4O + dx * this._dv4O;
		this.zCamO = left.zCamO + dx * this._dzCamO;
	}

	public computeDepth(): boolean {
		const safeIz =
			Math.abs(this.iz) > CoreConstants.EPSILON ? this.iz
			: this.iz >= 0 ? CoreConstants.EPSILON
			: -CoreConstants.EPSILON;
		this.zCam = 1 / safeIz;
		this.zCamValue = this.zCamO * this.zCam;
		return this.zCam > 0;
	}

	public writeFragmentInput(input: FragmentInput): void {
		input.zCam = this.zCam;
		input.world.x = this.worldOx * this.zCam;
		input.world.y = this.worldOy * this.zCam;
		input.world.z = this.worldOz * this.zCam;
		input.normal.x = this.normalOx * this.zCam;
		input.normal.y = this.normalOy * this.zCam;
		input.normal.z = this.normalOz * this.zCam;
		input.tangent.x = this.tangentOx * this.zCam;
		input.tangent.y = this.tangentOy * this.zCam;
		input.tangent.z = this.tangentOz * this.zCam;
		input.tangent.w = this.tangentOw * this.zCam;
		input.u = this.uO * this.zCam;
		input.v = this.vO * this.zCam;
		input.u2 = this.u2O * this.zCam;
		input.v2 = this.v2O * this.zCam;
		input.u3 = this.u3O * this.zCam;
		input.v3 = this.v3O * this.zCam;
		input.u4 = this.u4O * this.zCam;
		input.v4 = this.v4O * this.zCam;
		input.zCam = this.zCamValue;
	}

	public advance(): void {
		this.iz += this._diz;
		this.worldOx += this._dWorldOx;
		this.worldOy += this._dWorldOy;
		this.worldOz += this._dWorldOz;
		this.previousWorldOx += this._dPreviousWorldOx;
		this.previousWorldOy += this._dPreviousWorldOy;
		this.previousWorldOz += this._dPreviousWorldOz;
		this.normalOx += this._dNormalOx;
		this.normalOy += this._dNormalOy;
		this.normalOz += this._dNormalOz;
		this.tangentOx += this._dTangentOx;
		this.tangentOy += this._dTangentOy;
		this.tangentOz += this._dTangentOz;
		this.tangentOw += this._dTangentOw;
		this.uO += this._duO;
		this.vO += this._dvO;
		this.u2O += this._du2O;
		this.v2O += this._dv2O;
		this.u3O += this._du3O;
		this.v3O += this._dv3O;
		this.u4O += this._du4O;
		this.v4O += this._dv4O;
		this.zCamO += this._dzCamO;
	}
}

/**
 * @internal Shared interpolation helper for software rasterizer hot paths.
 */
export class SoftwareTriangleInterpolator {
	public readonly depthSpan = new SoftwareDepthSpan();
	public readonly fragmentSpan = new SoftwareFragmentSpan();

	private readonly _vertices: SoftwareInterpolatedVertex[];
	private readonly _edgeA: SoftwareEdgeSample = createEdgeSample();
	private readonly _edgeB: SoftwareEdgeSample = createEdgeSample();
	private _left: SoftwareEdgeSample = this._edgeA;
	private _right: SoftwareEdgeSample = this._edgeB;

	public constructor() {
		this._vertices = Array.from({ length: 3 }, createInterpolatedVertex);
	}

	public get left(): SoftwareEdgeSample {
		return this._left;
	}

	public get right(): SoftwareEdgeSample {
		return this._right;
	}

	public prepareCameraDepth(pts: ProjectedVertex[]): SoftwareInterpolatedVertex[] {
		const verts = this._vertices;
		for (let i = 0; i < 3; i++) {
			const p = pts[i];
			const iz = p.w;
			const linearDepth =
				p.zView !== undefined ? -p.zView
				: p.world.z !== undefined ? -p.world.z
				: 0;
			const v = verts[i];
			v.x = p.x;
			v.y = p.y;
			v.iz = iz;
			v.zCamO = linearDepth * iz;
		}
		return verts;
	}

	public prepareFragment(
		pts: ProjectedVertex[],
		face: ProjectedFace
	): SoftwareInterpolatedVertex[] {
		const verts = this._vertices;
		for (let i = 0; i < 3; i++) {
			const p = pts[i];
			const world = p.world ?? { x: 0, y: 0, z: 0 };
			const previousWorld = p.previousWorld ?? world;
			const normal = p.normal ?? face.normal ?? { x: 0, y: 0, z: 1 };
			const tangent = p.tangent ?? { x: 0, y: 0, z: 0, w: 0 };
			const iz = p.w;

			const v = verts[i];
			v.x = p.x;
			v.y = p.y;
			v.z = p.z;
			v.iz = iz;
			v.worldO.x = world.x * iz;
			v.worldO.y = world.y * iz;
			v.worldO.z = world.z * iz;
			v.previousWorldO.x = previousWorld.x * iz;
			v.previousWorldO.y = previousWorld.y * iz;
			v.previousWorldO.z = previousWorld.z * iz;
			v.normalO.x = normal.x * iz;
			v.normalO.y = normal.y * iz;
			v.normalO.z = normal.z * iz;
			v.tangentO.x = tangent.x * iz;
			v.tangentO.y = tangent.y * iz;
			v.tangentO.z = tangent.z * iz;
			v.tangentO.w = tangent.w * iz;
			v.uO = (p.u ?? 0) * iz;
			v.vO = (p.v ?? 0) * iz;
			v.u2O = (p.u2 ?? 0) * iz;
			v.v2O = (p.v2 ?? 0) * iz;
			v.u3O = (p.u3 ?? 0) * iz;
			v.v3O = (p.v3 ?? 0) * iz;
			v.u4O = (p.u4 ?? 0) * iz;
			v.v4O = (p.v4 ?? 0) * iz;

			const linearDepth =
				p.zView !== undefined ? -p.zView
				: p.world.z !== undefined ? -p.world.z
				: 0;
			v.zCamO = linearDepth * iz;
		}
		return verts;
	}

	public sampleScanlineEdges(
		vTop: SoftwareInterpolatedVertex,
		vMid: SoftwareInterpolatedVertex,
		vBot: SoftwareInterpolatedVertex,
		y: number
	): void {
		if (y < vMid.y) {
			fillEdgeSample(this._edgeA, vTop, vMid, y);
			fillEdgeSample(this._edgeB, vTop, vBot, y);
		} else {
			fillEdgeSample(this._edgeA, vMid, vBot, y);
			fillEdgeSample(this._edgeB, vTop, vBot, y);
		}

		if (this._edgeA.x > this._edgeB.x) {
			this._left = this._edgeB;
			this._right = this._edgeA;
			return;
		}

		this._left = this._edgeA;
		this._right = this._edgeB;
	}
}

function createInterpolatedVertex(): SoftwareInterpolatedVertex {
	return {
		x: 0,
		y: 0,
		z: 0,
		iz: 0,
		worldO: { x: 0, y: 0, z: 0 },
		previousWorldO: { x: 0, y: 0, z: 0 },
		normalO: { x: 0, y: 0, z: 0 },
		tangentO: { x: 0, y: 0, z: 0, w: 0 },
		uO: 0,
		vO: 0,
		u2O: 0,
		v2O: 0,
		u3O: 0,
		v3O: 0,
		u4O: 0,
		v4O: 0,
		zCamO: 0,
	};
}

function createEdgeSample(): SoftwareEdgeSample {
	return {
		x: 0,
		iz: 0,
		worldO: { x: 0, y: 0, z: 0 },
		previousWorldO: { x: 0, y: 0, z: 0 },
		normalO: { x: 0, y: 0, z: 0 },
		tangentO: { x: 0, y: 0, z: 0, w: 0 },
		uO: 0,
		vO: 0,
		u2O: 0,
		v2O: 0,
		u3O: 0,
		v3O: 0,
		u4O: 0,
		v4O: 0,
		zCamO: 0,
	};
}

function fillEdgeSample(
	res: SoftwareEdgeSample,
	vA: SoftwareInterpolatedVertex,
	vB: SoftwareInterpolatedVertex,
	y: number
): void {
	const dy = vB.y - vA.y;
	const t = dy === 0 ? 0 : (y - vA.y) / dy;
	res.x = vA.x + (vB.x - vA.x) * t;
	res.iz = vA.iz + (vB.iz - vA.iz) * t;
	res.worldO.x = vA.worldO.x + (vB.worldO.x - vA.worldO.x) * t;
	res.worldO.y = vA.worldO.y + (vB.worldO.y - vA.worldO.y) * t;
	res.worldO.z = vA.worldO.z + (vB.worldO.z - vA.worldO.z) * t;
	res.previousWorldO.x =
		vA.previousWorldO.x + (vB.previousWorldO.x - vA.previousWorldO.x) * t;
	res.previousWorldO.y =
		vA.previousWorldO.y + (vB.previousWorldO.y - vA.previousWorldO.y) * t;
	res.previousWorldO.z =
		vA.previousWorldO.z + (vB.previousWorldO.z - vA.previousWorldO.z) * t;
	res.normalO.x = vA.normalO.x + (vB.normalO.x - vA.normalO.x) * t;
	res.normalO.y = vA.normalO.y + (vB.normalO.y - vA.normalO.y) * t;
	res.normalO.z = vA.normalO.z + (vB.normalO.z - vA.normalO.z) * t;
	res.tangentO.x = vA.tangentO.x + (vB.tangentO.x - vA.tangentO.x) * t;
	res.tangentO.y = vA.tangentO.y + (vB.tangentO.y - vA.tangentO.y) * t;
	res.tangentO.z = vA.tangentO.z + (vB.tangentO.z - vA.tangentO.z) * t;
	res.tangentO.w = vA.tangentO.w + (vB.tangentO.w - vA.tangentO.w) * t;
	res.uO = vA.uO + (vB.uO - vA.uO) * t;
	res.vO = vA.vO + (vB.vO - vA.vO) * t;
	res.u2O = vA.u2O + (vB.u2O - vA.u2O) * t;
	res.v2O = vA.v2O + (vB.v2O - vA.v2O) * t;
	res.u3O = vA.u3O + (vB.u3O - vA.u3O) * t;
	res.v3O = vA.v3O + (vB.v3O - vA.v3O) * t;
	res.u4O = vA.u4O + (vB.u4O - vA.u4O) * t;
	res.v4O = vA.v4O + (vB.v4O - vA.v4O) * t;
	res.zCamO = vA.zCamO + (vB.zCamO - vA.zCamO) * t;
}
