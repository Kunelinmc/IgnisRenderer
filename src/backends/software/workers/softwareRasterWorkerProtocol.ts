export interface SoftwareRasterTileBounds {
	minTileX: number;
	minTileY: number;
	maxTileX: number;
	maxTileY: number;
}

export interface SoftwareRasterTileBinEntry {
	tileIndex: number;
	triangleIndices: number[];
}

export interface SoftwareRasterBinTaskPayload {
	type: "bin-main-pass";
	width: number;
	height: number;
	tileSize: number;
	triangleBounds: SoftwareRasterTileBounds[];
	startIndex: number;
	endIndex: number;
}

export interface SoftwareRasterBinTaskResult {
	type: "bin-main-pass";
	bins: SoftwareRasterTileBinEntry[];
}

export type SoftwareRasterWorkerTaskPayload = SoftwareRasterBinTaskPayload;
export type SoftwareRasterWorkerTaskResult = SoftwareRasterBinTaskResult;
