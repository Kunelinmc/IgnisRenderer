import type { WorkerScheduler } from "../../workers/WorkerScheduler";

export type SoftwareRasterMode = "scanline" | "tile";

export interface SoftwareTileOptions {
	tileSize?: number;
	workerCount?: number;
	scheduler?: WorkerScheduler;
	poolId?: string;
	defaultTimeoutMs?: number;
}

export interface SoftwareBackendOptions {
	rasterMode?: SoftwareRasterMode;
	tile?: SoftwareTileOptions;
}

export const DEFAULT_SOFTWARE_RASTER_MODE: SoftwareRasterMode = "scanline";
export const DEFAULT_SOFTWARE_TILE_SIZE = 32;
