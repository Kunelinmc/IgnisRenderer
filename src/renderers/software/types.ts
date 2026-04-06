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
