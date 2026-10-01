import type {
	BrowserTestCleanup,
	BrowserTestSession,
	DeferBrowserTestCleanup,
} from "./types.ts";

/**
 * @internal Browser fixture setup with reverse-order, idempotent cleanup.
 * Setup failures release acquired resources before reaching Playwright.
 */
export async function createBrowserTestSession<TResult>(
	setup: (defer: DeferBrowserTestCleanup) => Promise<TResult>,
): Promise<BrowserTestSession<TResult>> {
	const cleanups: BrowserTestCleanup[] = [];
	let disposal: Promise<void> | undefined;
	const dispose = (): Promise<void> => {
		disposal ??= (async () => {
			const errors: unknown[] = [];
			while (cleanups.length > 0) {
				try {
					await cleanups.pop()!();
				} catch (error) {
					errors.push(error);
				}
			}
			if (errors.length === 1) throw errors[0];
			if (errors.length > 1) {
				throw new AggregateError(errors, "Browser fixture cleanup failed.");
			}
		})();
		return disposal;
	};
	try {
		const result = await setup((cleanup) => { cleanups.push(cleanup); });
		return { result, dispose };
	} catch (error) {
		try {
			await dispose();
		} catch (cleanupError) {
			throw new AggregateError(
				[error, cleanupError], "Browser fixture setup and cleanup failed.",
			);
		}
		throw error;
	}
}

/**
 * @internal Creates an attached browser test canvas with stable CSS dimensions.
 * Renderer initialization measures the DOM rectangle before resizing its buffer.
 */
export function createTestCanvas(
	defer: DeferBrowserTestCleanup,
	width: number,
	height: number,
): HTMLCanvasElement {
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	canvas.style.width = `${width}px`;
	canvas.style.height = `${height}px`;
	canvas.style.display = "block";
	defer(() => { canvas.remove(); });
	document.body.append(canvas);
	return canvas;
}
