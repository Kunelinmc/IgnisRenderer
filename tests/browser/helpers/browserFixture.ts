import type { Page } from "@playwright/test";

import type { BrowserTestSession } from "./types.ts";

/**
 * @internal Runs browser fixture setup in an isolated shared test page.
 * `setup` executes in the browser and must not capture Playwright variables.
 * `use` runs in Playwright while browser resources remain alive. Both session
 * cleanup and handle release are attempted after result transfer or use fails.
 */
export async function withBrowserFixture<TResult, TOutput>(
	page: Page,
	setup: () => Promise<BrowserTestSession<TResult>>,
	use: (result: TResult) => TOutput | Promise<TOutput>,
): Promise<TOutput> {
	await page.goto("/tests/browser/fixtures/browser.html");
	const session = await page.evaluateHandle(setup);
	const errors: unknown[] = [];
	let output!: TOutput;
	try {
		const result = await session.evaluate((value) => value.result);
		output = await use(result);
	} catch (error) {
		errors.push(error);
	} finally {
		try {
			await session.evaluate((value) => value.dispose());
		} catch (error) {
			errors.push(error);
		} finally {
			try {
				await session.dispose();
			} catch (error) {
				errors.push(error);
			}
		}
	}
	if (errors.length === 1) throw errors[0];
	if (errors.length > 1) {
		throw new AggregateError(errors, "Browser fixture execution and cleanup failed.");
	}
	return output;
}
