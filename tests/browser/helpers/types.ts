/** @internal Browser test cleanup registered by fixture setup. */
export type BrowserTestCleanup = () => void | Promise<void>;

/** @internal Browser fixtures register cleanup as soon as resources are acquired. */
export type DeferBrowserTestCleanup = (cleanup: BrowserTestCleanup) => void;

/**
 * @internal Browser test resources retained through assertions and screenshots.
 * Playwright specs consume sessions through `withBrowserFixture()`.
 */
export interface BrowserTestSession<TResult> {
	readonly result: TResult;
	dispose(): Promise<void>;
}
