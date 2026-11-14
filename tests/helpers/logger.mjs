import { Logger } from "../../src/foundation/Logger.ts";

/** Records keyed engine warnings. The caller must reset Logger in its cleanup. */
export function recordLoggerWarnings(warnings = []) {
	Logger.configure({
		level: "warn",
		resetOnceKeys: true,
		sink: {
			warn: (_scope, value) => {
				const match = String(value).match(/^\[([^\]]+)\] (.*)$/s);
				warnings.push({ key: match?.[1], message: match?.[2] ?? String(value) });
			},
		},
	});
	return warnings;
}
