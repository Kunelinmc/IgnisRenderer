import assert from "node:assert/strict";
import { Logger } from "../src/foundation/Logger.ts";

function createSinkRecorder() {
	const records = {
		debug: [],
		info: [],
		warn: [],
		error: [],
	};
	return {
		records,
		sink: {
			debug(...args) {
				records.debug.push(args);
			},
			info(...args) {
				records.info.push(args);
			},
			warn(...args) {
				records.warn.push(args);
			},
			error(...args) {
				records.error.push(args);
			},
		},
	};
}

function testErrorOnceLogsOnlyOncePerKey() {
	const { records, sink } = createSinkRecorder();
	const logger = new Logger({ name: "TestLogger", level: "debug" }, sink);

	assert.equal(logger.errorOnce("error-key", "first"), true);
	assert.equal(logger.errorOnce("error-key", "second"), false);

	assert.equal(records.error.length, 1);
	assert.deepEqual(records.error[0], ["[TestLogger]", "first"]);
}

function testWarnOnceUsesIndependentKeys() {
	const { records, sink } = createSinkRecorder();
	const logger = new Logger({ level: "warn" }, sink);

	assert.equal(logger.warnOnce("key-a", "A"), true);
	assert.equal(logger.warnOnce("key-b", "B"), true);
	assert.equal(logger.warnOnce("key-a", "A2"), false);

	assert.equal(records.warn.length, 2);
	assert.deepEqual(records.warn[0], ["A"]);
	assert.deepEqual(records.warn[1], ["B"]);
}

function testOnceKeyNotConsumedWhenLevelBlocked() {
	const { records, sink } = createSinkRecorder();
	const logger = new Logger({ level: "error" }, sink);

	assert.equal(logger.warnOnce("blocked-key", "blocked"), false);
	assert.equal(logger.hasOnceKey("blocked-key"), false);
	assert.equal(records.warn.length, 0);

	logger.setLevel("warn");
	assert.equal(logger.warnOnce("blocked-key", "allowed"), true);
	assert.equal(records.warn.length, 1);
	assert.deepEqual(records.warn[0], ["allowed"]);
}

function testClearOnceKeyAndClearOnceKeys() {
	const { records, sink } = createSinkRecorder();
	const logger = new Logger({ level: "error" }, sink);

	assert.equal(logger.errorOnce("k1", "one"), true);
	assert.equal(logger.errorOnce("k2", "two"), true);
	assert.equal(records.error.length, 2);

	logger.clearOnceKey("k1");
	assert.equal(logger.errorOnce("k1", "one-again"), true);
	assert.equal(records.error.length, 3);

	logger.clearOnceKeys();
	assert.equal(logger.errorOnce("k2", "two-again"), true);
	assert.equal(records.error.length, 4);
}

function testGenericLogOnceWorks() {
	const { records, sink } = createSinkRecorder();
	const logger = new Logger({ level: "debug" }, sink);

	assert.equal(logger.logOnce("info", "info-key", "info-message"), true);
	assert.equal(logger.logOnce("info", "info-key", "ignored"), false);
	assert.equal(records.info.length, 1);
	assert.deepEqual(records.info[0], ["info-message"]);
}

function run() {
	testErrorOnceLogsOnlyOncePerKey();
	testWarnOnceUsesIndependentKeys();
	testOnceKeyNotConsumedWhenLevelBlocked();
	testClearOnceKeyAndClearOnceKeys();
	testGenericLogOnceWorks();
	console.log("Logger tests passed");
}

run();
