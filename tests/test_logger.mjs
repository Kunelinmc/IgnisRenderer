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

	assert.equal(logger.error("first", { onceKey: "error-key" }), true);
	assert.equal(logger.error("second", { onceKey: "error-key" }), false);

	assert.equal(records.error.length, 1);
	assert.deepEqual(records.error[0], ["[TestLogger]", "first"]);
}

function testWarnOnceUsesIndependentKeys() {
	const { records, sink } = createSinkRecorder();
	const logger = new Logger({ level: "warn" }, sink);

	assert.equal(logger.warn("A", { onceKey: "key-a" }), true);
	assert.equal(logger.warn("B", { onceKey: "key-b" }), true);
	assert.equal(logger.warn("A2", { onceKey: "key-a" }), false);

	assert.equal(records.warn.length, 2);
	assert.deepEqual(records.warn[0], ["A"]);
	assert.deepEqual(records.warn[1], ["B"]);
}

function testOnceKeyNotConsumedWhenLevelBlocked() {
	const { records, sink } = createSinkRecorder();
	const logger = new Logger({ level: "error" }, sink);

	assert.equal(logger.warn("blocked", { onceKey: "blocked-key" }), false);
	assert.equal(records.warn.length, 0);

	logger.setLevel("warn");
	assert.equal(logger.warn("allowed", { onceKey: "blocked-key" }), true);
	assert.equal(records.warn.length, 1);
	assert.deepEqual(records.warn[0], ["allowed"]);
}

function testFirstArgumentCanBeArray() {
	const { records, sink } = createSinkRecorder();
	const logger = new Logger({ level: "debug" }, sink);

	assert.equal(
		logger.info(["info-message", { code: 42 }], { onceKey: "info-key" }),
		true
	);
	assert.equal(
		logger.info(["ignored", { code: 43 }], { onceKey: "info-key" }),
		false
	);
	assert.equal(records.info.length, 1);
	assert.deepEqual(records.info[0], ["info-message", { code: 42 }]);
}

function run() {
	testErrorOnceLogsOnlyOncePerKey();
	testWarnOnceUsesIndependentKeys();
	testOnceKeyNotConsumedWhenLevelBlocked();
	testFirstArgumentCanBeArray();
	console.log("Logger tests passed");
}

run();
