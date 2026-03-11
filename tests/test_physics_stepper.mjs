import assert from 'node:assert/strict'
import { DefaultPhysicsSimulator } from '../src/simulation/physics/DefaultPhysicsSimulator.ts'

function createStepResult(worldId) {
	return {
		bodyStates: [],
		events: [],
		activeBodies: 0,
		sleepingBodies: 0,
		ccdBodies: 0,
		worldId,
	}
}

function testFixedStep() {
	const simulator = new DefaultPhysicsSimulator()
	const calls = []
	const context = {
		worlds: [
			{
				worldId: 'w1',
				config: {
					worldId: 'w1',
					mode: 'fixed',
					fixedDeltaMs: 10,
					maxSubsteps: 10,
					maxDeltaMs: 100,
				},
			},
		],
		stepWorld(worldId, deltaSeconds) {
			calls.push({ worldId, deltaSeconds })
			return createStepResult(worldId)
		},
	}

	simulator.beginFrame(context)
	const result = simulator.simulate(context, { deltaTimeMs: 25 })
	simulator.endFrame()

	assert.equal(result.worldResults.length, 1)
	assert.equal(result.worldResults[0].substeps, 2)
	assert.equal(calls.length, 2)
	assert.ok(Math.abs(calls[0].deltaSeconds - 0.01) < 1e-9)
}

function testVariableStep() {
	const simulator = new DefaultPhysicsSimulator()
	const calls = []
	const context = {
		worlds: [
			{
				worldId: 'w2',
				config: {
					worldId: 'w2',
					mode: 'variable',
					maxDeltaMs: 20,
				},
			},
		],
		stepWorld(worldId, deltaSeconds) {
			calls.push({ worldId, deltaSeconds })
			return createStepResult(worldId)
		},
	}

	simulator.beginFrame(context)
	const result = simulator.simulate(context, { deltaTimeMs: 50 })
	simulator.endFrame()

	assert.equal(calls.length, 1)
	assert.ok(Math.abs(calls[0].deltaSeconds - 0.02) < 1e-9)
	assert.equal(result.worldResults[0].consumedDeltaMs, 20)
}

function testMaxSubstepsCap() {
	const simulator = new DefaultPhysicsSimulator()
	const calls = []
	const context = {
		worlds: [
			{
				worldId: 'w3',
				config: {
					worldId: 'w3',
					mode: 'fixed',
					fixedDeltaMs: 5,
					maxSubsteps: 3,
					maxDeltaMs: 100,
				},
			},
		],
		stepWorld(worldId, deltaSeconds) {
			calls.push({ worldId, deltaSeconds })
			return createStepResult(worldId)
		},
	}

	simulator.beginFrame(context)
	const result = simulator.simulate(context, { deltaTimeMs: 100 })
	simulator.endFrame()

	assert.equal(calls.length, 3)
	assert.equal(result.worldResults[0].substeps, 3)
	assert.equal(result.worldResults[0].consumedDeltaMs, 15)
}

function run() {
	testFixedStep()
	testVariableStep()
	testMaxSubstepsCap()
	console.log('Physics stepper tests passed')
}

run()
