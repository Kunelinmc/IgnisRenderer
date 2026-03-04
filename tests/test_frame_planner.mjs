import assert from 'node:assert/strict'
import { FramePlanner } from '../src/core/pipeline/FramePlanner.ts'

function createFrame(overrides = {}) {
	return {
		sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 1 },
		lights: [],
		camera: null,
		shadowMaps: new Map(),
		opaquePackets: [{}],
		transparentPackets: [],
		shadowCasterPackets: [],
		shadowTransmitterPackets: [],
		reflectivePackets: [],
		...overrides,
	}
}

function run() {
	const baseResolved = {
		enableLighting: true,
		enableGamma: true,
		enableSH: false,
		enableShadows: true,
		enableReflection: false,
		enableSkybox: false,
		enableSSAO: false,
		enableVolumetric: false,
		enableFXAA: true,
		warnings: [],
	}

	const frame = createFrame({
		shadowCasterPackets: [{}],
		transparentPackets: [{}],
		reflectivePackets: [{}],
	})
	const plan = FramePlanner.build(
		frame,
		{
			...baseResolved,
			enableReflection: true,
			enableSSAO: true,
			enableVolumetric: true,
		}
	)

	assert.deepEqual(
		plan.map((pass) => pass.stage),
		[
			'shadow',
			'reflection',
			'main-opaque',
			'main-transparent',
			'ssao',
			'volumetric',
			'fxaa',
			'gamma',
		]
	)
	assert.equal(plan.find((pass) => pass.stage === 'shadow')?.enabled, true)
	assert.equal(plan.find((pass) => pass.stage === 'reflection')?.enabled, true)
	assert.equal(plan.find((pass) => pass.stage === 'main-opaque')?.enabled, true)
	assert.equal(
		plan.find((pass) => pass.stage === 'main-transparent')?.enabled,
		true
	)
	assert.equal(plan.find((pass) => pass.stage === 'ssao')?.enabled, true)
	assert.equal(plan.find((pass) => pass.stage === 'volumetric')?.enabled, true)
	assert.equal(plan.find((pass) => pass.stage === 'fxaa')?.enabled, true)
	assert.equal(plan.find((pass) => pass.stage === 'gamma')?.enabled, true)

	const disabledPlan = FramePlanner.build(createFrame(), baseResolved)
	assert.equal(
		disabledPlan.find((pass) => pass.stage === 'shadow')?.enabled,
		false
	)
	assert.equal(
		disabledPlan.find((pass) => pass.stage === 'main-transparent')?.enabled,
		false
	)
	assert.equal(
		disabledPlan.find((pass) => pass.stage === 'reflection')?.enabled,
		false
	)

	console.log('Frame planner tests passed')
}

run()
