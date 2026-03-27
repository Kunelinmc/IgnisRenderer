import assert from "node:assert/strict";
import {
	CommonMaterialLibraryPlugin,
} from "../src/addons/CommonMaterialLibraryPlugin.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";

function testBuiltInPresetCoverage() {
	const plugin = new CommonMaterialLibraryPlugin();
	const presetIds = plugin.listPresetIds();

	assert.ok(
		presetIds.length >= 12,
		`Expected many built-in presets, got ${presetIds.length}`
	);
	assert.ok(presetIds.includes("water"));
	assert.ok(presetIds.includes("brushed-steel"));
	assert.ok(presetIds.includes("matte-plastic"));
	assert.ok(presetIds.includes("simulated-normal-map"));
}

function testWaterPresetMaterialCreation() {
	const plugin = new CommonMaterialLibraryPlugin();
	const material = plugin.createWaterMaterial();

	assert.ok(material instanceof PBRMaterial);
	assert.equal(material.type, "PBR");
	assert.ok(material.transmissionFactor >= 0.9);
	assert.ok(material.clearcoat >= 0.9);
	assert.ok(material.roughness <= 0.2);
}

function testSimulatedNormalMapPreset() {
	const plugin = new CommonMaterialLibraryPlugin({ normalMapSize: 32 });
	const materialA = plugin.createSimulatedNormalMapMaterial();
	const materialB = plugin.createSimulatedNormalMapMaterial();

	assert.ok(materialA.normalMap);
	assert.ok(materialB.normalMap);
	assert.notEqual(
		materialA.normalMap,
		materialB.normalMap,
		"Expected procedural normal map texture to be generated per instance"
	);
	assert.equal(materialA.normalMap.colorSpace, "Linear");
	assert.equal(materialA.normalMap.width, 32);
	assert.equal(materialA.normalMap.height, 32);
	assert.ok(materialA.normalScale > 1.0);

	const textureData = materialA.normalMap.data;
	assert.ok(textureData instanceof Uint8ClampedArray);
	let hasDetail = false;
	for (let i = 0; i < textureData.length; i += 4) {
		if (
			textureData[i] !== 128 ||
			textureData[i + 1] !== 128 ||
			textureData[i + 2] !== 255
		) {
			hasDetail = true;
			break;
		}
	}
	assert.equal(
		hasDetail,
		true,
		"Expected procedural normal map to contain non-flat details"
	);
}

function testOverrideDoesNotMutatePresetDefaults() {
	const plugin = new CommonMaterialLibraryPlugin();
	const overridden = plugin.createMaterial("water", {
		name: "Custom Water",
		roughness: 0.43,
		metalness: 0.12,
	});
	const defaultWater = plugin.createWaterMaterial();

	assert.equal(overridden.name, "Custom Water");
	assert.equal(overridden.roughness, 0.43);
	assert.equal(overridden.metalness, 0.12);
	assert.notEqual(defaultWater.roughness, overridden.roughness);
	assert.notEqual(defaultWater.metalness, overridden.metalness);
}

function testCustomPresetRegistration() {
	const plugin = new CommonMaterialLibraryPlugin();
	plugin.registerPreset({
		id: "fabric-satin",
		name: "Fabric Satin",
		description: "Soft satin-like cloth material.",
		category: "surface",
		tags: ["fabric", "cloth", "satin"],
		factory: () => ({
			albedo: { r: 180, g: 132, b: 192 },
			metalness: 0.0,
			roughness: 0.76,
			sheenColorFactor: { r: 196, g: 170, b: 202 },
			sheenRoughnessFactor: 0.22,
		}),
	});

	assert.equal(plugin.hasPreset("fabric-satin"), true);

	const material = plugin.createMaterial("fabric-satin");
	assert.ok(material instanceof PBRMaterial);
	assert.equal(material.name, "Fabric Satin");
	assert.ok(material.roughness > 0.7);
	assert.ok(material.sheenRoughnessFactor > 0);

	assert.equal(plugin.unregisterPreset("fabric-satin"), true);
	assert.equal(plugin.hasPreset("fabric-satin"), false);
	assert.equal(
		plugin.unregisterPreset("water"),
		false,
		"Built-in presets should not be removable"
	);
}

function testRejectsMalformedPresetDefinitions() {
	const plugin = new CommonMaterialLibraryPlugin();

	assert.throws(
		() => plugin.registerPreset(undefined),
		/definition must be an object/
	);
	assert.throws(
		() => plugin.registerPreset([]),
		/definition must be an object/
	);
	assert.throws(
		() =>
			plugin.registerPreset({
				id: "missing-factory",
				name: "Missing Factory",
				description: "Invalid preset without factory function.",
				category: "surface",
				tags: ["invalid"],
			}),
		/missing a factory/
	);
}

function run() {
	try {
		testBuiltInPresetCoverage();
		testWaterPresetMaterialCreation();
		testSimulatedNormalMapPreset();
		testOverrideDoesNotMutatePresetDefaults();
		testCustomPresetRegistration();
		testRejectsMalformedPresetDefinitions();
		console.log("Common material library plugin tests passed");
	} catch (error) {
		console.error("Common material library plugin tests failed");
		console.error(error);
		process.exit(1);
	}
}

run();
