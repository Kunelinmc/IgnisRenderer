		let phongAmbient = model.phongAmbientShininess.rgb;
		let phongSpecular = model.phongSpecularShading.rgb;
		let shininess = max(model.phongAmbientShininess.a, 0.0);

		var ambientBase = frame.ambientColor.rgb;
		if (useSHAmbient()) {
			ambientBase =
				sampleDiffuseProbeIrradiance(input.worldPosition, normal) / 255.0;
		}
		var ambient = ambientBase * phongAmbient;
		var direct = vec3<f32>(0.0);

		let directionalCount = u32(frame.lightCounts.x + 0.5);
		for (var i: u32 = 0u; i < directionalCount; i = i + 1u) {
			let lightDirection = safeNormalize(
				frame.directionalLights[i].direction.xyz,
				vec3<f32>(0.0, 1.0, 0.0)
			);
			let radiance = frame.directionalLights[i].color.xyz;
			let nDotL = max(dot(normal, lightDirection), 0.0);
			if (nDotL <= 0.0) {
				continue;
			}

			let shadow = sampleDirectionalShadowVisibility(
				i,
				input.worldPosition,
				shadowNormal,
				lightDirection,
				linearDepth
			);
			let halfVector = safeNormalize(viewDir + lightDirection, viewDir);
			let specFactor = select(0.0, pow(max(dot(normal, halfVector), 0.0), shininess), nDotL > 0.0);
			direct += radiance * shadow * nDotL * baseColor;
			direct += radiance * shadow * specFactor * phongSpecular;
		}

		if (isClusteredLightingEnabled()) {
			let clusterHeader = getClusterHeaderForFragment(
				input.position.xy,
				linearDepth
			);
			let clusterEntryCount = getClusterEntryCount(clusterHeader);
			let clusterLightCount = activeClusteredLightCount();
			for (var entryIndex: u32 = 0u; entryIndex < clusterEntryCount; entryIndex = entryIndex + 1u) {
				let packedRef = clusterIndices.indices[clusterHeader.offset + entryIndex];
				let clusterRef = decodeClusteredLightRef(packedRef);
				if (clusterRef.lightIndex >= clusterLightCount) {
					continue;
				}
				if (clusterRef.lightType == CLUSTER_LIGHT_TYPE_AREA) {
					let areaRecord = clusteredRecordToAreaLight(clusterRef.lightIndex);
					for (
						var sampleIndex: u32 = 0u;
						sampleIndex < AREA_LIGHT_SAMPLE_COUNT;
						sampleIndex = sampleIndex + 1u
					) {
						let areaLight = evaluateAreaLight(
							areaRecord,
							input.worldPosition,
							sampleIndex
						);
						if (!areaLight.valid) {
							continue;
						}

						let lightDirection = areaLight.direction;
						let nDotL = max(dot(normal, lightDirection), 0.0);
						if (nDotL <= 0.0) {
							continue;
						}

						let halfVector = safeNormalize(viewDir + lightDirection, viewDir);
						let specFactor = select(
							0.0,
							pow(max(dot(normal, halfVector), 0.0), shininess),
							nDotL > 0.0
						);
						direct += areaLight.radiance * nDotL * baseColor;
						direct += areaLight.radiance * specFactor * phongSpecular;
					}
					continue;
				}
				let positionRange = clusterPositionRanges.values[clusterRef.lightIndex];
				let colorInner = clusterColorInners.values[clusterRef.lightIndex];
				let toLight = positionRange.xyz - input.worldPosition;
				let distanceSq = dot(toLight, toLight);
				let distanceValue = sqrt(max(distanceSq, EPSILON));
				let lightRange = positionRange.w;
				if (distanceValue > lightRange) {
					continue;
				}

				let lightDirection = toLight / distanceValue;
				var attenuation = pointAttenuation(distanceSq, lightRange);
				var shadow = vec3<f32>(1.0);

				// Spot light cone + shadow
				if (clusterRef.lightType == CLUSTER_LIGHT_TYPE_SPOT) {
					let directionOuter =
						clusterDirectionOuters.values[clusterRef.lightIndex];
					let lightToPoint = -lightDirection;
					let coneDirection = safeNormalize(
						directionOuter.xyz,
						vec3<f32>(0.0, -1.0, 0.0)
					);
					let coneAttenuation = spotAttenuation(
						dot(lightToPoint, coneDirection),
						directionOuter.w,
						colorInner.w
					);
					if (coneAttenuation <= 0.0) {
						continue;
					}
					attenuation *= coneAttenuation;
					let shadowIndex =
						clusterMetadata.values[clusterRef.lightIndex].shadowIndex;
					if (clusterRef.shadowed && shadowIndex < 8u) {
						shadow = sampleSpotShadowVisibility(
							shadowIndex,
							input.worldPosition,
							shadowNormal,
							lightDirection
						);
					}
				} else if (clusterRef.lightType != CLUSTER_LIGHT_TYPE_POINT) {
					continue;
				}

				let radiance = colorInner.xyz * attenuation;
				let nDotL = max(dot(normal, lightDirection), 0.0);
				if (nDotL <= 0.0) {
					continue;
				}

				let halfVector = safeNormalize(viewDir + lightDirection, viewDir);
				let specFactor = select(0.0, pow(max(dot(normal, halfVector), 0.0), shininess), nDotL > 0.0);
				direct += radiance * shadow * nDotL * baseColor;
				direct += radiance * shadow * specFactor * phongSpecular;
			}
		} else {
			let pointCount = u32(frame.lightCounts.y + 0.5);
			for (var i: u32 = 0u; i < pointCount; i = i + 1u) {
				let toLight = frame.pointLights[i].positionRange.xyz - input.worldPosition;
				let distanceSq = dot(toLight, toLight);
				let distanceValue = sqrt(max(distanceSq, EPSILON));
				let lightRange = frame.pointLights[i].positionRange.w;
				if (distanceValue > lightRange) {
					continue;
				}

				let lightDirection = toLight / distanceValue;
				let attenuation = pointAttenuation(distanceSq, lightRange);
				let radiance = frame.pointLights[i].color.xyz * attenuation;
				let nDotL = max(dot(normal, lightDirection), 0.0);
				if (nDotL <= 0.0) {
					continue;
				}

				let halfVector = safeNormalize(viewDir + lightDirection, viewDir);
				let specFactor = select(0.0, pow(max(dot(normal, halfVector), 0.0), shininess), nDotL > 0.0);
				direct += radiance * nDotL * baseColor;
				direct += radiance * specFactor * phongSpecular;
			}

			let spotCount = u32(frame.lightCounts.z + 0.5);
			for (var i: u32 = 0u; i < spotCount; i = i + 1u) {
				let toLight = frame.spotLights[i].positionRange.xyz - input.worldPosition;
				let distanceSq = dot(toLight, toLight);
				let distanceValue = sqrt(max(distanceSq, EPSILON));
				let lightRange = frame.spotLights[i].positionRange.w;
				if (distanceValue > lightRange) {
					continue;
				}

				let lightDirection = toLight / distanceValue;
				let lightToPoint = -lightDirection;
				let coneDirection = safeNormalize(
					frame.spotLights[i].directionOuter.xyz,
					vec3<f32>(0.0, -1.0, 0.0)
				);
				let coneAttenuation = spotAttenuation(
					dot(lightToPoint, coneDirection),
					frame.spotLights[i].directionOuter.w,
					frame.spotLights[i].colorInner.w
				);
				if (coneAttenuation <= 0.0) {
					continue;
				}

				let attenuation = pointAttenuation(distanceSq, lightRange) * coneAttenuation;
				let radiance = frame.spotLights[i].colorInner.xyz * attenuation;
				let nDotL = max(dot(normal, lightDirection), 0.0);
				if (nDotL <= 0.0) {
					continue;
				}

				let shadow = sampleSpotShadowVisibility(
					i,
					input.worldPosition,
					shadowNormal,
					lightDirection
				);
				let halfVector = safeNormalize(viewDir + lightDirection, viewDir);
				let specFactor = select(0.0, pow(max(dot(normal, halfVector), 0.0), shininess), nDotL > 0.0);
				direct += radiance * shadow * nDotL * baseColor;
				direct += radiance * shadow * specFactor * phongSpecular;
			}
		}

		if (!isClusteredLightingEnabled()) {
			let areaCount = areaLightCount();
			for (var i: u32 = 0u; i < areaCount; i = i + 1u) {
				for (
					var sampleIndex: u32 = 0u;
					sampleIndex < AREA_LIGHT_SAMPLE_COUNT;
					sampleIndex = sampleIndex + 1u
				) {
					let areaLight = evaluateAreaLight(
						frame.areaLights[i],
						input.worldPosition,
						sampleIndex
					);
					if (!areaLight.valid) {
						continue;
					}

					let lightDirection = areaLight.direction;
					let nDotL = max(dot(normal, lightDirection), 0.0);
					if (nDotL <= 0.0) {
						continue;
					}

					let halfVector = safeNormalize(viewDir + lightDirection, viewDir);
					let specFactor = select(
						0.0,
						pow(max(dot(normal, halfVector), 0.0), shininess),
						nDotL > 0.0
					);
					direct += areaLight.radiance * nDotL * baseColor;
					direct += areaLight.radiance * specFactor * phongSpecular;
				}
			}
		}

		let finalLinear = ambient + direct + emissive;
		return buildSceneOutput(
			finalLinear,
			alpha,
			baseColor,
			normal,
			1.0,
			0.0,
			emissive,
			1.0,
			motion,
			linearDepth
		);
	}
