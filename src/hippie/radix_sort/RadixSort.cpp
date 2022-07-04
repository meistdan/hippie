/**
 * \file	RadixSort.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	RadixSort class source file.
 *			Based on 'High Performance and ScalableRadix Sorting'
 *			by Duane Merrill and Andrew Grimshaw
 */

#include "gpu/HipCompiler.h"
#include "RadixSortPolicy.h"
#include "RadixSortKernels.h"
#include "RadixSort.h"
#include "util/Logger.h"

void RadixSort::init() {
	upsweepConfig.blockThreads = UPSWEEP_BLOCK_THREADS;
	scanConfig.tileSize = SCAN_BLOCK_THREADS * 4;
	scanConfig.blockThreads = SCAN_BLOCK_THREADS;
	downsweepConfig.tileSize = DOWNSWEEP_BLOCK_THREADS * DOWNSWEEP_ITEMS_PER_THREAD;
	downsweepConfig.blockThreads = DOWNSWEEP_BLOCK_THREADS;
	downsweepConfig.radixBits = DOWNSWEEP_RADIX_BITS;
	downsweepConfig.altRadixBits = DOWNSWEEP_RADIX_BITS - 1;
	downsweepConfig.smemConfig = DOWNSWEEP_SMEM_CONFIG;
	downsweepConfig.maxGridSize = getMaxGridSize(false);
	downsweepConfig.altMaxGridSize = getMaxGridSize(true);
}

int RadixSort::getMaxGridSize(bool alt) {

	int arch = HipModule::getComputeCapability() * 10;
	int warpThreads = 1 << LOG_WARP_THREADS;
	int maxSmBlocks = MAX_SM_BLOCKS(arch);
	int maxSmWarps = MAX_SM_THREADS(arch) / warpThreads;
	int regsByBlock = REGS_BY_BLOCK(arch);
	int maxSmRegisters = MAX_SM_REGISTERS(arch);
	int warpAllocUnit = WARP_ALLOC_UNIT(arch);
	int smemAllocUnit = SMEM_ALLOC_UNIT(arch);
	int regAllocUnit = REG_ALLOC_UNIT(arch);
	int smemBytes = SMEM_BYTES(arch);
	int subscriptionFactor = SUBSCRIPTION_FACTOR(arch);
	int smCounts = HipModule::getSMCount();

	// Register and shared memory usage.
	HipModule * module = compiler.compile();
	HipKernel downsweepKernel = module->getKernel("downsweepKernel");
	int numRegs = downsweepKernel.getNumRegs();
	int numSmem = downsweepKernel.getNumSmem();

	// Number of warps per threadblock.
    int blockWarps = (downsweepConfig.blockThreads +  warpThreads - 1) / warpThreads;

    // Max warp occupancy.
    int maxWarpOccupancy = maxSmWarps / blockWarps;

    // Maximum register occupancy.
    int maxRegOccupancy;
    if (regsByBlock) {
        // Allocates registers by threadblock.
        int blockRegs = roundUpNearest(numRegs * warpThreads * blockWarps, regAllocUnit);
        maxRegOccupancy = maxSmRegisters / blockRegs;
    } else {
        // Allocates registers by warp.
        int smSides = warpAllocUnit;
        int smRegistersPerSide = maxSmRegisters / smSides;
        int regsPerWarp = roundUpNearest(numRegs * warpThreads, regAllocUnit);
        int warpsPerSide = smRegistersPerSide / regsPerWarp;
        int warps = warpsPerSide * smSides;
        maxRegOccupancy = warps / blockWarps;
    }

    // Shared memory per threadblock.
	int blockAllocatedSmem = roundUpNearest(numSmem, smemAllocUnit);

    // Max shared memory occupancy.
	int maxSmemOccupancy = blockAllocatedSmem > 0 ? smemBytes / blockAllocatedSmem : maxSmBlocks;

    // Max occupancy.
    int maxSmOccupancy = qMin(qMin(maxSmBlocks, maxWarpOccupancy), qMin(maxSmemOccupancy, maxRegOccupancy));

	return maxSmOccupancy * smCounts * subscriptionFactor;

}

float RadixSort::dispatch(
	Buffer & keys1,
	Buffer & keys2,
	Buffer & values1,
	Buffer & values2,
	Buffer & spine,
	bool & swapBuffers,
	int begin, 
	int end, 
	int spineSize, 
	int beginBit, 
	int endBit,
	bool alt
) {

	// Hip module.
	HipModule * module = compiler.compile();

	// Elapsed time.
	float time = 0.0f;

	// Offsets.
	int keyOffset = begin * sizeof(unsigned long long);
	int valueOffset = begin * sizeof(int);
	int numberOfItems = end - begin;

	// Even share.
	int totalGrains = (numberOfItems + downsweepConfig.tileSize - 1) / downsweepConfig.tileSize;
	int gridSize = MIN(totalGrains, alt ? downsweepConfig.altMaxGridSize : downsweepConfig.maxGridSize);
	int grainsPerBlock = totalGrains / gridSize;
	int bigBlocks = totalGrains - (grainsPerBlock * gridSize);
	int normalShare = grainsPerBlock * downsweepConfig.tileSize;
	int normalBaseOffset = bigBlocks * downsweepConfig.tileSize;
	int bigShare = normalShare + downsweepConfig.tileSize;

	// Iterate over digit places.
	int currentBit = beginBit;
	while (currentBit < endBit) {

		// Upsweep kernel.
		HipKernel upsweepKernel = module->getKernel("upsweepKernel");
		upsweepKernel.setParams(numberOfItems, currentBit, bigBlocks, bigShare, normalShare, normalBaseOffset, totalGrains, 
			swapBuffers ? keys2.getMutableHipPtr(keyOffset) : keys1.getMutableHipPtr(keyOffset), spine.getMutableHipPtr(), alt);
		time += upsweepKernel.launchTimed(gridSize * upsweepConfig.blockThreads, Vec2i(upsweepConfig.blockThreads, 1));
		
		// Scan kernel.
		HipKernel scanKernel = module->getKernel("scanKernel");
		scanKernel.setParams(spineSize, spine.getMutableHipPtr());
		time += scanKernel.launchTimed(scanConfig.blockThreads, Vec2i(scanConfig.blockThreads, 1));

		// Downsweep kernel.
		HipKernel downsweepKernel = module->getKernel("downsweepKernel");
		if (!swapBuffers) {
			downsweepKernel.setParams(currentBit, numberOfItems, bigBlocks, bigShare, normalShare, normalBaseOffset, totalGrains, spine.getHipPtr(), 
				keys1.getHipPtr(keyOffset), keys2.getMutableHipPtr(keyOffset), values1.getHipPtr(valueOffset), values2.getMutableHipPtr(valueOffset), alt);
		} else {
			downsweepKernel.setParams(currentBit, numberOfItems, bigBlocks, bigShare, normalShare, normalBaseOffset, totalGrains, spine.getHipPtr(), 
				keys2.getHipPtr(keyOffset), keys1.getMutableHipPtr(keyOffset), values2.getHipPtr(valueOffset), values1.getMutableHipPtr(valueOffset), alt);
		}
		time += downsweepKernel.launchTimed(gridSize * downsweepConfig.blockThreads, Vec2i(downsweepConfig.blockThreads, 1));

		// Swap buffers.
		swapBuffers = !swapBuffers;

		// Update current bit position
		currentBit += alt ? downsweepConfig.altRadixBits : downsweepConfig.radixBits;

	}
	
	return time;

}

RadixSort::RadixSort() {
    compiler.setSourceFile("../src/hippie/radix_sort/RadixSortKernels.cu");
	compiler.compile();
	init();
}

RadixSort::~RadixSort() {
}

bool RadixSort::getSwap(int beginBit, int endBit) {
	int numPasses = 0;
	int numBits = endBit - beginBit;
    int remainingBits = numBits % downsweepConfig.radixBits;
	if (remainingBits != 0) {
		int maxAltPasses = downsweepConfig.radixBits - remainingBits;
		int altEndBit = qMin(endBit, beginBit + (maxAltPasses * downsweepConfig.altRadixBits));
		numPasses += altEndBit / downsweepConfig.altRadixBits;
		beginBit = altEndBit;
	}
	numPasses += (endBit - beginBit) / downsweepConfig.radixBits;
	return (numPasses & 1);
}

float RadixSort::sort(
	Buffer & keys1,
	Buffer & keys2,
	Buffer & values1,
	Buffer & values2,
	Buffer & spine,
	bool & swapBuffers,
	int begin,
	int end,
	int beginBit,
	int endBit
) {

	// Get spine sizes (conservative).
	int spineSize = (downsweepConfig.maxGridSize * (1 << downsweepConfig.radixBits)) + scanConfig.tileSize;
    int altSpineSize = (downsweepConfig.altMaxGridSize * (1 << downsweepConfig.altRadixBits)) + scanConfig.tileSize;
	int spineAlocSpineSize = qMax(spineSize, altSpineSize);

	const int ALIGN_BYTES = 256;
    const int ALIGN_MASK = ~(ALIGN_BYTES - 1);

	// Allocate temporaries.
	size_t spineAlocSize = sizeof(int) * spineAlocSpineSize;
	size_t spineAlocBytes = (spineAlocSize + ALIGN_BYTES - 1) & ALIGN_MASK;

	// Resize aux buffer.
	if (size_t(spine.getSize()) < spineAlocBytes) spine.resizeDiscard(spineAlocBytes);

	// Run radix sorting passes.
	int numBits = endBit - beginBit;
    int remainingBits = numBits % downsweepConfig.radixBits;

	// Init. swap flag.
	swapBuffers = false;

	// Elapsed time.
	float time = 0.0f;

	if (remainingBits != 0) {

		// Run passes of alternate configuration.
        int maxAltPasses = downsweepConfig.radixBits - remainingBits;
		int altEndBit = qMin(endBit, beginBit + (maxAltPasses * downsweepConfig.altRadixBits));

		// Dispatch.
		time += dispatch(keys1, keys2, values1, values2, spine, swapBuffers, begin, end, altSpineSize, beginBit, altEndBit, true);

		// Sort remaining bits.
		beginBit = altEndBit;

	}

	// Dispatch.
	time += dispatch(keys1, keys2, values1, values2, spine, swapBuffers, begin, end, spineSize, beginBit, endBit, false);

	return time;

}

float RadixSort::sort(
	Buffer & keys1,
	Buffer & keys2,
	Buffer & values1,
	Buffer & values2,
	Buffer & spine,
	bool & swapBuffers,
	int numberOfItems,
	int beginBit,
	int endBit
) {
	return sort(keys1, keys2, values1, values2, spine, swapBuffers, 0, numberOfItems, beginBit, endBit);
}
