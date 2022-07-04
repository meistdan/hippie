/**
 * \file	RadixSortKernels.cu
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	RadixSort kernels source file.
 *			Based on 'High Performance and ScalableRadix Sorting'
 *			by Duane Merrill and Andrew Grimshaw
 */

#include "radix_sort/RadixSortKernels.h"
#include "radix_sort/RadixSortUtil.h"
#include "radix_sort/Reduce.h"
#include "radix_sort/Scan.h"

//---------------------------------------------------------------------------
// UPSWEEP KERNEL
//---------------------------------------------------------------------------

template <int BLOCK_THREADS, int COUNTER_LANES>
DEVICE_INLINE void resetDigitCounters(volatile unsigned int (&packedCounters)[COUNTER_LANES][BLOCK_THREADS]) {
    #pragma unroll
    for (int LANE = 0; LANE < COUNTER_LANES; LANE++) {
        packedCounters[LANE][threadIdx.x] = 0;
    }
}

template <int LANES_PER_WARP, int PACKING_RATIO>
DEVICE_INLINE void resetUnpackedCounters(int (&localCounts)[LANES_PER_WARP][PACKING_RATIO]) {
    #pragma unroll
    for (int LANE = 0; LANE < LANES_PER_WARP; LANE++) {
        #pragma unroll
        for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++) {
            localCounts[LANE][UNPACKED_COUNTER] = 0;
        }
    }
}

template <int BLOCK_THREADS, int COUNTER_LANES, int LANES_PER_WARP, int PACKING_RATIO, int WARPS>
DEVICE_INLINE void unpackDigitCounts(int (&localCounts)[LANES_PER_WARP][PACKING_RATIO], volatile unsigned char (&digitCounters)[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO]) {
    unsigned int warpIndex = threadIdx.x >> LOG_WARP_THREADS;
    unsigned int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);
    #pragma unroll
    for (int LANE = 0; LANE < LANES_PER_WARP; LANE++) {
        const int counterLane = (LANE * WARPS) + warpIndex;
        if (counterLane < COUNTER_LANES) {
            #pragma unroll
            for (int PACKED_COUNTER = 0; PACKED_COUNTER < BLOCK_THREADS; PACKED_COUNTER += WARP_THREADS) {
                #pragma unroll
                for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++) {
                    int counter = digitCounters[counterLane][warpThreadIndex + PACKED_COUNTER][UNPACKED_COUNTER];
                    localCounts[LANE][UNPACKED_COUNTER] += counter;
                }
            }
        }
    }
}

template <int COUNTER_LANES, int LANES_PER_WARP, int PACKING_RATIO, int LOG_PACKING_RATIO, int WARPS, int RADIX_DIGITS>
DEVICE_INLINE void reduceUnpackedCounts(int & binCount, int (&localCounts)[LANES_PER_WARP][PACKING_RATIO], volatile int (&digitPartials)[RADIX_DIGITS][WARP_THREADS + 1]) {
	unsigned int warpIndex = threadIdx.x >> LOG_WARP_THREADS;
    unsigned int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);
    #pragma unroll
    for (int LANE = 0; LANE < LANES_PER_WARP; LANE++) {
        int counterLane = (LANE * WARPS) + warpIndex;
        if (counterLane < COUNTER_LANES) {
            int digitRow = counterLane << LOG_PACKING_RATIO;
            #pragma unroll
            for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++) {
                digitPartials[digitRow + UNPACKED_COUNTER][warpThreadIndex] = localCounts[LANE][UNPACKED_COUNTER];
            }
        }
    }
    __syncthreads();
	if (threadIdx.x < RADIX_DIGITS) {
		binCount = ThreadReduce<volatile int, WARP_THREADS>::reduce((volatile int*)digitPartials[threadIdx.x]);
    }
}

template <int BLOCK_THREADS, int COUNTER_LANES, int LOG_COUNTER_LANES, int PACKING_RATIO, int LOG_PACKING_RATIO>
DEVICE_INLINE void bucket(int currentBit, unsigned long long key, volatile unsigned char (&digitCounters)[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO]) {
    unsigned long long subCounter = bitfieldExtract(key, currentBit, LOG_PACKING_RATIO);
    unsigned long long rowOffset = bitfieldExtract(key, currentBit + LOG_PACKING_RATIO, LOG_COUNTER_LANES);
    digitCounters[rowOffset][threadIdx.x][subCounter]++;
}

template <int COUNT, int MAX>
struct UpsweepIterate {
    
	template <int ITEMS_PER_THREAD, int BLOCK_THREADS, int COUNTER_LANES, int LOG_COUNTER_LANES, int PACKING_RATIO, int LOG_PACKING_RATIO>
    DEVICE_INLINE static void bucketKeys(
		int currentBit, 
		unsigned long long (&keys)[ITEMS_PER_THREAD], 
		volatile unsigned char (&digitCounters)[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO]
	) {
		bucket<BLOCK_THREADS, COUNTER_LANES, LOG_COUNTER_LANES, PACKING_RATIO, LOG_PACKING_RATIO>(currentBit, keys[COUNT], digitCounters);
		UpsweepIterate<COUNT + 1, MAX>::template bucketKeys<ITEMS_PER_THREAD, BLOCK_THREADS, COUNTER_LANES, LOG_COUNTER_LANES, PACKING_RATIO, LOG_PACKING_RATIO>(currentBit, keys, digitCounters);
    }

};

template <int MAX>
struct UpsweepIterate<MAX, MAX> {

    template <int ITEMS_PER_THREAD, int BLOCK_THREADS, int COUNTER_LANES, int LOG_COUNTER_LANES, int PACKING_RATIO, int LOG_PACKING_RATIO>
    DEVICE_INLINE static void bucketKeys(
		int currentBit, 
		unsigned long long (&keys)[ITEMS_PER_THREAD], 
		volatile unsigned char (&digitCounters)[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO]
	) {}

};

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, int COUNTER_LANES, int LOG_COUNTER_LANES, int PACKING_RATIO, int LOG_PACKING_RATIO>
DEVICE_INLINE void processFullTile(int currentBit, int blockOffset, unsigned long long * keysIn, volatile unsigned char (&digitCounters)[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO]) {
	unsigned long long keys[ITEMS_PER_THREAD];
    loadDirectStriped<ITEMS_PER_THREAD, BLOCK_THREADS, unsigned long long>(keysIn + blockOffset, keys);
	UpsweepIterate<0, ITEMS_PER_THREAD>::template bucketKeys<ITEMS_PER_THREAD, BLOCK_THREADS, COUNTER_LANES, LOG_COUNTER_LANES, PACKING_RATIO, LOG_PACKING_RATIO>(currentBit, keys, digitCounters);
}

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, int COUNTER_LANES, int LOG_COUNTER_LANES, int PACKING_RATIO, int LOG_PACKING_RATIO>
DEVICE_INLINE void processPartialTile(int currentBit, int blockOffset, int blockEnd, unsigned long long * keysIn, volatile unsigned char (&digitCounters)[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO]) {
    blockOffset += threadIdx.x;
    while (blockOffset < blockEnd) {
        unsigned long long key = keysIn[blockOffset];
		bucket<BLOCK_THREADS, COUNTER_LANES, LOG_COUNTER_LANES, PACKING_RATIO, LOG_PACKING_RATIO>(currentBit, key, digitCounters);
		blockOffset += BLOCK_THREADS;
	}
}

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, int RADIX_BITS>
DEVICE_INLINE void processRegion(int currentBit, int blockOffset, int blockEnd, unsigned long long * keysIn, int * spine) {

	// Constants.
	const int TILE_ITEMS = ITEMS_PER_THREAD * BLOCK_THREADS;
	const int WARPS = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS;
	const int PACKING_RATIO = sizeof(unsigned int) / sizeof(unsigned char);
    const int LOG_PACKING_RATIO = LOG(PACKING_RATIO);
    const int LOG_COUNTER_LANES = MAX(0, RADIX_BITS - LOG_PACKING_RATIO);
    const int COUNTER_LANES = 1 << LOG_COUNTER_LANES;
	const int LANES_PER_WARP = MAX(1, (COUNTER_LANES + WARPS - 1) / WARPS);
	const int UNROLL_COUNT = MIN(64, 255 / ITEMS_PER_THREAD);
	const int UNROLLED_ELEMENTS = UNROLL_COUNT * TILE_ITEMS;
	const int RADIX_DIGITS = 1 << RADIX_BITS;

	// Shared memory storage.
	__shared__ union {
		volatile unsigned char digitCounters[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO];
        volatile unsigned int packedCounters[COUNTER_LANES][BLOCK_THREADS];
        volatile int digitPartials[RADIX_DIGITS][WARP_THREADS + 1];
	} shared;

	// Local memory storage.
	int localCounts[LANES_PER_WARP][PACKING_RATIO];

	// Reset counters.
	resetDigitCounters<BLOCK_THREADS, COUNTER_LANES>(shared.packedCounters);
	resetUnpackedCounters<LANES_PER_WARP, PACKING_RATIO>(localCounts);

	// Unroll batches of full tiles. 
	while (blockOffset + UNROLLED_ELEMENTS <= blockEnd) {
		
		for (int i = 0; i < UNROLL_COUNT; ++i) {
			processFullTile<ITEMS_PER_THREAD, BLOCK_THREADS, COUNTER_LANES, LOG_COUNTER_LANES, PACKING_RATIO, LOG_PACKING_RATIO>(currentBit, blockOffset, keysIn, shared.digitCounters);
			blockOffset += TILE_ITEMS;
		}

		__syncthreads();

		// Aggregate back into local_count registers to prevent overflow.
		unpackDigitCounts<BLOCK_THREADS, COUNTER_LANES, LANES_PER_WARP, PACKING_RATIO, WARPS>(localCounts, shared.digitCounters);

		__syncthreads();

		// Reset composite counters in lanes.
		resetDigitCounters<BLOCK_THREADS, COUNTER_LANES>(shared.packedCounters);

	}

	// Unroll single full tiles.
    while (blockOffset + TILE_ITEMS <= blockEnd) {
        processFullTile<ITEMS_PER_THREAD, BLOCK_THREADS, COUNTER_LANES, LOG_COUNTER_LANES, PACKING_RATIO, LOG_PACKING_RATIO>(currentBit, blockOffset, keysIn, shared.digitCounters);
        blockOffset += TILE_ITEMS;
    }

	// Process partial tile if necessary.
	processPartialTile<ITEMS_PER_THREAD, BLOCK_THREADS, COUNTER_LANES, LOG_COUNTER_LANES, PACKING_RATIO, LOG_PACKING_RATIO>(currentBit, blockOffset, blockEnd, keysIn, shared.digitCounters);

	__syncthreads();

	// Aggregate back into local_count registers.
    unpackDigitCounts<BLOCK_THREADS, COUNTER_LANES, LANES_PER_WARP, PACKING_RATIO, WARPS>(localCounts, shared.digitCounters);

    __syncthreads();

	// Final raking reduction of counts by bin.
	int binCount;
	reduceUnpackedCounts<COUNTER_LANES, LANES_PER_WARP, PACKING_RATIO, LOG_PACKING_RATIO, WARPS, RADIX_DIGITS>(binCount, localCounts, shared.digitPartials);

	// Write out digit counts (striped).
    if (threadIdx.x < RADIX_DIGITS) spine[gridDim.x * threadIdx.x + blockIdx.x] = binCount;

}

extern "C" GLOBAL void upsweepKernel(int numberOfItems, int currentBit, int bigBlocks, int bigShare, 
	int normalShare, int normalBaseOffset, int totalGrains, unsigned long long * keysIn, int * spine, bool alt) {

	// Compute block offset and block end.
	int blockOffset = numberOfItems;
	int blockEnd = numberOfItems;
	if (blockIdx.x < bigBlocks) {
        blockOffset = (blockIdx.x * bigShare);
		blockEnd = blockOffset + bigShare;
	} else if (blockIdx.x < totalGrains) {
        blockOffset = normalBaseOffset + (blockIdx.x * normalShare);
        blockEnd = MIN(numberOfItems, blockOffset + normalShare);
	}

	// Process region.
	if (alt) processRegion<UPSWEEP_ITEMS_PER_THREAD, UPSWEEP_BLOCK_THREADS, UPSWEEP_RADIX_BITS - 1>(currentBit, blockOffset, blockEnd, keysIn, spine);
	else processRegion<UPSWEEP_ITEMS_PER_THREAD, UPSWEEP_BLOCK_THREADS, UPSWEEP_RADIX_BITS>(currentBit, blockOffset, blockEnd, keysIn, spine);

}

//---------------------------------------------------------------------------
// SCAN KERNEL
//---------------------------------------------------------------------------

template <int COUNT, int MAX>
struct ScanIterate {

    template <int HALF_WARP_THREADS, int SMEM_ELEMENTS>
    DEVICE_INLINE static void warpScan(int & partial, volatile int (&warpSum)[SMEM_ELEMENTS]) {
        const int OFFSET = 1 << COUNT;
        warpSum[HALF_WARP_THREADS + threadIdx.x] = partial;
        partial += warpSum[HALF_WARP_THREADS + threadIdx.x - OFFSET];
        ScanIterate<COUNT + 1, MAX>::template warpScan<HALF_WARP_THREADS>(partial, warpSum);
    }

};

template <int MAX>
struct ScanIterate<MAX, MAX> {

    template <int HALF_WARP_THREADS, int SMEM_ELEMENTS>
    DEVICE_INLINE static void warpScan(int & partial, volatile int (&warpSum)[SMEM_ELEMENTS]) {}

};

extern "C" GLOBAL void scanKernel(int numberOfBins, int4 * spine) {

	// Constants.
	const int BLOCK_THREADS = SCAN_BLOCK_THREADS;
	const int TILE_ITEMS = BLOCK_THREADS * 4;
	const int SEGMENT_LENGTH = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS;
	const int RAKING_THREADS = (BLOCK_THREADS + SEGMENT_LENGTH - 1) / SEGMENT_LENGTH;
	const int GRID_ELEMENTS = RAKING_THREADS * (SEGMENT_LENGTH + 1);
	const int STEPS = LOG(RAKING_THREADS);
	const int SMEM_ELEMENTS = RAKING_THREADS + (1 << (STEPS - 1));
	const int HALF_WARP_THREADS = (1 << (STEPS - 1));

	// Shared memory storage.
	__shared__ volatile int rakingGrid[GRID_ELEMENTS];
	__shared__ volatile int warpSum[SMEM_ELEMENTS];
	
	// Local memory storage.
	int cachedSegment[SEGMENT_LENGTH];

	// Only valid blocks.
	if (blockIdx.x > 0) return;

	// Block offset.
	int blockOffset = 0;

	// Total sum.
	int totalSum = 0;

	// Process full input tiles.
    while ((blockOffset << 2) + TILE_ITEMS <= numberOfBins) {

		// Load items.
		int4 items = spine[threadIdx.x + blockOffset];

		// Place thread partial into shared memory raking grid.
		int threadSum = items.x + items.y + items.z + items.w;
		rakingGrid[threadIdx.x + threadIdx.x / SEGMENT_LENGTH] = threadSum; 

		// Barrier.
		__syncthreads();

		// Raking threads only.
		if (threadIdx.x < RAKING_THREADS) {

			// Raking pointer.
			volatile int * rakingPtr = rakingGrid + threadIdx.x * (SEGMENT_LENGTH + 1);

			// Copy data into registers.
			#pragma unroll
			for (int i = 0; i < SEGMENT_LENGTH; i++) {
				cachedSegment[i] = rakingPtr[i];
			}

			// Upsweep reduction.
			int rakingSum = ThreadReduce<int, SEGMENT_LENGTH>::reduce((int*)cachedSegment);

			// Inclusive sum.
			int inclusiveSum = rakingSum;

			// Init. identity.
			warpSum[threadIdx.x] = 0;

			// Scan steps.
			ScanIterate<0, STEPS>::warpScan<HALF_WARP_THREADS>(inclusiveSum, warpSum);

			// Share partial into buffer.
			warpSum[HALF_WARP_THREADS + threadIdx.x] = inclusiveSum;

			// Compute warp-wide prefix from aggregate, then broadcast to other lanes.
			if (threadIdx.x == 0) warpSum[0] = totalSum;

			// Update.
			totalSum += warpSum[SMEM_ELEMENTS - 1];
			inclusiveSum += warpSum[0];

			// Exclusive sum.
			rakingSum = inclusiveSum - rakingSum;

			// Exclusive raking downsweep scan.
			ThreadScanExclusive<int, SEGMENT_LENGTH>::scan((int*)cachedSegment, rakingSum);

			// Copy data back to smem.
			#pragma unroll
			for (int i = 0; i < SEGMENT_LENGTH; i++) {
				rakingPtr[i] = cachedSegment[i];
			}

		} 

		// Barrier.
		__syncthreads();

		// Grab thread prefix from shared memory.
        threadSum = rakingGrid[threadIdx.x + threadIdx.x / SEGMENT_LENGTH];

		// Exclusive scan in registers with prefix.
		items.w = items.x + items.y + items.z + threadSum;
		items.z = items.w - items.z;
		items.y = items.z - items.y;
		items.x = threadSum;

		// Store items.
		spine[threadIdx.x + blockOffset] = items;

		// Next itearion.
		blockOffset += BLOCK_THREADS;

    }
}

//---------------------------------------------------------------------------
// DOWNSWEEP KERNEL
//---------------------------------------------------------------------------

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
DEVICE_INLINE void copy(T * in, T * out, int blockOffset, int blockEnd) {

	// Full tile.
    while (blockOffset + (ITEMS_PER_THREAD * BLOCK_THREADS) <= blockEnd) {
        
		// Local memory storage.
		T items[ITEMS_PER_THREAD];
		
		// Load data.
		loadDirectStriped<ITEMS_PER_THREAD, BLOCK_THREADS, T>(in + blockOffset, items);

		// Barrier.
		__syncthreads();

		// Save data.
		storeDirectStriped<ITEMS_PER_THREAD, BLOCK_THREADS, T>(out + blockOffset, items);
		
		// Next tile.
		blockOffset += (ITEMS_PER_THREAD * BLOCK_THREADS);

    }

	// Partial tile.
	if (blockOffset < blockEnd) {
        
		// Local memory storage.
        T items[ITEMS_PER_THREAD];

		// Valid items.
		int validItems = blockEnd - blockOffset;

		// Load data.
		loadDirectStriped<ITEMS_PER_THREAD, BLOCK_THREADS, T>(in + blockOffset, items, validItems);

		// Barrier.
        __syncthreads();

		// Save data.
		storeDirectStriped<ITEMS_PER_THREAD, BLOCK_THREADS, T>(out + blockOffset, items, validItems);
        
    }

}

template <int COUNT, int MAX>
struct DownsweepIterate {

	template <int ITEMS_PER_THREAD, int BLOCK_THREADS, int COUNTER_LANES, int LOG_COUNTER_LANES, int PACKING_RATIO, int LOG_PACKING_RATIO>
    DEVICE_INLINE static void decodeKeys(
		int currentBit,
		unsigned long long (&keys)[ITEMS_PER_THREAD], 
		unsigned short (&threadPrefixes)[ITEMS_PER_THREAD], 
        volatile unsigned short * (&digitCountersPtrs)[ITEMS_PER_THREAD], 
		volatile unsigned short (&digitCounters)[COUNTER_LANES + 1][BLOCK_THREADS][PACKING_RATIO]
	) {
        unsigned long long subCounter = bitfieldExtract(keys[COUNT], currentBit + LOG_COUNTER_LANES, LOG_PACKING_RATIO);
        unsigned long long counterLane = bitfieldExtract(keys[COUNT], currentBit, LOG_COUNTER_LANES);
        digitCountersPtrs[COUNT] = &digitCounters[counterLane][threadIdx.x][subCounter];
		threadPrefixes[COUNT] = *digitCountersPtrs[COUNT];
        *digitCountersPtrs[COUNT] = threadPrefixes[COUNT] + 1;
        DownsweepIterate<COUNT + 1, MAX>::template decodeKeys<ITEMS_PER_THREAD, BLOCK_THREADS, COUNTER_LANES, LOG_COUNTER_LANES, PACKING_RATIO, 
			LOG_PACKING_RATIO>(currentBit, keys, threadPrefixes, digitCountersPtrs, digitCounters);
	}

	template <int ITEMS_PER_THREAD>
    DEVICE_INLINE static void updateRanks(
		int (&ranks)[ITEMS_PER_THREAD],
        unsigned short (&threadPrefixes)[ITEMS_PER_THREAD], 
		volatile unsigned short * (&digitCountersPtrs)[ITEMS_PER_THREAD]
	) {
        ranks[COUNT] = threadPrefixes[COUNT] + *digitCountersPtrs[COUNT];
        DownsweepIterate<COUNT + 1, MAX>::updateRanks(ranks, threadPrefixes, digitCountersPtrs);
    }

	template <int HALF_WARP_THREADS, int WARPS, int SMEM_ELEMENTS>
    DEVICE_INLINE static void warpScan(int warpThreadIndex, int warpIndex, PackedCounter & partial, volatile PackedCounter (&warpSum)[WARPS][SMEM_ELEMENTS]) {
        const int OFFSET = 1 << COUNT;
        warpSum[warpIndex][HALF_WARP_THREADS + warpThreadIndex] = partial;
        partial += warpSum[warpIndex][HALF_WARP_THREADS + warpThreadIndex - OFFSET];
        DownsweepIterate<COUNT + 1, MAX>::template warpScan<HALF_WARP_THREADS, WARPS, SMEM_ELEMENTS>(warpThreadIndex, warpIndex, partial, warpSum);
    }

};

template <int MAX>
struct DownsweepIterate<MAX, MAX> {

	template <int ITEMS_PER_THREAD, int BLOCK_THREADS, int COUNTER_LANES, int LOG_COUNTER_LANES, int PACKING_RATIO, int LOG_PACKING_RATIO>
    DEVICE_INLINE static void decodeKeys(
		int currentBit,
		unsigned long long (&keys)[ITEMS_PER_THREAD], 
		unsigned short (&threadPrefixes)[ITEMS_PER_THREAD], 
        volatile unsigned short * (&digitCountersPtrs)[ITEMS_PER_THREAD], 
		volatile unsigned short (&digitCounters)[COUNTER_LANES + 1][BLOCK_THREADS][PACKING_RATIO]) {}

	template <int ITEMS_PER_THREAD>
    DEVICE_INLINE static void updateRanks(
		int (&ranks)[ITEMS_PER_THREAD],
        unsigned short (&threadPrefixes)[ITEMS_PER_THREAD], 
		volatile unsigned short * (&digitCountersPtrs)[ITEMS_PER_THREAD]) {}
	
	template <int HALF_WARP_THREADS, int WARPS, int SMEM_ELEMENTS>
    DEVICE_INLINE static void warpScan(int warpThreadIndex, int warpIndex, PackedCounter & partial, volatile PackedCounter (&warpSum)[WARPS][SMEM_ELEMENTS]) {}

};

template <bool FULL_TILE, int ITEMS_PER_THREAD, int BLOCK_THREADS, int TILE_ITEMS, int RADIX_BITS, int RADIX_DIGITS>
DEVICE_INLINE void scatterKeys(
	int currentBit,
	int validItems,
	int (&ranks)[ITEMS_PER_THREAD],
	unsigned long long * keysOut,
	unsigned long long (&keys)[ITEMS_PER_THREAD],
	volatile unsigned long long (&keysExchange)[TILE_ITEMS],
	int (&relativeBinOffsetsExchange)[ITEMS_PER_THREAD],
	volatile int (&relativeBinOffsets)[RADIX_DIGITS + 1]
) {
		
	// Exchange keys through shared memory.
    scatterToStriped<ITEMS_PER_THREAD, BLOCK_THREADS, unsigned long long>(keys, keysExchange, ranks);

	// Compute striped local ranks.
    int localRanks[ITEMS_PER_THREAD];
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        localRanks[ITEM] = threadIdx.x + (ITEM * BLOCK_THREADS);
    }

	// Compute scatter offsets.
	#pragma unroll
    for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++) {
        unsigned long long digit = bitfieldExtract(keys[KEY], currentBit, RADIX_BITS);
        relativeBinOffsetsExchange[KEY] = relativeBinOffsets[digit];
    }

	// Scatter to global.
	#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if (FULL_TILE || (localRanks[ITEM] < validItems)) {
            keysOut[relativeBinOffsetsExchange[ITEM] + localRanks[ITEM]] = keys[ITEM];
        }
    }

}

template <bool FULL_TILE, int ITEMS_PER_THREAD, int BLOCK_THREADS, int TILE_ITEMS, int RADIX_DIGITS>
DEVICE_INLINE void scatterValues(
	int validItems,
	int (&ranks)[ITEMS_PER_THREAD],
	int * valuesOut,
	int (&values)[ITEMS_PER_THREAD],
	volatile int (&valuesExchange)[TILE_ITEMS],
	int (&relativeBinOffsetsExchange)[ITEMS_PER_THREAD],
	volatile int (&relativeBinOffsets)[RADIX_DIGITS + 1]
) {

	// Barrier.
	__syncthreads();

	// Exchange values through shared memory.
	scatterToStriped<ITEMS_PER_THREAD, BLOCK_THREADS, int>(values, valuesExchange, ranks);

	// Compute striped local ranks.
	int localRanks[ITEMS_PER_THREAD];
	#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		localRanks[ITEM] = threadIdx.x + (ITEM * BLOCK_THREADS);
	}

	// Scatter to global.
	#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if (FULL_TILE || (localRanks[ITEM] < validItems)) {
            valuesOut[relativeBinOffsetsExchange[ITEM] + localRanks[ITEM]] = values[ITEM];
        }
    }

}

template <bool OUTER_SCAN, int ITEMS_PER_THREAD, int BLOCK_THREADS, int COUNTER_LANES, int LOG_COUNTER_LANES, 
	int PACKING_RATIO, int LOG_PACKING_RATIO, int WARPS, int SMEM_ELEMENTS, int HALF_WARP_THREADS, int STEPS, int RADIX_DIGITS>
DEVICE_INLINE void rankKeys(
	int currentBit,
	int (&ranks)[ITEMS_PER_THREAD],
	unsigned long long (&keys)[ITEMS_PER_THREAD], 
	volatile PackedCounter (&warpSum)[WARPS][SMEM_ELEMENTS],
	volatile PackedCounter (&warpAggregates)[WARPS],
	volatile unsigned short (&digitCounters)[COUNTER_LANES + 1][BLOCK_THREADS][PACKING_RATIO], 
	volatile PackedCounter (&rakingGrid)[BLOCK_THREADS][COUNTER_LANES + 1],
	int & inclusiveDigitPrefix
) {

	// Local memory storage.
	unsigned short threadPrefixes[ITEMS_PER_THREAD];
	volatile unsigned short * digitCountersPtrs[ITEMS_PER_THREAD];
	PackedCounter cachedSegment[COUNTER_LANES + 1];

	// Reset shared memory digit counters.
    #pragma unroll
    for (int LANE = 0; LANE < COUNTER_LANES + 1; LANE++) {
		*((PackedCounter*) digitCounters[LANE][threadIdx.x]) = 0;
	}

	// Decode keys and update digit counters.
	DownsweepIterate<0, ITEMS_PER_THREAD>::template decodeKeys<ITEMS_PER_THREAD, BLOCK_THREADS, COUNTER_LANES, LOG_COUNTER_LANES, 
		PACKING_RATIO, LOG_PACKING_RATIO>(currentBit, keys, threadPrefixes, digitCountersPtrs, digitCounters);

	// Barrier.
    __syncthreads();

	// Scan counters.
	PackedCounter * smemRakingPtr = (PackedCounter*)rakingGrid[threadIdx.x];
    PackedCounter * rakingPtr;

	if (OUTER_SCAN) {
		#pragma unroll
        for (int i = 0; i < COUNTER_LANES + 1; i++) {
            cachedSegment[i] = smemRakingPtr[i];
        }
        rakingPtr = cachedSegment;
	} else {
		rakingPtr = smemRakingPtr;
	}

	// Init. identity.
	int warpIndex = threadIdx.x >> LOG_WARP_THREADS;
	int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);
	warpSum[warpIndex][warpThreadIndex] = 0;

	// Raking sum.
	PackedCounter rakingSum = ThreadReduce<PackedCounter, COUNTER_LANES + 1>::reduce(rakingPtr);

	// Compute exclusive sum.
	PackedCounter inclusiveSum = rakingSum;
	DownsweepIterate<0, STEPS>::template warpScan<HALF_WARP_THREADS, WARPS, SMEM_ELEMENTS>(warpThreadIndex, warpIndex, inclusiveSum, warpSum);
	warpSum[warpIndex][HALF_WARP_THREADS + warpThreadIndex] = inclusiveSum;
	PackedCounter exclusiveSum = inclusiveSum - rakingSum;

	// Aggregates.
	PackedCounter warpAggregate = warpSum[warpIndex][SMEM_ELEMENTS - 1];
	warpAggregates[warpIndex] = warpAggregate;

	// Barrier.
    __syncthreads();

	// Block aggregate.
	PackedCounter blockAggregate = warpAggregates[0];

	#pragma unroll
    for (int WARP = 1; WARP < WARPS; WARP++) {
        if (warpIndex == WARP) exclusiveSum += blockAggregate;
        blockAggregate += warpAggregates[WARP];
    }
	
	// Propagate totals in packed fields.
    #pragma unroll
    for (int PACKED = 1; PACKED < PACKING_RATIO; PACKED++) {
        exclusiveSum += blockAggregate << (sizeof(unsigned short) * 8 * PACKED);
    }

	// Exclusive downsweep.
	ThreadScanExclusive<PackedCounter, COUNTER_LANES + 1>::scan(rakingPtr, exclusiveSum);

	// Copy data back to smem.
	if (OUTER_SCAN) {
        #pragma unroll
        for (int i = 0; i < COUNTER_LANES + 1; i++) {
            smemRakingPtr[i] = cachedSegment[i];
        }
    }

	// Barrier.
    __syncthreads();

	// Extract the local ranks of each key.
    DownsweepIterate<0, ITEMS_PER_THREAD>::updateRanks(ranks, threadPrefixes, digitCountersPtrs);

	// Get the inclusive and exclusive digit totals corresponding to the calling thread.
    if (threadIdx.x < RADIX_DIGITS) {
        int counterLane = (threadIdx.x & (COUNTER_LANES - 1));
        int subCounter = threadIdx.x >> (LOG_COUNTER_LANES);
        inclusiveDigitPrefix = digitCounters[counterLane + 1][0][subCounter];
    }

}

template <bool FULL_TILE, bool OUTER_SCAN, int ITEMS_PER_THREAD, int BLOCK_THREADS, int TILE_ITEMS, int COUNTER_LANES, int LOG_COUNTER_LANES, int PACKING_RATIO, 
	int LOG_PACKING_RATIO, int WARPS, int SMEM_ELEMENTS, int HALF_WARP_THREADS, int STEPS, int RADIX_BITS, int RADIX_DIGITS, LoadAlgorithm ALGORITHM>
DEVICE_INLINE void processTile(
	int currentBit, 
	int blockOffset, 
	int & binOffset,
	unsigned long long * keysIn, 
	unsigned long long * keysOut,
	int * valuesIn,
	int * valuesOut,
	volatile unsigned long long (&keysExchange)[TILE_ITEMS],
	volatile int (&valuesExchange)[TILE_ITEMS],
	volatile int (&relativeBinOffsets)[RADIX_DIGITS + 1],
	volatile PackedCounter (&warpSum)[WARPS][SMEM_ELEMENTS],
	volatile PackedCounter (&warpAggregates)[WARPS],
	volatile unsigned short (&digitCounters)[COUNTER_LANES + 1][BLOCK_THREADS][PACKING_RATIO], 
	volatile PackedCounter (&rakingGrid)[BLOCK_THREADS][COUNTER_LANES + 1],
	int validItems = TILE_ITEMS
) {

	// Local memory storage.
    unsigned long long keys[ITEMS_PER_THREAD];
	int values[ITEMS_PER_THREAD];
    int ranks[ITEMS_PER_THREAD];
    int relativeBinOffsetsExchange[ITEMS_PER_THREAD];

	// Assign max-key to all keys.
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        keys[i] = -1ull;
    }

	// Load tile of keys.
	BlockLoad<ITEMS_PER_THREAD, BLOCK_THREADS, unsigned long long, FULL_TILE, ALGORITHM>::load(keysIn + blockOffset, keys, keysExchange, validItems);

	// Barrier.
    __syncthreads();

	// Rank the keys.
	int inclusiveDigitPrefix;
	rankKeys<OUTER_SCAN, ITEMS_PER_THREAD, BLOCK_THREADS, COUNTER_LANES, LOG_COUNTER_LANES, PACKING_RATIO, LOG_PACKING_RATIO, WARPS, SMEM_ELEMENTS, 
		HALF_WARP_THREADS, STEPS, RADIX_DIGITS>(currentBit, ranks, keys, warpSum, warpAggregates, digitCounters, rakingGrid, inclusiveDigitPrefix);

	// Update global scatter base offsets for each digit.
	int exclusiveDigitPrefix = 0;
    if (threadIdx.x < RADIX_DIGITS) {
        relativeBinOffsets[threadIdx.x] = 0;
        relativeBinOffsets[threadIdx.x + 1] = inclusiveDigitPrefix;
        exclusiveDigitPrefix = relativeBinOffsets[threadIdx.x];
		binOffset -= exclusiveDigitPrefix;
        relativeBinOffsets[threadIdx.x] = binOffset;
        binOffset += inclusiveDigitPrefix;
	}

	// Barrier.
	__syncthreads();
	
	// Scatter keys.
	scatterKeys<FULL_TILE, ITEMS_PER_THREAD, BLOCK_THREADS, TILE_ITEMS, RADIX_BITS, RADIX_DIGITS>(currentBit, validItems, ranks, keysOut, keys, keysExchange, relativeBinOffsetsExchange, relativeBinOffsets);

	// Barrier.
    __syncthreads();

	// Load values.
	BlockLoad<ITEMS_PER_THREAD, BLOCK_THREADS, int, FULL_TILE, ALGORITHM>::load(valuesIn + blockOffset, values, valuesExchange, validItems);

	// Barrier.
    __syncthreads();

	// Gather/scatter values.
	scatterValues<FULL_TILE, ITEMS_PER_THREAD, BLOCK_THREADS, TILE_ITEMS, RADIX_DIGITS>(validItems, ranks, valuesOut, values, valuesExchange, relativeBinOffsetsExchange, relativeBinOffsets);

}

template <bool OUTER_SCAN, int ITEMS_PER_THREAD, int BLOCK_THREADS, int RADIX_BITS, LoadAlgorithm ALGORITHM>
DEVICE_INLINE void processRegion(int blockOffset, int blockEnd, int currentBit, int numberOfItems, 
	int * spine, unsigned long long * keysIn, unsigned long long * keysOut, int * valuesIn, int * valuesOut) {

	// Constants.
	const int RADIX_DIGITS = 1 << RADIX_BITS;
	const int TILE_ITEMS = ITEMS_PER_THREAD * BLOCK_THREADS;
	const int PACKING_RATIO = sizeof(PackedCounter) / sizeof(unsigned short);
	const int LOG_PACKING_RATIO = LOG(PACKING_RATIO);
	const int LOG_COUNTER_LANES = MAX((RADIX_BITS - LOG_PACKING_RATIO), 0);
    const int COUNTER_LANES = 1 << LOG_COUNTER_LANES;
	const int WARPS = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS;
	const int STEPS = LOG(WARP_THREADS);
	const int HALF_WARP_THREADS = 1 << (STEPS - 1);
	const int SMEM_ELEMENTS = WARP_THREADS + HALF_WARP_THREADS;

	// Shared memory storage.
	__shared__ volatile bool earlyExit;
	__shared__ volatile int relativeBinOffsets[RADIX_DIGITS + 1];
	__shared__ volatile PackedCounter warpSum[WARPS][SMEM_ELEMENTS];
	__shared__ volatile PackedCounter warpAggregates[WARPS];
	__shared__ union {
		volatile int values[TILE_ITEMS];
		volatile unsigned long long keys[TILE_ITEMS];
	} exchangeShared;
	__shared__ union {
		volatile unsigned short digitCounters[COUNTER_LANES + 1][BLOCK_THREADS][PACKING_RATIO];
		volatile PackedCounter rakingGrid[BLOCK_THREADS][COUNTER_LANES + 1];
	} rankShared;

	// Bin offset.
	int binOffset = 0;

	// Whether early exit.
	if (threadIdx.x < RADIX_DIGITS) {

		// Predicate.
        int firstBlockBinOffset = spine[gridDim.x * threadIdx.x];
        int predicate = ((firstBlockBinOffset == 0) || (firstBlockBinOffset == numberOfItems));
		earlyExit = __all(predicate);

		// Set bin offset.
		binOffset = spine[(gridDim.x * threadIdx.x) + blockIdx.x];

	}

	// Barrier.
	__syncthreads();

	// Early exit.
	if (earlyExit) {

		// Copy keys.
		copy<ITEMS_PER_THREAD, BLOCK_THREADS, unsigned long long>(keysIn, keysOut, blockOffset, blockEnd);

        // Copy values.
		copy<ITEMS_PER_THREAD, BLOCK_THREADS, int>(valuesIn, valuesOut, blockOffset, blockEnd);

	}

	// Standard procedure.
	else {

		// Full tile.
        while (blockOffset + TILE_ITEMS <= blockEnd) {
            
			processTile<true, OUTER_SCAN, ITEMS_PER_THREAD, BLOCK_THREADS, TILE_ITEMS, COUNTER_LANES, LOG_COUNTER_LANES, PACKING_RATIO, LOG_PACKING_RATIO, WARPS, SMEM_ELEMENTS, 
				HALF_WARP_THREADS, STEPS, RADIX_BITS, RADIX_DIGITS, ALGORITHM>(currentBit, blockOffset, binOffset, keysIn, keysOut, valuesIn, valuesOut, exchangeShared.keys, exchangeShared.values, 
				relativeBinOffsets, warpSum, warpAggregates, rankShared.digitCounters, rankShared.rakingGrid);

			blockOffset += TILE_ITEMS;

			// Barrier.
            __syncthreads();

        }

        // Partial tile.
        if (blockOffset < blockEnd) {
			processTile<false, OUTER_SCAN, ITEMS_PER_THREAD, BLOCK_THREADS, TILE_ITEMS, COUNTER_LANES, LOG_COUNTER_LANES, PACKING_RATIO, LOG_PACKING_RATIO, WARPS, SMEM_ELEMENTS, 
				HALF_WARP_THREADS, STEPS, RADIX_BITS, RADIX_DIGITS, ALGORITHM>(currentBit, blockOffset, binOffset, keysIn, keysOut, valuesIn, valuesOut, exchangeShared.keys, exchangeShared.values, 
				relativeBinOffsets, warpSum, warpAggregates, rankShared.digitCounters, rankShared.rakingGrid, blockEnd - blockOffset);
        }

	}

}

extern "C" GLOBAL void downsweepKernel(int currentBit, int numberOfItems, int bigBlocks, int bigShare, int normalShare, int normalBaseOffset, 
	int totalGrains, int * spine, unsigned long long * keysIn, unsigned long long * keysOut, int * valuesIn, int * valuesOut, bool alt) {

	// Compute block offset and block end.
	int blockOffset = numberOfItems;
	int blockEnd = numberOfItems;
	if (blockIdx.x < bigBlocks) {
        blockOffset = (blockIdx.x * bigShare);
		blockEnd = blockOffset + bigShare;
	} else if (blockIdx.x < totalGrains) {
        blockOffset = normalBaseOffset + (blockIdx.x * normalShare);
        blockEnd = MIN(numberOfItems, blockOffset + normalShare);
	}

	// Process region.
	if (alt) processRegion<DOWNSWEEP_OUTER_SCAN, DOWNSWEEP_ITEMS_PER_THREAD, DOWNSWEEP_BLOCK_THREADS, DOWNSWEEP_RADIX_BITS - 1, 
		DOWNSWEEP_LOAD_ALGORITHM>(blockOffset, blockEnd, currentBit, numberOfItems, spine, keysIn, keysOut, valuesIn, valuesOut);
	else processRegion<DOWNSWEEP_OUTER_SCAN, DOWNSWEEP_ITEMS_PER_THREAD, DOWNSWEEP_BLOCK_THREADS, DOWNSWEEP_RADIX_BITS, 
		DOWNSWEEP_LOAD_ALGORITHM>(blockOffset, blockEnd, currentBit, numberOfItems, spine, keysIn, keysOut, valuesIn, valuesOut);

}
