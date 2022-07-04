/**
 * \file	RadixSortUtil.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	A header file containing device function used in radix sort kernels.
 */

#ifndef _RADIX_SORT_UTIL_H_
#define _RADIX_SORT_UTIL_H_

#include "Globals.h"

// Load algorithms.
enum LoadAlgorithm {
	LOAD_DIRECT,
	LOAD_TRANSPOSE,
	LOAD_WARP_TRANSPOSE
};

// Bitfield extract.
DEVICE_INLINE unsigned int bitfieldExtract(unsigned long long source, unsigned int bitStart, unsigned int numberOfBits) {
	const unsigned long long MASK = (1ull << numberOfBits) - 1;
    return (source >> bitStart) & MASK;
}

// Exchange.
template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
DEVICE_INLINE void stripedToBlocked(T (&items)[ITEMS_PER_THREAD], volatile T (&tempStorage)[BLOCK_THREADS * ITEMS_PER_THREAD]) {
	#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int itemOffset = int(i * BLOCK_THREADS) + threadIdx.x;
        tempStorage[itemOffset] = items[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int itemOffset = (threadIdx.x * ITEMS_PER_THREAD) + i;
        items[i] = tempStorage[itemOffset];
    }
}

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
DEVICE_INLINE void warpStripedToBlocked(T (&items)[ITEMS_PER_THREAD], volatile T (&tempStorage)[BLOCK_THREADS * ITEMS_PER_THREAD]) {
	int warpLane = threadIdx.x & (WARP_THREADS - 1);
    int warpIndex = threadIdx.x >> LOG_WARP_THREADS;
	int warpOffset = warpIndex * (MIN(BLOCK_THREADS, WARP_THREADS) * ITEMS_PER_THREAD);
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int itemOffset = warpOffset + (i * MIN(BLOCK_THREADS, WARP_THREADS)) + warpLane;
        tempStorage[itemOffset] = items[i];
    }
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int itemOffset = warpOffset + i + (warpLane * ITEMS_PER_THREAD);
        items[i] = tempStorage[itemOffset];
    }
}

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
DEVICE_INLINE void scatterToStriped(T (&items)[ITEMS_PER_THREAD], volatile T (&tempStorage)[BLOCK_THREADS * ITEMS_PER_THREAD], int (&ranks)[ITEMS_PER_THREAD]) {
	#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        int itemOffset = ranks[ITEM];
        tempStorage[itemOffset] = items[ITEM];
    }
    __syncthreads();
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        int itemOffset = ITEM * BLOCK_THREADS + threadIdx.x;
        items[ITEM] = tempStorage[itemOffset];
    }
}

// Store.
template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
DEVICE_INLINE void storeDirectStriped(T * out, T (&items)[ITEMS_PER_THREAD]) {
	#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        out[(i * BLOCK_THREADS) + threadIdx.x] = items[i];
    }
}

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
DEVICE_INLINE void storeDirectStriped(T * out, T (&items)[ITEMS_PER_THREAD], int validItems) {
	#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((i * BLOCK_THREADS) + threadIdx.x < validItems) {
            out[(i * BLOCK_THREADS) + threadIdx.x] = items[i];
        }
    }
}

// Load.
template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
DEVICE_INLINE void loadDirectStriped(T * in, T (&items)[ITEMS_PER_THREAD]) {
	#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] = in[(i * BLOCK_THREADS) + threadIdx.x];
    }
}

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
DEVICE_INLINE void loadDirectStriped(T * in, T (&items)[ITEMS_PER_THREAD], int validItems) {
	int bounds = validItems - threadIdx.x;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (i * BLOCK_THREADS < bounds) {
            items[i] = in[threadIdx.x + (i * BLOCK_THREADS)];
        }
    }
}

template <int ITEMS_PER_THREAD, typename T>
DEVICE_INLINE void loadDirectBlocked(T * in, T (&items)[ITEMS_PER_THREAD]) {
	#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] = in[(threadIdx.x * ITEMS_PER_THREAD) + i];
    }
}

template <int ITEMS_PER_THREAD, typename T>
DEVICE_INLINE void loadDirectBlocked(T * in, T (&items)[ITEMS_PER_THREAD], int validItems) {
	int bounds = validItems - (threadIdx.x * ITEMS_PER_THREAD);
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (i < bounds) {
            items[i] = in[(threadIdx.x * ITEMS_PER_THREAD) + i];
        }
    }
}

template <int ITEMS_PER_THREAD, typename T>
DEVICE_INLINE void loadDirectWarpStriped(T * in, T (&items)[ITEMS_PER_THREAD]) {
	int threadIndex = threadIdx.x & (WARP_THREADS - 1);
    int warpIndex = threadIdx.x >> LOG_WARP_THREADS;
    int warpOffset = warpIndex * WARP_THREADS * ITEMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] = in[warpOffset + threadIndex + (i * WARP_THREADS)];
    }
}

template <int ITEMS_PER_THREAD, typename T>
DEVICE_INLINE void loadDirectWarpStriped(T * in, T (&items)[ITEMS_PER_THREAD], int validItems) {
	int threadIndex = threadIdx.x & (WARP_THREADS - 1);
    int warpIndex = threadIdx.x >> LOG_WARP_THREADS;
    int warpOffset = warpIndex * WARP_THREADS * ITEMS_PER_THREAD;
	int bounds = validItems - warpOffset - threadIndex;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((i * WARP_THREADS) < bounds) {
            items[i] = in[warpOffset + threadIndex + (i * WARP_THREADS)];
        }
    }
}

// Generic load.
template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T, bool FULL_TILE, LoadAlgorithm ALGORITHM>
struct BlockLoad {
	DEVICE_INLINE static void load(
		T * in,
		T (&items)[ITEMS_PER_THREAD], 
		volatile T (&tempStorage)[ITEMS_PER_THREAD * BLOCK_THREADS], 
		int validItems
	) {}
};

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
struct BlockLoad<ITEMS_PER_THREAD, BLOCK_THREADS, T, true, LOAD_DIRECT> {
	DEVICE_INLINE static void load(
		T * in,
		T (&items)[ITEMS_PER_THREAD], 
		volatile T (&tempStorage)[ITEMS_PER_THREAD * BLOCK_THREADS], 
		int validItems
	) {
		loadDirectBlocked<ITEMS_PER_THREAD, T>(in, items);
	}
};

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
struct BlockLoad<ITEMS_PER_THREAD, BLOCK_THREADS, T, true, LOAD_TRANSPOSE> {
	DEVICE_INLINE static void load(
		T * in,
		T (&items)[ITEMS_PER_THREAD], 
		volatile T (&tempStorage)[ITEMS_PER_THREAD * BLOCK_THREADS], 
		int validItems
	) {
		loadDirectStriped<ITEMS_PER_THREAD, BLOCK_THREADS, T>(in, items);
		stripedToBlocked<ITEMS_PER_THREAD, BLOCK_THREADS, T>(items, tempStorage);
	}
};

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
struct BlockLoad<ITEMS_PER_THREAD, BLOCK_THREADS, T, true, LOAD_WARP_TRANSPOSE> {
	DEVICE_INLINE static void load(
		T * in,
		T (&items)[ITEMS_PER_THREAD], 
		volatile T (&tempStorage)[ITEMS_PER_THREAD * BLOCK_THREADS], 
		int validItems
	) {
		loadDirectWarpStriped<ITEMS_PER_THREAD, T>(in, items);
		warpStripedToBlocked<ITEMS_PER_THREAD, BLOCK_THREADS, T>(items, tempStorage);
	}
};

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
struct BlockLoad<ITEMS_PER_THREAD, BLOCK_THREADS, T, false, LOAD_DIRECT> {
	DEVICE_INLINE static void load(
		T * in,
		T (&items)[ITEMS_PER_THREAD], 
		volatile T (&tempStorage)[ITEMS_PER_THREAD * BLOCK_THREADS], 
		int validItems
	) {
		loadDirectBlocked<ITEMS_PER_THREAD, T>(in, items, validItems);
	}
};

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
struct BlockLoad<ITEMS_PER_THREAD, BLOCK_THREADS, T, false, LOAD_TRANSPOSE> {
	DEVICE_INLINE static void load(
		T * in,
		T (&items)[ITEMS_PER_THREAD], 
		volatile T (&tempStorage)[ITEMS_PER_THREAD * BLOCK_THREADS], 
		int validItems
	) {
		loadDirectStriped<ITEMS_PER_THREAD, BLOCK_THREADS, T>(in, items, validItems);
		stripedToBlocked<ITEMS_PER_THREAD, BLOCK_THREADS, T>(items, tempStorage);
	}
};

template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename T>
struct BlockLoad<ITEMS_PER_THREAD, BLOCK_THREADS, T, false, LOAD_WARP_TRANSPOSE> {
	DEVICE_INLINE static void load(
		T * in,
		T (&items)[ITEMS_PER_THREAD], 
		volatile T (&tempStorage)[ITEMS_PER_THREAD * BLOCK_THREADS], 
		int validItems
	) {
		loadDirectWarpStriped<ITEMS_PER_THREAD, T>(in, items, validItems);
		warpStripedToBlocked<ITEMS_PER_THREAD, BLOCK_THREADS, T>(items, tempStorage);
	}
};

#endif /* _RADIX_SORT_UTIL_H_ */
