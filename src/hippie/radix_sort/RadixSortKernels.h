/**
 * \file	RadixSortKernels.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	RadixSort kernels header file.
 */

#ifndef _RADIX_SORT_KERNELS_H_
#define _RADIX_SORT_KERNELS_H_

#include "RadixSortPolicy.h"

#define UPSWEEP_BLOCK_THREADS 128
#define UPSWEEP_ITEMS_PER_THREAD 15
#define UPSWEEP_RADIX_BITS 5
#define SCAN_BLOCK_THREADS 1024
#define DOWNSWEEP_BLOCK_THREADS 128
#define DOWNSWEEP_ITEMS_PER_THREAD 7
#define DOWNSWEEP_RADIX_BITS 5
#define DOWNSWEEP_SMEM_CONFIG 8
#define DOWNSWEEP_LOAD_ALGORITHM LOAD_DIRECT
#define DOWNSWEEP_OUTER_SCAN true

#if DOWNSWEEP_SMEM_CONFIG == 8
typedef unsigned long long PackedCounter;
#else
typedef unsigned int PackedCounter;
#endif

#ifdef __KERNELCC__
extern "C" {

__launch_bounds__ (UPSWEEP_BLOCK_THREADS, 1)
GLOBAL void upsweepKernel(int numberOfItems, int currentBit, int bigBlocks, int bigShare, int normalShare, 
	int normalBaseOffset, int totalGrains, unsigned long long * keysIn, int * spine, bool alt);

__launch_bounds__ (SCAN_BLOCK_THREADS, 1)
GLOBAL void scanKernel(int numberOfBins, int4 * spine);

__launch_bounds__ (DOWNSWEEP_BLOCK_THREADS, 1)
GLOBAL void downsweepKernel(int currentBit, int numberOfItems, int bigBlocks, int bigShare, int normalShare, int normalBaseOffset, 
	int totalGrains, int * spine, unsigned long long * keysIn, unsigned long long * keysOut, int * valuesIn, int * valuesOut, bool alt);

}
#endif

#endif /* _RADIX_SORT_KERNELS_H_ */
