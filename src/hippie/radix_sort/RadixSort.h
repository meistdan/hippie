/**
 * \file	RadixSort.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	RadixSort class header file.
 */

#ifndef _RADIX_SORT_H_
#define _RADIX_SORT_H_

#include "gpu/Buffer.h"

class RadixSort {

private:

	struct UpsweepConfig {
		int blockThreads;
	} upsweepConfig;

	struct ScanConfing {
		int blockThreads;
		int tileSize;
	} scanConfig;

	struct DownsweepConfig {
		int blockThreads;
        int tileSize;
        int radixBits;
		int altRadixBits;
		int smemConfig;
        int maxGridSize;
		int altMaxGridSize;
	} downsweepConfig;

	HipCompiler compiler;

	void init(void);

	int getMaxGridSize(bool alt);

	float dispatch(
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
	);

public:

	RadixSort(void);
	~RadixSort(void);

	bool getSwap(int beginBit, int endBit);

	float sort(
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
	);

	float sort(
		Buffer & keys1,
		Buffer & keys2,
		Buffer & values1,
		Buffer & values2,
		Buffer & spine,
		bool & swapBuffers,
		int numberOfItems,
		int beginBit,
		int endBit
	);

};

#endif /* _RADIX_SORT_H_ */
