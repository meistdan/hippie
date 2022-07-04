/**
 * \file	Scan.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	A header file containing device thread unrolled 
 *			sequential prefix scan implemented in templates.
 */


#ifndef _SCAN_H_
#define _SCAN_H_

#include "RadixSortUtil.h"

template <typename T, int LENGTH>
struct ThreadScanExclusive {
    
	DEVICE_INLINE static T scan(T inclusive, T exclusive, T * input) {
		inclusive = exclusive + *input;
		input[0] = exclusive;
		exclusive = inclusive;
		return ThreadScanExclusive<T, LENGTH - 1>::scan(inclusive, exclusive, input + 1);
	}

	DEVICE_INLINE static T scan(T * input, T prefix) {
		T inclusive = input[0] + prefix;
		input[0] = prefix;
		T exclusive = inclusive;
		return ThreadScanExclusive<T, LENGTH - 1>::scan(inclusive, exclusive, input + 1);
	}

};

template <typename T>
struct ThreadScanExclusive<T, 0> {

	DEVICE_INLINE static T scan(T inclusive, T exclusive, T * input) {
		return inclusive;
	}

};


#endif /* _SCAN_H_ */
