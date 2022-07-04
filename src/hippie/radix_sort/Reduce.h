/**
 * \file	Reduce.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	A header file containing device thread unrolled 
 *			sequential reduction implemented in templates.
 */

#ifndef _REDUCE_H_
#define _REDUCE_H_

template <typename T, int LENGTH>
struct ThreadReduce {

	DEVICE_INLINE static T reduce(T * input, T prefix) {
		prefix += input[0];
		return ThreadReduce<T, LENGTH - 1>::reduce(input + 1, prefix);
	}

	DEVICE_INLINE static T reduce(T * input) {
		T prefix = input[0];
		return ThreadReduce<T, LENGTH - 1>::reduce(input + 1, prefix);
	}

};

template <typename T>
struct ThreadReduce<T, 0> {
    
	DEVICE_INLINE static T reduce(T * input, T prefix) {
		return prefix;
	}

};

#endif /* _REDUCE_H_ */
