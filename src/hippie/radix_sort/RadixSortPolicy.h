/**
 * \file	RadixSortPolicy.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	A header file containing device information macros.
 */

#ifndef _RADIX_SORT_POLICY_H_
#define _RADIX_SORT_POLICY_H_

#include "Globals.h"

// Log2 macros.
#define LOG2(n) ((n&2)?1:0)
#define LOG4(n) ((n&(0xC))?(2+LOG2(n>>2)):(LOG2(n)))
#define LOG8(n) ((n&0xF0)?(4+LOG4(n>>4)):(LOG4(n)))
#define LOG16(n) ((n&0xFF00)?(8+LOG8(n>>8)):(LOG8(n)))
#define LOG32(n) ((n&0xFFFF0000)?(16+LOG16(n>>16)):(LOG16(n)))
#define LOG(n) (n==0?0:LOG32(n))

// Logarithm of number of smem banks.
#define LOG_SMEM_BANKS(arch) ((arch >= 200) ? (5) : (4))

/// Number of smem banks.
#define SMEM_BANKS(arch) (1 << LOG_SMEM_BANKS(arch))

// Number of bytes per smem bank.
#define SMEM_BANK_BYTES (4)

// Number of smem bytes provisioned per SM.
#define SMEM_BYTES(arch) ((arch >= 200) ? (48 * 1024) : (16 * 1024))

// Smem allocation size in bytes.
#define SMEM_ALLOC_UNIT(arch) ((arch >= 300) ? (256) : ((arch >= 200) ? (128) : (512)))

// Whether or not the architecture allocates registers by block (or by warp).
#define REGS_BY_BLOCK(arch) ((arch >= 200) ? (false) : (true))

// Number of registers allocated at a time per block (or by warp).
#define REG_ALLOC_UNIT(arch) ((arch >= 300) ? (256) : ((arch >= 200) ? (64) : ((arch >= 120) ? (512) : (256))))

// Granularity of warps for which registers are allocated.
#define WARP_ALLOC_UNIT(arch) ((arch >= 300) ? (4) : (2))

// Maximum number of threads per SM.
#define MAX_SM_THREADS(arch) ((arch >= 300) ? (2048) : ((arch >= 200) ? (1536) : ((arch >= 120) ? (1024) : (768))))

// Maximum number of thread blocks per SM.
#define MAX_SM_BLOCKS(arch) ((arch >= 300) ? (16) : (8))

// Maximum number of threads per thread block.
#define MAX_BLOCK_THREADS(arch) ((arch >= 200) ? (1024) : (512))

// Maximum number of registers per SM.
#define MAX_SM_REGISTERS(arch) ((arch >= 300) ? (64 * 1024) : ((arch >= 200) ? (32 * 1024) : ((arch >= 120) ? (16 * 1024) : (8 * 1024))))

// Oversubscription factor.
#define SUBSCRIPTION_FACTOR(arch) ((arch >= 300) ? (5) : (3))

#endif /* _RADIX_SORT_POLICY_H_ */
