#pragma once
#include "Globals.h"

 //------------------------------------------------------------------------
 // Constants.
 //------------------------------------------------------------------------

enum
{
    MaxBlockHeight = 6,            // Upper bound for blockDim.y.
    EntrypointSentinel = 0x76543210,   // Bottom-most stack entry, indicating the end of traversal.
};

//------------------------------------------------------------------------
// Kernel configuration. Written by queryConfig() in each CU file.
//------------------------------------------------------------------------

struct KernelConfig
{
    int         blockWidth;             // Desired blockDim.x.
    int         blockHeight;            // Desired blockDim.y.
    int         desiredWarps;           // Desired warps.
};

//------------------------------------------------------------------------
// Function signature for trace().
//------------------------------------------------------------------------

#define TRACE_FUNC \
    extern "C" GLOBAL void trace( \
        int             numRays,        /* Total number of rays in the batch. */ \
        bool            anyHit,         /* False if rays need to find the closest hit. */ \
        float4*         rays,           /* Ray input: float3 origin, float tmin, float3 direction, float tmax. */ \
        int4*           results,        /* Ray output: int triangleID, float hitT, int2 padding. */ \
        float4*         nodes,          /* AOS: 64/128/256 bytes per node. */ \
        float4*         tris,           /* AOS: 48 bytes per triangle. */ \
		int2*			stats,			/* Traversed nodes, tested triangles. */ \
		int*			indices,		/* Ray indices */ \
        int*            triIndices)     /* Triangle index remapping table. */

#define TRACE_FUNC_STATS \
    extern "C" GLOBAL void traceStats( \
        int             numRays,        /* Total number of rays in the batch. */ \
        bool            anyHit,         /* False if rays need to find the closest hit. */ \
        float4*         rays,           /* Ray input: float3 origin, float tmin, float3 direction, float tmax. */ \
        int4*           results,        /* Ray output: int triangleID, float hitT, int2 padding. */ \
        float4*         nodes,          /* AOS: 64 bytes per node. */ \
        float4*         tris,           /* AOS: 48 bytes per triangle. */ \
		int2*			stats,			/* Traversed nodes, tested triangles. */ \
		int*			indices,		/* Ray indices */ \
        int*            triIndices)     /* Triangle index remapping table. */

#define TRACE_FUNC_SORT \
    extern "C" GLOBAL void traceSort( \
        int             numRays,        /* Total number of rays in the batch. */ \
        bool            anyHit,         /* False if rays need to find the closest hit. */ \
        float4*         rays,           /* Ray input: float3 origin, float tmin, float3 direction, float tmax. */ \
        int4*           results,        /* Ray output: int triangleID, float hitT, int2 padding. */ \
        float4*         nodes,          /* AOS: 64 bytes per node. */ \
        float4*         tris,           /* AOS: 48 bytes per triangle. */ \
		int2*			stats,			/* Traversed nodes, tested triangles. */ \
		int*			indices,		/* Ray indices */ \
        int*            triIndices)     /* Triangle index remapping table. */

#define TRACE_FUNC_STATS_SORT \
    extern "C" GLOBAL void traceStatsSort( \
        int             numRays,        /* Total number of rays in the batch. */ \
        bool            anyHit,         /* False if rays need to find the closest hit. */ \
        float4*         rays,           /* Ray input: float3 origin, float tmin, float3 direction, float tmax. */ \
        int4*           results,        /* Ray output: int triangleID, float hitT, int2 padding. */ \
        float4*         nodes,          /* AOS: 64 bytes per node. */ \
        float4*         tris,           /* AOS: 48 bytes per triangle. */ \
		int2*			stats,			/* Traversed nodes, tested triangles. */ \
		int*			indices,	    /* Ray indices */ \
        int*            triIndices)     /* Triangle index remapping table. */

//------------------------------------------------------------------------
// Temporary data stored in shared memory to reduce register pressure.
//------------------------------------------------------------------------

struct RayStruct
{
    float   idirx;  // 1.0f / ray.direction.x
    float   idiry;  // 1.0f / ray.direction.y
    float   idirz;  // 1.0f / ray.direction.z
    float   tmin;   // ray.tmin
    float   dummy;  // Padding to avoid bank conflicts.
};

//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------

#ifdef __KERNELCC__
extern "C"
{
    DEVICE int g_warpCounter;
    DEVICE KernelConfig g_config;   // Output of queryConfig().
    GLOBAL void queryConfig(void);  // Launched once when the kernel is loaded.
    TRACE_FUNC;                         // Launched for each batch of rays.
}
#endif

//------------------------------------------------------------------------
// Utilities.
//------------------------------------------------------------------------

#define FETCH_GLOBAL(NAME, IDX, TYPE) ((const TYPE*)NAME)[IDX]
#define STORE_RESULT(RAY, TRI, T) ((int2*)results)[(RAY) * 2] = make_int2(TRI, __float_as_int(T))
#define STORE_RESULT_WITH_BAR_COORDS(RAY, TRI, T, U, V) (results[RAY] = make_int4(TRI, __float_as_int(T), __float_as_int(U), __float_as_int(V)))
#define STORE_STATS(RAY, N, T) (stats[RAY] = make_int2(N, T))

//------------------------------------------------------------------------

#ifdef __KERNELCC__

template <class T> DEVICE_INLINE void swap(T& a, T& b)
{
    T t = a;
    a = b;
    b = t;
}

DEVICE_INLINE float min4(float a, float b, float c, float d)
{
    return fminf(fminf(fminf(a, b), c), d);
}

DEVICE_INLINE float max4(float a, float b, float c, float d)
{
    return fmaxf(fmaxf(fmaxf(a, b), c), d);
}

DEVICE_INLINE float min3(float a, float b, float c)
{
    return fminf(fminf(a, b), c);
}

DEVICE_INLINE float max3(float a, float b, float c)
{
    return fmaxf(fmaxf(a, b), c);
}

// Using integer min,max
DEVICE_INLINE float fminf2(float a, float b)
{
    int a2 = __float_as_int(a);
    int b2 = __float_as_int(b);
    return __int_as_float(a2 < b2 ? a2 : b2);
}

DEVICE_INLINE float fmaxf2(float a, float b)
{
    int a2 = __float_as_int(a);
    int b2 = __float_as_int(b);
    return __int_as_float(a2 > b2 ? a2 : b2);
}

// Using video instructions.
#ifdef __CUDACC__
DEVICE_INLINE int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
DEVICE_INLINE int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
DEVICE_INLINE int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
DEVICE_INLINE int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
#else
DEVICE_INLINE int   min_min(int a, int b, int c) { return min(min(a, b), c); }
DEVICE_INLINE int   min_max(int a, int b, int c) { return max(min(a, b), c); }
DEVICE_INLINE int   max_min(int a, int b, int c) { return min(max(a, b), c); }
DEVICE_INLINE int   max_max(int a, int b, int c) { return max(max(a, b), c); }
#endif
DEVICE_INLINE float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
DEVICE_INLINE float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
DEVICE_INLINE float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
DEVICE_INLINE float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }


DEVICE_INLINE float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
    float t1 = fmin_fmax(a0, a1, d);
    float t2 = fmin_fmax(b0, b1, t1);
    float t3 = fmin_fmax(c0, c1, t2);
    return t3;
}

DEVICE_INLINE float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
    float t1 = fmax_fmin(a0, a1, d);
    float t2 = fmax_fmin(b0, b1, t1);
    float t3 = fmax_fmin(c0, c1, t2);
    return t3;
}

// Experimentally determined best mix of float/int/video minmax instructions for Kepler.
DEVICE_INLINE float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
DEVICE_INLINE float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }

// Same for Fermi.
DEVICE_INLINE float spanBeginFermi(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return magic_max7(a0, a1, b0, b1, c0, c1, d); }
DEVICE_INLINE float spanEndFermi(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return magic_min7(a0, a1, b0, b1, c0, c1, d); }

// Sorting
DEVICE_INLINE void swap(float& dist, int& index, const int mask, const unsigned int dir)
{
    const float comp_dist = __shfl_xor(dist, mask);
    const int comp_index = __shfl_xor(index, mask);
    const bool flip = (dist != comp_dist) && (dist > comp_dist == dir);
    index = flip ? index : comp_index;
    dist = flip ? dist : comp_dist;
}

DEVICE_INLINE unsigned int bfe(const unsigned int i, const unsigned int k)
{
#if __CUDACC__
    unsigned int ret;
    asm("{bfe.u32 %0, %1, %2, 1;}" : "=r"(ret) : "r"(i), "r"(k));
    return ret;
#else
    return (i >> k) & 1;
#endif
}

#endif

//------------------------------------------------------------------------
