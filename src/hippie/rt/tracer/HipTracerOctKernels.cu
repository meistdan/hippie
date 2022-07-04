#include "rt/tracer/HipTracerKernels.h"

//------------------------------------------------------------------------

#define STACK_SIZE              128         // Size of the traversal stack in local memory.
#define BLOCK_HEIGHT             2
#define NUM_SUBWARPS             4
#define SUBWARP_WIDTH            8
#define HIP_INF_F __int_as_float(0x7f800000)
//#define STACKED_T

//------------------------------------------------------------------------

extern "C" GLOBAL void queryConfig(void)
{
	g_config.blockWidth = 32;
	g_config.blockHeight = BLOCK_HEIGHT;
	g_config.desiredWarps = 960;
}

//------------------------------------------------------------------------

TRACE_FUNC
{
// Traversal stack in HIP thread-local memory.
__shared__ volatile int traversalStack[NUM_SUBWARPS][BLOCK_HEIGHT][STACK_SIZE];

#ifdef STACKED_T
__shared__ volatile float tStack[NUM_SUBWARPS][BLOCK_HEIGHT][STACK_SIZE];
#endif

// Live state during traversal, stored in registers.

float   origx, origy, origz;            // Ray origin.
float   tmin;
float   hitT;
int     rayidx;
int     stackPtr = -1;
float   oodx, oody, oodz;
float   dirx, diry, dirz;
float   idirx, idiry, idirz;
int	    hitIndex = -1;
float   hitU, hitV;

const int offset = (threadIdx.x & 0x00000007);
const int subwarp = (threadIdx.x >> 3);
const int subwarp_mask = (0x000000ff << (threadIdx.x & 0xfffffff8));

// Initialize persistent threads.
// Persistent threads: fetch and process rays in a loop.
do
{
	// Fetch new rays from the global pool using lane 0.
	if (stackPtr < 0)
	{
		if (threadIdx.x == 0)
			rayidx = atomicAdd(&g_warpCounter, NUM_SUBWARPS);
		rayidx = __shfl(rayidx, 0) + subwarp;

		if (rayidx >= numRays)
			break;

		// Fetch ray.

		const float4 o = FETCH_GLOBAL(rays, rayidx * 2 + 0, float4);
		const float4 d = FETCH_GLOBAL(rays, rayidx * 2 + 1, float4);
		origx = o.x;
		origy = o.y;
		origz = o.z;
		tmin = o.w;
		dirx = d.x;
		diry = d.y;
		dirz = d.z;
		hitT = d.w;
		const float ooeps = exp2f(-80.0f); // Avoid div by zero.
		idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
		idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
		idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
		oodx = origx * idirx;
		oody = origy * idiry;
		oodz = origz * idirz;
		// Setup traversal.
		stackPtr = 0;
		hitIndex = -1;
		if (!offset)
			traversalStack[subwarp][threadIdx.y][0] = 0;
#ifdef STACKED_T
		tStack[subwarp][threadIdx.y][offset] = 0.f;
#endif

	}

	// Traversal loop.

	while (stackPtr >= 0)
	{

#ifdef STACKED_T
		if (tStack[subwarp][threadIdx.y][stackPtr] > hitT)
		{
			--stackPtr;
			continue;
		}
#endif

		const int curr = traversalStack[subwarp][threadIdx.y][stackPtr--];

		if (curr >= 0)
		{
			// Fetch AABBs of the two child nodes.
			const float4 xy = nodes[16 * curr + offset];
			const float4 zi = nodes[16 * curr + SUBWARP_WIDTH + offset];

			// Intersect the ray against the child nodes.
			const float lox = xy.x * idirx - oodx;
			const float hix = xy.y * idirx - oodx;
			const float loy = xy.z * idiry - oody;
			const float hiy = xy.w * idiry - oody;		
			const float loz = zi.x * idirz - oodz;
			const float hiz = zi.y * idirz - oodz;
			int link = __float_as_int(zi.z);


			const float cmin = spanBeginKepler(lox, hix, loy, hiy, loz, hiz, tmin);
			const bool hit = cmin <= spanEndKepler(lox, hix, loy, hiy, loz, hiz, hitT);
			

			float dist = hit ? cmin : HIP_INF_F;
			const int  hits = __popc(__ballot(hit)&subwarp_mask);
			if (!hits) continue;
			stackPtr += hits;


			//sort hits
			swap(dist, link, 0x01, bfe(threadIdx.x, 1) ^ bfe(threadIdx.x, 0));
			swap(dist, link, 0x02, bfe(threadIdx.x, 2) ^ bfe(threadIdx.x, 1));
			swap(dist, link, 0x01, bfe(threadIdx.x, 2) ^ bfe(threadIdx.x, 0));
			swap(dist, link, 0x04, bfe(threadIdx.x, 2));
			swap(dist, link, 0x02, bfe(threadIdx.x, 1));
			swap(dist, link, 0x01, bfe(threadIdx.x, 0));

			if (dist < HIP_INF_F)
			{
				traversalStack[subwarp][threadIdx.y][stackPtr - offset] = link;
#ifdef STACKED_T
				tStack[subwarp][threadIdx.y][stackPtr - offset] = dist;
#endif
			}
		}
		else {

            // Triangle bounds.
            const float4 triBounds = nodes[16 * (~curr) + SUBWARP_WIDTH];

            bool hit = false;
            for (int triAddr = __float_as_int(triBounds.z) + offset; triAddr < __float_as_int(triBounds.w); triAddr += SUBWARP_WIDTH)
            {
                
                // Tris in TEX (good to fetch as a single batch)
                const float4 v00 = tris[3 * triAddr + 0];
                const float4 v11 = tris[3 * triAddr + 1];
                const float4 v22 = tris[3 * triAddr + 2];

                const float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                const float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                float t = Oz * invDz;

                if (t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

                    const float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                    const 	float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                    const 	float u = Ox + t*Dx;

                    if (u >= 0.0f)
                    {
                        // Compute and check barycentric v.

                        const float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                        const float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                        const float v = Oy + t*Dy;

                        if (v >= 0.0f && u + v <= 1.0f)
                        {
                            // Record intersection.
                            // Closest intersection not required => terminate.
                            hit = true;
                            hitT = t;
                            hitU = u;
                            hitV = v;
                            hitIndex = triAddr;
                        }
                    }
                }
            }

			// Sort triangles

			const int hits = __ballot(hit)&subwarp_mask;
			if (!hits) continue;

            hitT = hit ? hitT : HIP_INF_F;

			#pragma unroll
			for (int mask = 4; mask > 0; mask >>= 1) {
                const float tmp_t = __shfl_xor(hitT, mask);
                const int tmp_addr = __shfl_xor(hitIndex, mask);
                const float tmp_u = __shfl_xor(hitU, mask);
                const float tmp_v = __shfl_xor(hitV, mask);
                hitIndex = tmp_t < hitT ? tmp_addr : hitIndex;
                hitU = tmp_t < hitT ? tmp_u : hitU;
                hitV = tmp_t < hitT ? tmp_v : hitV;
                hitT = fminf(hitT, tmp_t);
			}

			if (anyHit && hits)
			{
				stackPtr = -1;
				break;
			}

		}
	}

	if (offset == 0) {
		if (hitIndex == -1) { STORE_RESULT(rayidx, -1, hitT); }
		else { STORE_RESULT_WITH_BAR_COORDS(rayidx, triIndices[hitIndex], hitT, hitU, hitV); }
	}

} while (true);
}

//------------------------------------------------------------------------

TRACE_FUNC_STATS
{
    // Traversal stack in HIP thread-local memory.
    __shared__ volatile int traversalStack[NUM_SUBWARPS][BLOCK_HEIGHT][STACK_SIZE];

#ifdef STACKED_T
__shared__ volatile float tStack[NUM_SUBWARPS][BLOCK_HEIGHT][STACK_SIZE];
#endif

// Live state during traversal, stored in registers.

float   origx, origy, origz;            // Ray origin.
float   tmin;
float   hitT;
int     rayidx;
int     stackPtr = -1;
float   oodx, oody, oodz;
float   dirx, diry, dirz;
float   idirx, idiry, idirz;
int	    hitIndex = -1;
float   hitU, hitV;
int traversedNodes;
int testedTriangles;

const int offset = (threadIdx.x & 0x00000007);
const int subwarp = (threadIdx.x >> 3);
const int subwarp_mask = (0x000000ff << (threadIdx.x & 0xfffffff8));

// Initialize persistent threads.
// Persistent threads: fetch and process rays in a loop.
do
{
    // Fetch new rays from the global pool using lane 0.
    if (stackPtr < 0)
    {
        traversedNodes = 0;
        testedTriangles = 0;

        if (threadIdx.x == 0)
            rayidx = atomicAdd(&g_warpCounter, NUM_SUBWARPS);
        rayidx = __shfl(rayidx, 0) + subwarp;

        if (rayidx >= numRays)
            break;

        // Fetch ray.

        const float4 o = FETCH_GLOBAL(rays, rayidx * 2 + 0, float4);
        const float4 d = FETCH_GLOBAL(rays, rayidx * 2 + 1, float4);
        origx = o.x;
        origy = o.y;
        origz = o.z;
        tmin = o.w;
        dirx = d.x;
        diry = d.y;
        dirz = d.z;
        hitT = d.w;
        const float ooeps = exp2f(-80.0f); // Avoid div by zero.
        idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
        idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
        idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
        oodx = origx * idirx;
        oody = origy * idiry;
        oodz = origz * idirz;
        // Setup traversal.
        stackPtr = 0;
        hitIndex = -1;
        if (!offset)
            traversalStack[subwarp][threadIdx.y][0] = 0;
#ifdef STACKED_T
        tStack[subwarp][threadIdx.y][offset] = 0.f;
#endif

    }

    // Traversal loop.

    while (stackPtr >= 0)
    {

#ifdef STACKED_T
        if (tStack[subwarp][threadIdx.y][stackPtr] > hitT)
        {
            --stackPtr;
            continue;
        }
#endif

        const int curr = traversalStack[subwarp][threadIdx.y][stackPtr--];

        if (curr >= 0)
        {
            if (offset == 0) ++traversedNodes;

            // Fetch AABBs of the two child nodes.
            const float4 xy = nodes[16 * curr + offset];
            const float4 zi = nodes[16 * curr + SUBWARP_WIDTH + offset];

            // Intersect the ray against the child nodes.
            const float lox = xy.x * idirx - oodx;
            const float hix = xy.y * idirx - oodx;
            const float loy = xy.z * idiry - oody;
            const float hiy = xy.w * idiry - oody;
            const float loz = zi.x * idirz - oodz;
            const float hiz = zi.y * idirz - oodz;
            int link = __float_as_int(zi.z);


            const float cmin = spanBeginKepler(lox, hix, loy, hiy, loz, hiz, tmin);
            const bool hit = cmin <= spanEndKepler(lox, hix, loy, hiy, loz, hiz, hitT);


            float dist = hit ? cmin : HIP_INF_F;
            const int  hits = __popc(__ballot(hit)&subwarp_mask);
            if (!hits) continue;
            stackPtr += hits;


            //sort hits
            swap(dist, link, 0x01, bfe(threadIdx.x, 1) ^ bfe(threadIdx.x, 0));
            swap(dist, link, 0x02, bfe(threadIdx.x, 2) ^ bfe(threadIdx.x, 1));
            swap(dist, link, 0x01, bfe(threadIdx.x, 2) ^ bfe(threadIdx.x, 0));
            swap(dist, link, 0x04, bfe(threadIdx.x, 2));
            swap(dist, link, 0x02, bfe(threadIdx.x, 1));
            swap(dist, link, 0x01, bfe(threadIdx.x, 0));

            if (dist < HIP_INF_F)
            {
                traversalStack[subwarp][threadIdx.y][stackPtr - offset] = link;
#ifdef STACKED_T
                tStack[subwarp][threadIdx.y][stackPtr - offset] = dist;
#endif
            }
        }
        else {

            // Triangle bounds.
            const float4 triBounds = nodes[16 * (~curr) + SUBWARP_WIDTH];

            bool hit = false;
            for (int triAddr = __float_as_int(triBounds.z) + offset; triAddr < __float_as_int(triBounds.w); triAddr += SUBWARP_WIDTH)
            {
                ++testedTriangles;

                // Tris in TEX (good to fetch as a single batch)
                const float4 v00 = tris[3 * triAddr + 0];
                const float4 v11 = tris[3 * triAddr + 1];
                const float4 v22 = tris[3 * triAddr + 2];

                const float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                const float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                float t = Oz * invDz;

                if (t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

                    const float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                    const 	float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                    const 	float u = Ox + t*Dx;

                    if (u >= 0.0f)
                    {
                        // Compute and check barycentric v.

                        const float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                        const float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                        const float v = Oy + t*Dy;

                        if (v >= 0.0f && u + v <= 1.0f)
                        {
                            // Record intersection.
                            // Closest intersection not required => terminate.
                            hit = true;
                            hitT = t;
                            hitU = u;
                            hitV = v;
                            hitIndex = triAddr;
                        }
                    }
                }
            }

            // Sort triangles

            const int hits = __ballot(hit)&subwarp_mask;
            if (!hits) continue;

            hitT = hit ? hitT : HIP_INF_F;

#pragma unroll
            for (int mask = 4; mask > 0; mask >>= 1) {
                const float tmp_t = __shfl_xor(hitT, mask);
                const int tmp_addr = __shfl_xor(hitIndex, mask);
                const float tmp_u = __shfl_xor(hitU, mask);
                const float tmp_v = __shfl_xor(hitV, mask);
                hitIndex = tmp_t < hitT ? tmp_addr : hitIndex;
                hitU = tmp_t < hitT ? tmp_u : hitU;
                hitV = tmp_t < hitT ? tmp_v : hitV;
                hitT = fminf(hitT, tmp_t);
            }

            if (anyHit && hits)
            {
                stackPtr = -1;
                break;
            }

        }
    }

    if (offset == 0) {
        if (hitIndex == -1) { 
            STORE_RESULT(rayidx, -1, hitT); 
            STORE_STATS(rayidx, 0, 0);
        }
        else { 
            STORE_RESULT_WITH_BAR_COORDS(rayidx, triIndices[hitIndex], hitT, hitU, hitV);
            STORE_STATS(rayidx, traversedNodes, testedTriangles);
        }
    }

} while (true);
}

//------------------------------------------------------------------------

TRACE_FUNC_SORT
{
    // Traversal stack in HIP thread-local memory.
    __shared__ volatile int traversalStack[NUM_SUBWARPS][BLOCK_HEIGHT][STACK_SIZE];

#ifdef STACKED_T
__shared__ volatile float tStack[NUM_SUBWARPS][BLOCK_HEIGHT][STACK_SIZE];
#endif

// Live state during traversal, stored in registers.

float   origx, origy, origz;            // Ray origin.
float   tmin;
float   hitT;
int     rayidx;
int     stackPtr = -1;
float   oodx, oody, oodz;
float   dirx, diry, dirz;
float   idirx, idiry, idirz;
int	    hitIndex = -1;
float   hitU, hitV;

const int offset = (threadIdx.x & 0x00000007);
const int subwarp = (threadIdx.x >> 3);
const int subwarp_mask = (0x000000ff << (threadIdx.x & 0xfffffff8));

// Initialize persistent threads.
// Persistent threads: fetch and process rays in a loop.
do
{
    // Fetch new rays from the global pool using lane 0.
    if (stackPtr < 0)
    {
        if (threadIdx.x == 0)
            rayidx = atomicAdd(&g_warpCounter, NUM_SUBWARPS);
        rayidx = __shfl(rayidx, 0) + subwarp;

        if (rayidx >= numRays)
            break;

        // Fetch ray.

        rayidx = indices[rayidx];
        const float4 o = FETCH_GLOBAL(rays, rayidx * 2 + 0, float4);
        const float4 d = FETCH_GLOBAL(rays, rayidx * 2 + 1, float4);
        origx = o.x;
        origy = o.y;
        origz = o.z;
        tmin = o.w;
        dirx = d.x;
        diry = d.y;
        dirz = d.z;
        hitT = d.w;
        const float ooeps = exp2f(-80.0f); // Avoid div by zero.
        idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
        idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
        idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
        oodx = origx * idirx;
        oody = origy * idiry;
        oodz = origz * idirz;
        // Setup traversal.
        stackPtr = 0;
        hitIndex = -1;
        if (!offset)
            traversalStack[subwarp][threadIdx.y][0] = 0;
#ifdef STACKED_T
        tStack[subwarp][threadIdx.y][offset] = 0.f;
#endif

    }

    // Traversal loop.

    while (stackPtr >= 0)
    {

#ifdef STACKED_T
        if (tStack[subwarp][threadIdx.y][stackPtr] > hitT)
        {
            --stackPtr;
            continue;
        }
#endif

        const int curr = traversalStack[subwarp][threadIdx.y][stackPtr--];

        if (curr >= 0)
        {
            // Fetch AABBs of the two child nodes.
            const float4 xy = nodes[16 * curr + offset];
            const float4 zi = nodes[16 * curr + SUBWARP_WIDTH + offset];

            // Intersect the ray against the child nodes.
            const float lox = xy.x * idirx - oodx;
            const float hix = xy.y * idirx - oodx;
            const float loy = xy.z * idiry - oody;
            const float hiy = xy.w * idiry - oody;
            const float loz = zi.x * idirz - oodz;
            const float hiz = zi.y * idirz - oodz;
            int link = __float_as_int(zi.z);


            const float cmin = spanBeginKepler(lox, hix, loy, hiy, loz, hiz, tmin);
            const bool hit = cmin <= spanEndKepler(lox, hix, loy, hiy, loz, hiz, hitT);


            float dist = hit ? cmin : HIP_INF_F;
            const int  hits = __popc(__ballot(hit)&subwarp_mask);
            if (!hits) continue;
            stackPtr += hits;


            //sort hits
            swap(dist, link, 0x01, bfe(threadIdx.x, 1) ^ bfe(threadIdx.x, 0));
            swap(dist, link, 0x02, bfe(threadIdx.x, 2) ^ bfe(threadIdx.x, 1));
            swap(dist, link, 0x01, bfe(threadIdx.x, 2) ^ bfe(threadIdx.x, 0));
            swap(dist, link, 0x04, bfe(threadIdx.x, 2));
            swap(dist, link, 0x02, bfe(threadIdx.x, 1));
            swap(dist, link, 0x01, bfe(threadIdx.x, 0));

            if (dist < HIP_INF_F)
            {
                traversalStack[subwarp][threadIdx.y][stackPtr - offset] = link;
#ifdef STACKED_T
                tStack[subwarp][threadIdx.y][stackPtr - offset] = dist;
#endif
            }
        }
        else {

            // Triangle bounds.
            const float4 triBounds = nodes[16 * (~curr) + SUBWARP_WIDTH];

            bool hit = false;
            for (int triAddr = __float_as_int(triBounds.z) + offset; triAddr < __float_as_int(triBounds.w); triAddr += SUBWARP_WIDTH)
            {

                // Tris in TEX (good to fetch as a single batch)
                const float4 v00 = tris[3 * triAddr + 0];
                const float4 v11 = tris[3 * triAddr + 1];
                const float4 v22 = tris[3 * triAddr + 2];

                const float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                const float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                float t = Oz * invDz;

                if (t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

                    const float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                    const 	float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                    const 	float u = Ox + t*Dx;

                    if (u >= 0.0f)
                    {
                        // Compute and check barycentric v.

                        const float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                        const float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                        const float v = Oy + t*Dy;

                        if (v >= 0.0f && u + v <= 1.0f)
                        {
                            // Record intersection.
                            // Closest intersection not required => terminate.
                            hit = true;
                            hitT = t;
                            hitU = u;
                            hitV = v;
                            hitIndex = triAddr;
                        }
                    }
                }
            }

            // Sort triangles

            const int hits = __ballot(hit)&subwarp_mask;
            if (!hits) continue;

            hitT = hit ? hitT : HIP_INF_F;

#pragma unroll
            for (int mask = 4; mask > 0; mask >>= 1) {
                const float tmp_t = __shfl_xor(hitT, mask);
                const int tmp_addr = __shfl_xor(hitIndex, mask);
                const float tmp_u = __shfl_xor(hitU, mask);
                const float tmp_v = __shfl_xor(hitV, mask);
                hitIndex = tmp_t < hitT ? tmp_addr : hitIndex;
                hitU = tmp_t < hitT ? tmp_u : hitU;
                hitV = tmp_t < hitT ? tmp_v : hitV;
                hitT = fminf(hitT, tmp_t);
            }

            if (anyHit && hits)
            {
                stackPtr = -1;
                break;
            }

        }
    }

    if (offset == 0) {
        if (hitIndex == -1) { STORE_RESULT(rayidx, -1, hitT); }
        else { STORE_RESULT_WITH_BAR_COORDS(rayidx, triIndices[hitIndex], hitT, hitU, hitV); }
    }

} while (true);
}

//------------------------------------------------------------------------

TRACE_FUNC_STATS_SORT
{
    // Traversal stack in HIP thread-local memory.
    __shared__ volatile int traversalStack[NUM_SUBWARPS][BLOCK_HEIGHT][STACK_SIZE];

#ifdef STACKED_T
__shared__ volatile float tStack[NUM_SUBWARPS][BLOCK_HEIGHT][STACK_SIZE];
#endif

// Live state during traversal, stored in registers.

float   origx, origy, origz;            // Ray origin.
float   tmin;
float   hitT;
int     rayidx;
int     stackPtr = -1;
float   oodx, oody, oodz;
float   dirx, diry, dirz;
float   idirx, idiry, idirz;
int	    hitIndex = -1;
float   hitU, hitV;
int traversedNodes;
int testedTriangles;

const int offset = (threadIdx.x & 0x00000007);
const int subwarp = (threadIdx.x >> 3);
const int subwarp_mask = (0x000000ff << (threadIdx.x & 0xfffffff8));

// Initialize persistent threads.
// Persistent threads: fetch and process rays in a loop.
do
{
    // Fetch new rays from the global pool using lane 0.
    if (stackPtr < 0)
    {
        traversedNodes = 0;
        testedTriangles = 0;

        if (threadIdx.x == 0)
            rayidx = atomicAdd(&g_warpCounter, NUM_SUBWARPS);
        rayidx = __shfl(rayidx, 0) + subwarp;

        if (rayidx >= numRays)
            break;

        // Fetch ray.

        rayidx = indices[rayidx];
        const float4 o = FETCH_GLOBAL(rays, rayidx * 2 + 0, float4);
        const float4 d = FETCH_GLOBAL(rays, rayidx * 2 + 1, float4);
        origx = o.x;
        origy = o.y;
        origz = o.z;
        tmin = o.w;
        dirx = d.x;
        diry = d.y;
        dirz = d.z;
        hitT = d.w;
        const float ooeps = exp2f(-80.0f); // Avoid div by zero.
        idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
        idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
        idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
        oodx = origx * idirx;
        oody = origy * idiry;
        oodz = origz * idirz;
        // Setup traversal.
        stackPtr = 0;
        hitIndex = -1;
        if (!offset)
            traversalStack[subwarp][threadIdx.y][0] = 0;
#ifdef STACKED_T
        tStack[subwarp][threadIdx.y][offset] = 0.f;
#endif

    }

    // Traversal loop.

    while (stackPtr >= 0)
    {

#ifdef STACKED_T
        if (tStack[subwarp][threadIdx.y][stackPtr] > hitT)
        {
            --stackPtr;
            continue;
        }
#endif

        const int curr = traversalStack[subwarp][threadIdx.y][stackPtr--];

        if (curr >= 0)
        {
            if (offset == 0) ++traversedNodes;

            // Fetch AABBs of the two child nodes.
            const float4 xy = nodes[16 * curr + offset];
            const float4 zi = nodes[16 * curr + SUBWARP_WIDTH + offset];

            // Intersect the ray against the child nodes.
            const float lox = xy.x * idirx - oodx;
            const float hix = xy.y * idirx - oodx;
            const float loy = xy.z * idiry - oody;
            const float hiy = xy.w * idiry - oody;
            const float loz = zi.x * idirz - oodz;
            const float hiz = zi.y * idirz - oodz;
            int link = __float_as_int(zi.z);


            const float cmin = spanBeginKepler(lox, hix, loy, hiy, loz, hiz, tmin);
            const bool hit = cmin <= spanEndKepler(lox, hix, loy, hiy, loz, hiz, hitT);


            float dist = hit ? cmin : HIP_INF_F;
            const int  hits = __popc(__ballot(hit)&subwarp_mask);
            if (!hits) continue;
            stackPtr += hits;


            //sort hits
            swap(dist, link, 0x01, bfe(threadIdx.x, 1) ^ bfe(threadIdx.x, 0));
            swap(dist, link, 0x02, bfe(threadIdx.x, 2) ^ bfe(threadIdx.x, 1));
            swap(dist, link, 0x01, bfe(threadIdx.x, 2) ^ bfe(threadIdx.x, 0));
            swap(dist, link, 0x04, bfe(threadIdx.x, 2));
            swap(dist, link, 0x02, bfe(threadIdx.x, 1));
            swap(dist, link, 0x01, bfe(threadIdx.x, 0));

            if (dist < HIP_INF_F)
            {
                traversalStack[subwarp][threadIdx.y][stackPtr - offset] = link;
#ifdef STACKED_T
                tStack[subwarp][threadIdx.y][stackPtr - offset] = dist;
#endif
            }
        }
        else {

            // Triangle bounds.
            const float4 triBounds = nodes[16 * (~curr) + SUBWARP_WIDTH];

            bool hit = false;
            for (int triAddr = __float_as_int(triBounds.z) + offset; triAddr < __float_as_int(triBounds.w); triAddr += SUBWARP_WIDTH)
            {
                ++testedTriangles;

                // Tris in TEX (good to fetch as a single batch)
                const float4 v00 = tris[3 * triAddr + 0];
                const float4 v11 = tris[3 * triAddr + 1];
                const float4 v22 = tris[3 * triAddr + 2];

                const float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                const float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                float t = Oz * invDz;

                if (t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

                    const float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                    const 	float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                    const 	float u = Ox + t*Dx;

                    if (u >= 0.0f)
                    {
                        // Compute and check barycentric v.

                        const float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                        const float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                        const float v = Oy + t*Dy;

                        if (v >= 0.0f && u + v <= 1.0f)
                        {
                            // Record intersection.
                            // Closest intersection not required => terminate.
                            hit = true;
                            hitT = t;
                            hitU = u;
                            hitV = v;
                            hitIndex = triAddr;
                        }
                    }
                }
            }

            // Sort triangles

            const int hits = __ballot(hit)&subwarp_mask;
            if (!hits) continue;

            hitT = hit ? hitT : HIP_INF_F;

#pragma unroll
            for (int mask = 4; mask > 0; mask >>= 1) {
                const float tmp_t = __shfl_xor(hitT, mask);
                const int tmp_addr = __shfl_xor(hitIndex, mask);
                const float tmp_u = __shfl_xor(hitU, mask);
                const float tmp_v = __shfl_xor(hitV, mask);
                hitIndex = tmp_t < hitT ? tmp_addr : hitIndex;
                hitU = tmp_t < hitT ? tmp_u : hitU;
                hitV = tmp_t < hitT ? tmp_v : hitV;
                hitT = fminf(hitT, tmp_t);
            }

            if (anyHit && hits)
            {
                stackPtr = -1;
                break;
            }

        }
    }

    if (offset == 0) {
        if (hitIndex == -1) {
            STORE_RESULT(rayidx, -1, hitT);
            STORE_STATS(rayidx, 0, 0);
        }
        else {
            STORE_RESULT_WITH_BAR_COORDS(rayidx, triIndices[hitIndex], hitT, hitU, hitV);
            STORE_STATS(rayidx, traversedNodes, testedTriangles);
        }
    }

} while (true);
}

//------------------------------------------------------------------------