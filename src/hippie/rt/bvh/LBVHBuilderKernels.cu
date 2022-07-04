/**
 * \file	LBVHBuilderKernels.cu
 * \author	Daniel Meister
 * \date	2015/11/27
 * \brief	LBVHBuilder kernels soruce file.
 */

#include "rt/bvh/LBVHBuilderKernels.h"
#include "rt/bvh/HipBVHUtil.h"

template <typename T>
DEVICE void construct(
    const int n,
    const int i,
    int * nodeParentIndices,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    T * mortonCodes
) {

    // Determine direction of the range (+1 or -1).
    const int d = sgn(delta(i, i + 1, n, mortonCodes) - delta(i, i - 1, n, mortonCodes));

    // Compute upper bound for the length of the range.
    const int deltaMin = delta(i, i - d, n, mortonCodes);
    int lmax = 2;
    while (delta(i, i + lmax * d, n, mortonCodes) > deltaMin) lmax <<= 1;

    // Find the other end using binary search.
    int l = 0;
    for (int t = lmax >> 1; t >= 1; t >>= 1)
        if (delta(i, i + (l + t) * d, n, mortonCodes) > deltaMin)
            l += t;
    const int j = i + l * d;

    // Find the split position using binary search.
    const int deltaNode = delta(i, j, n, mortonCodes);
    int s = 0;
    int k = 2;
    int t;
    do {
        t = divCeil(l, k);
        k <<= 1;
        if (delta(i, i + (s + t) * d, n, mortonCodes) > deltaNode)
            s += t;
    } while (t > 1);
    const int gamma = i + s * d + min(d, 0);

    // Output child pointers.
    int left = gamma;
    int right = gamma + 1;
    if (min(i, j) == gamma) left += n - 1;
    if (max(i, j) == gamma + 1) right += n - 1;

    // Write node etc.
    nodeLeftIndices[i] = left;
    nodeRightIndices[i] = right;

    // Parent indices.
    nodeParentIndices[left] = i;
    nodeParentIndices[right] = i;

}

extern "C" GLOBAL void setupBoxes(
    const int numberOfTriangles,
    int * triangleIndices,
    Vec3i * triangles,
    Vec3f * vertices,
    Vec4f * referenceBoxesMin,
    Vec4f * referenceBoxesMax
) {

    // Triangle index.
    const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (triangleIndex < numberOfTriangles) {

        // Triangle.		
        Vec3i triangle = triangles[triangleIndex];
        Vec3f v0 = vertices[triangle.x];
        Vec3f v1 = vertices[triangle.y];
        Vec3f v2 = vertices[triangle.z];

        // Box.
        AABB box;
        box.grow(v0);
        box.grow(v1);
        box.grow(v2);

        // Assign index and box.
        triangleIndices[triangleIndex] = triangleIndex;
        referenceBoxesMin[triangleIndex] = Vec4f(box.mn, 0.0f);
        referenceBoxesMax[triangleIndex] = Vec4f(box.mx, 0.0f);

    }

}

extern "C" GLOBAL void computeMortonCodes30(
    const int numberOfReferences,
    const int mortonCodeBits,
    unsigned int * mortonCodes,
    int * referenceIndices,
    Vec4f * referenceBoxesMin,
    Vec4f * referenceBoxesMax
) {

    // Reference index.
    const int referenceIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Scene box.
    AABB _sceneBox = *(AABB*)sceneBox;
    Vec3f scale = 1.0f / _sceneBox.diagonal();

    if (referenceIndex < numberOfReferences) {

        // Box.
        AABB box;
        box.grow(Vec3f(referenceBoxesMin[referenceIndex]));
        box.grow(Vec3f(referenceBoxesMax[referenceIndex]));

        // Assign index and Morton code.
        mortonCodes[referenceIndex] = mortonCode((box.centroid() - _sceneBox.mn) * scale) >> (30 - mortonCodeBits);
        referenceIndices[referenceIndex] = referenceIndex;

    }

}

extern "C" GLOBAL void computeMortonCodes60(
    const int numberOfReferences,
    const int mortonCodeBits,
    unsigned long long * mortonCodes,
    int * referenceIndices,
    Vec4f * referenceBoxesMin,
    Vec4f * referenceBoxesMax
) {

    // Reference index.
    const int referenceIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Scene box.
    AABB _sceneBox = *(AABB*)sceneBox;
    Vec3f scale = 1.0f / _sceneBox.diagonal();

    if (referenceIndex < numberOfReferences) {

        // Box.
        AABB box;
        box.grow(Vec3f(referenceBoxesMin[referenceIndex]));
        box.grow(Vec3f(referenceBoxesMax[referenceIndex]));

        // Assign index and Morton code.
        mortonCodes[referenceIndex] = mortonCode64((box.centroid() - _sceneBox.mn) * scale) >> (60 - mortonCodeBits);
        referenceIndices[referenceIndex] = referenceIndex;

    }

}

extern "C" GLOBAL  void setupLeaves(
    const int numberOfReferences,
    int * referenceIndices0,
    int * referenceIndices1,
    int * triangleIndices0,
    int * triangleIndices1,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    Vec4f * nodeBoxesMin,
    Vec4f * nodeBoxesMax,
    Vec4f * referenceBoxesMin,
    Vec4f * referenceBoxesMax
) {

    // Box index.
    const int boxIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (boxIndex < numberOfReferences) {

        // Reference index.
        const int referenceIndex = referenceIndices0[boxIndex];

        // Leaf node.
        const int nodeIndex = boxIndex + numberOfReferences - 1;
        nodeLeftIndices[nodeIndex] = boxIndex;
        nodeRightIndices[nodeIndex] = boxIndex + 1;
        nodeBoxesMin[nodeIndex] = referenceBoxesMin[referenceIndex];
        nodeBoxesMax[nodeIndex] = referenceBoxesMax[referenceIndex];
        triangleIndices1[boxIndex] = triangleIndices0[referenceIndex];
        referenceIndices1[boxIndex] = nodeIndex;

    }

}

extern "C" GLOBAL void construct30(
    const int n,
    int * nodeParentIndices,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    unsigned int * mortonCodes
) {

    // Thread index.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n - 1) {
        construct(
            n,
            i,
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            mortonCodes
        );
    }

    // Root parent index.
    if (i == 0)
        nodeParentIndices[0] = -1;


}

extern "C" GLOBAL void construct60(
    const int n,
    int * nodeParentIndices,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    unsigned long long * mortonCodes
) {

    // Thread index.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n - 1) {
        construct(
            n,
            i,
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            mortonCodes
        );
    }

    // Root parent index.
    if (i == 0)
        nodeParentIndices[0] = -1;

}

extern "C" GLOBAL void refit(
    const int numberOfNodes,
    int * termCounters,
    int * nodeParentIndices,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    Vec4f * nodeBoxesMin,
    Vec4f * nodeBoxesMax
) {

    // Leaf index.
    const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x + (numberOfNodes >> 1);

    if (leafIndex < numberOfNodes) {

        // Node index.
        int nodeIndex = nodeParentIndices[leafIndex];

        // Go up to the root.
        while (nodeIndex >= 0 && atomicAdd(&termCounters[nodeIndex], 1) > 0) {

            // Sync. global memory writes.
            __threadfence();

            // Node.
            int nodeLeftIndex = nodeLeftIndices[nodeIndex];
            int nodeRightIndex = nodeRightIndices[nodeIndex];

            // Box.
            AABB box;

            // Min.
            box.grow(Vec3f(nodeBoxesMin[nodeLeftIndex]));
            box.grow(Vec3f(nodeBoxesMin[nodeRightIndex]));
            nodeBoxesMin[nodeIndex] = Vec4f(box.mn, 0.0f);

            // Max.
            box.grow(Vec3f(nodeBoxesMax[nodeLeftIndex]));
            box.grow(Vec3f(nodeBoxesMax[nodeRightIndex]));
            nodeBoxesMax[nodeIndex] = Vec4f(box.mx, 0.0f);

            // Go to the parent.
            nodeIndex = nodeParentIndices[nodeIndex];

        }

    }

}
