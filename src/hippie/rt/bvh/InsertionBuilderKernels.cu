/**
 * \file	InsetionBuilderKernels.cu
 * \author	Daniel Meister
 * \date	2017/02/07
 * \brief	InsertionBuilder kernels soruce file.
 */

#include "rt/bvh/InsertionBuilderKernels.h"
#include "rt/bvh/HipBVHUtil.h"

extern "C" GLOBAL void findBestNode(
    const int numberOfNodes,
    const int numberOfReferences,
    const int mod,
    const int remainder,
    int * nodeParentIndices,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    int * outNodeIndices,
    float * areaReductions,
    Vec4f * nodeBoxesMin,
    Vec4f * nodeBoxesMax
) {

    // Thread index.
    const int inNodeIndex = mod * (blockDim.x * blockIdx.x + threadIdx.x) + remainder;

    // Best node found so far.
    float bestAreaReduction, areaReduction;
    int bestOutNodeIndex;

    // Node state.
    float inNodeParentArea, directAreaReductionBound;
    int inNodeParentIndex, inNodeSiblingIndex, outNodeIndex, highestIndex;
    AABB inNodeBox, outNodeBox, mergedBox, highestBox;

    // Down flag.
    bool down;

    if (inNodeIndex > 0 && inNodeIndex < numberOfNodes) {

        // Node parent index.
        inNodeParentIndex = nodeParentIndices[inNodeIndex];

        // Node boxes.
        inNodeBox = AABB(Vec3f(nodeBoxesMin[inNodeIndex]), Vec3f(nodeBoxesMax[inNodeIndex]));
        inNodeParentArea = AABB(Vec3f(nodeBoxesMin[inNodeParentIndex]), Vec3f(nodeBoxesMax[inNodeParentIndex])).area();
        directAreaReductionBound = inNodeParentArea - inNodeBox.area();
        highestBox.reset();

        // Sibling index.
        outNodeIndex = inNodeIndex;
        if (nodeLeftIndices[inNodeParentIndex] == outNodeIndex)
            inNodeSiblingIndex = nodeRightIndices[inNodeParentIndex];
        else
            inNodeSiblingIndex = nodeLeftIndices[inNodeParentIndex];

        // Switch to sibling and go down.
        outNodeIndex = inNodeSiblingIndex;
        highestIndex = inNodeParentIndex;
        down = true;

        // Best node found so far.
        bestOutNodeIndex = -1;
        bestAreaReduction = 0.0f;
        areaReduction = 0.0f;

        // Main loop.
        while (true) {

            // Bounding boxes.
            outNodeBox = AABB(Vec3f(nodeBoxesMin[outNodeIndex]), Vec3f(nodeBoxesMax[outNodeIndex]));
            mergedBox = AABB(inNodeBox, outNodeBox);

            // Down.
            if (down) {

                // Check area reduction, skip the node's sibling (original position).
                if (outNodeIndex != inNodeSiblingIndex) {

                    // Check area reduction.
                    float directAreaReduction = inNodeParentArea - mergedBox.area();
                    if (bestAreaReduction < directAreaReduction + areaReduction) {
                        bestAreaReduction = directAreaReduction + areaReduction;
                        bestOutNodeIndex = outNodeIndex;
                    }
                }   

                // Add area reduction.
                float areaReductionCur = outNodeBox.area() - mergedBox.area();
                areaReduction += areaReductionCur;
                
                // Leaf or pruning => Go up.
                if (outNodeIndex >= numberOfReferences - 1 || areaReduction + directAreaReductionBound <= bestAreaReduction) {
                    down = false;
                }
                // Interior => Go to the left child.
                else {
                    outNodeIndex = nodeLeftIndices[outNodeIndex];
                }
            }

            // Up.
            else {

                // Parent index.
                int outNodeParentIndex = nodeParentIndices[outNodeIndex];

                // Subtract node's area.
                float areaReductionCur = outNodeBox.area() - mergedBox.area();
                areaReduction -= areaReductionCur;

                // Back to the highest node.
                if (outNodeParentIndex == highestIndex) {

                    // Update cumulative box.
                    highestBox.grow(outNodeBox);

                    // Go back to the highest node.
                    outNodeIndex = outNodeParentIndex;

                    // Check area reduction, skip the node's parent.
                    if (outNodeIndex != inNodeParentIndex && outNodeIndex != 0 ) {

                        mergedBox = AABB(inNodeBox, highestBox);
                        float directAreaReduction = inNodeParentArea - mergedBox.area();
                        if (bestAreaReduction < directAreaReduction + areaReduction) {
                            bestAreaReduction = directAreaReduction + areaReduction;
                            bestOutNodeIndex = outNodeIndex;
                        }

                        // Add area reduction.
                        outNodeBox = AABB(Vec3f(nodeBoxesMin[outNodeIndex]), Vec3f(nodeBoxesMax[outNodeIndex]));
                        float areaReductionCur = outNodeBox.area() - highestBox.area();
                        areaReduction += areaReductionCur;

                    }

                    // The highest node is root => Done.
                    outNodeParentIndex = nodeParentIndices[outNodeIndex];
                    if (outNodeParentIndex < 0) {
                        break;
                    }

                    // Update the highest node.
                    highestIndex = outNodeParentIndex;

                    // Go down.
                    down = true;

                    // Switch to sibling.
                    if (nodeLeftIndices[highestIndex] == outNodeIndex) {
                        outNodeIndex = nodeRightIndices[highestIndex];
                    }
                    else {
                        outNodeIndex = nodeLeftIndices[highestIndex];
                    }

                }

                // Still bellow the highest node.
                else {

                    // Switch to right sibling.
                    if (nodeLeftIndices[outNodeParentIndex] == outNodeIndex) {
                        down = true;
                        outNodeIndex = nodeRightIndices[outNodeParentIndex];
                    }

                    // Go up.
                    else {
                        outNodeIndex = outNodeParentIndex;
                    }

                }

            }

        }

        // Save the best out node and area reduction.
        areaReductions[inNodeIndex] = bestAreaReduction;
        outNodeIndices[inNodeIndex] = bestOutNodeIndex;
    }

}

extern "C" GLOBAL void lockNodes(
    const int numberOfNodes,
    const int mod,
    const int remainder,
    int * nodeParentIndices,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    int * outNodeIndices,
    float * areaReductions,
    unsigned long long * locks
) {

    // Thread index.
    const int inNodeIndex = mod * (blockDim.x * blockIdx.x + threadIdx.x) + remainder;

    if (inNodeIndex < numberOfNodes && inNodeIndex > 0) {

        // Best out node index and area reduction.
        int outNodeIndex = outNodeIndices[inNodeIndex];
        float areaReduction = areaReductions[inNodeIndex];

        // Area reduction and out node index.
        unsigned long long lock = ((unsigned long long)(__float_as_int(areaReduction)) << 32ull) | (unsigned long long)(inNodeIndex);

        // Only successfully found positions.
        if (outNodeIndex >= 0) {

            // In node parent index.
            int inNodeParentIndex = nodeParentIndices[inNodeIndex];

            // In node sibling index.
            int inNodeSiblingIndex;

            // Came from left.
            if (inNodeIndex == nodeLeftIndices[inNodeParentIndex]) {
                inNodeSiblingIndex = nodeRightIndices[inNodeParentIndex];
            }
            // Came from right.
            else {
                inNodeSiblingIndex = nodeLeftIndices[inNodeParentIndex];
            }

            // Lock in node, its parent, and sibling.
            atomicMax(&locks[inNodeIndex], lock);
            atomicMax(&locks[inNodeSiblingIndex], lock);
            atomicMax(&locks[inNodeParentIndex], lock);

            // Parent is not the root => lock parent's parent.
            if (inNodeParentIndex > 0) {
                int inNodeParentParentIndex = nodeParentIndices[inNodeParentIndex];
                atomicMax(&locks[inNodeParentParentIndex], lock);
            }

            // Parent is root => lock sibling's children.
            else if (inNodeSiblingIndex < numberOfNodes - 1) {
                int inNodeSiblingLeftIndex = nodeLeftIndices[inNodeSiblingIndex];
                int inNodeSiblingRightIndex = nodeRightIndices[inNodeSiblingIndex];
                atomicMax(&locks[inNodeSiblingLeftIndex], lock);
                atomicMax(&locks[inNodeSiblingRightIndex], lock);
            }

            // Lock out node.
            atomicMax(&locks[outNodeIndex], lock);

            // Parent is not the root => lock parent's parent.
            if (outNodeIndex > 0) {
                int outNodeParentIndex = nodeParentIndices[outNodeIndex];
                atomicMax(&locks[outNodeParentIndex], lock);
            }
            // Parent is root => lock root's children.
            else {
                int outNodeLeftIndex = nodeLeftIndices[outNodeIndex];
                int outNodeRightIndex = nodeRightIndices[outNodeIndex];
                atomicMax(&locks[outNodeLeftIndex], lock);
                atomicMax(&locks[outNodeRightIndex], lock);
            }

        }

    }

}

extern "C" GLOBAL void checkLocks(
    const int numberOfNodes,
    const int mod,
    const int remainder,
    int * nodeParentIndices,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    int * outNodeIndices,
    float * areaReductions,
    unsigned long long * locks
) {

    // Thread index.
    const int inNodeIndex = mod * (blockDim.x * blockIdx.x + threadIdx.x) + remainder;

    if (inNodeIndex < numberOfNodes && inNodeIndex > 0) {

        // Best out node index and area reduction.
        float areaReduction = areaReductions[inNodeIndex];
        int bestOutNodeIndex = outNodeIndices[inNodeIndex];
        int outNodeIndex = bestOutNodeIndex;

        // Area reduction and out node index.
        unsigned long long lock = ((unsigned long long)(__float_as_int(areaReduction)) << 32) | (unsigned long long)(inNodeIndex);

        // Only nodes with positive area reductions.
        if (bestOutNodeIndex >= 0) {

            // Increment inserted nodes.
            atomicAdd(&foundNodes, 1);

            // In node parent index.
            int inNodeParentIndex = nodeParentIndices[inNodeIndex];

            // In node sibling index and state.
            int inNodeSiblingIndex;

            // Came from left.
            if (inNodeIndex == nodeLeftIndices[inNodeParentIndex]) {
                inNodeSiblingIndex = nodeRightIndices[inNodeParentIndex];
            }
            // Came from right.
            else {
                inNodeSiblingIndex = nodeLeftIndices[inNodeParentIndex];
            }

            // Check in node, its parent, and sibling.
            if (locks[inNodeIndex] != lock) bestOutNodeIndex = -1;
            if (locks[inNodeSiblingIndex] != lock) bestOutNodeIndex = -1;
            if (locks[inNodeParentIndex] != lock) bestOutNodeIndex = -1;

            // Parent is not the root => check parent's parent.
            if (inNodeParentIndex > 0) {
                int inNodeParentParentIndex = nodeParentIndices[inNodeParentIndex];
                if (locks[inNodeParentParentIndex] != lock) bestOutNodeIndex = -1;
            }
            // Parent is root => check sibling's children.
            else if (inNodeSiblingIndex < numberOfNodes - 1) {
                int inNodeSiblingLeftIndex = nodeLeftIndices[inNodeSiblingIndex];
                int inNodeSiblingRightIndex = nodeRightIndices[inNodeSiblingIndex];
                if (locks[inNodeSiblingLeftIndex] != lock) bestOutNodeIndex = -1;
                if (locks[inNodeSiblingRightIndex] != lock) bestOutNodeIndex = -1;
            }

            // Check out node.
            if (locks[outNodeIndex] != lock) bestOutNodeIndex = -1;

            // Parent is not the root => check parent's parent.
            if (outNodeIndex > 0) {
                int outNodeParentIndex = nodeParentIndices[outNodeIndex];
                if (locks[outNodeParentIndex] != lock) bestOutNodeIndex = -1;
            }

            // Out node index.
            outNodeIndices[inNodeIndex] = bestOutNodeIndex;

        }

    }

}

extern "C" GLOBAL void reinsert(
    const int numberOfNodes,
    const int mod,
    const int remainder,
    int * nodeParentIndices,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    int * outNodeIndices,
    float * areaReductions
) {

    // Thread index.
    const int inNodeIndex = mod * (blockDim.x * blockIdx.x + threadIdx.x) + remainder;

    if (inNodeIndex < numberOfNodes && inNodeIndex > 0) {

        // Best out node index.
        int outNodeIndex = outNodeIndices[inNodeIndex];

        // Only nodes with positive area reductions.
        if (outNodeIndex >= 0) {

            // In node parent index.
            int inNodeParentIndex = nodeParentIndices[inNodeIndex];

            // In node sibling index.
            int inNodeSiblingIndex;

            // Came from left.
            if (inNodeIndex == nodeLeftIndices[inNodeParentIndex]) {
                inNodeSiblingIndex = nodeRightIndices[inNodeParentIndex];
            }
            // Came from right.
            else {
                inNodeSiblingIndex = nodeLeftIndices[inNodeParentIndex];
            }

            // Remove.
            if (inNodeParentIndex != 0) {
                int inNodeParentParentIndex = nodeParentIndices[inNodeParentIndex];
                if (nodeLeftIndices[inNodeParentParentIndex] == inNodeParentIndex)
                    nodeLeftIndices[inNodeParentParentIndex] = inNodeSiblingIndex;
                else
                    nodeRightIndices[inNodeParentParentIndex] = inNodeSiblingIndex;
                nodeParentIndices[inNodeSiblingIndex] = inNodeParentParentIndex;
            }
            else {
                int inNodeSiblingLeftIndex = nodeLeftIndices[inNodeSiblingIndex];
                int inNodeSiblingRightIndex = nodeRightIndices[inNodeSiblingIndex];
                nodeLeftIndices[0] = inNodeSiblingLeftIndex;
                nodeRightIndices[0] = inNodeSiblingRightIndex;
                nodeParentIndices[inNodeSiblingLeftIndex] = 0;
                nodeParentIndices[inNodeSiblingRightIndex] = 0;
                inNodeParentIndex = inNodeSiblingIndex;
            }

            // Insert.
            if (outNodeIndex != 0) {
                int outNodeParentIndex = nodeParentIndices[outNodeIndex];
                if (nodeLeftIndices[outNodeParentIndex] == outNodeIndex)
                    nodeLeftIndices[outNodeParentIndex] = inNodeParentIndex;
                else
                    nodeRightIndices[outNodeParentIndex] = inNodeParentIndex;
                nodeParentIndices[inNodeParentIndex] = outNodeParentIndex;
                nodeLeftIndices[inNodeParentIndex] = outNodeIndex;
                nodeRightIndices[inNodeParentIndex] = inNodeIndex;
                nodeParentIndices[outNodeIndex] = inNodeParentIndex;
                nodeParentIndices[inNodeIndex] = inNodeParentIndex;
            }
            else {
                int outNodeLeftIndex = nodeLeftIndices[0];
                int outNodeRightIndex = nodeRightIndices[0];
                nodeLeftIndices[inNodeParentIndex] = outNodeLeftIndex;
                nodeRightIndices[inNodeParentIndex] = outNodeRightIndex;
                nodeParentIndices[outNodeLeftIndex] = inNodeParentIndex;
                nodeParentIndices[outNodeRightIndex] = inNodeParentIndex;
                nodeParentIndices[inNodeParentIndex] = 0;
                nodeParentIndices[inNodeIndex] = 0;
                nodeLeftIndices[0] = inNodeIndex;
                nodeRightIndices[0] = inNodeParentIndex;
            }

            // Increment inserted nodes.
            atomicAdd(&insertedNodes, 1);

        }

    }

}

extern "C" GLOBAL void computeCost(
    const int numberOfNodes,
    const int numberOfReferences,
    const float sceneBoxArea,
    const float ct,
    const float ci,
    Vec4f * nodeBoxesMin,
    Vec4f * nodeBoxesMax
) {

    // Thread index.
    const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Cost.
    float _cost = 0.0f;

    if (nodeIndex < numberOfNodes) {

        // Node box and area.
        AABB nodeBox = AABB(Vec3f(nodeBoxesMin[nodeIndex]), Vec3f(nodeBoxesMax[nodeIndex]));
        float P = nodeBox.area() / sceneBoxArea;

        // Leaf.
        if (nodeIndex >= numberOfReferences - 1) {
            _cost += ci * P;
        }

        // Interior node.
        else {
            _cost += ct * P;
        }

    }

    // Cache.
    __shared__ volatile float cache[REDUCTION_BLOCK_THREADS];

    // Cost reduction.
    cache[threadIdx.x] = _cost;
    cache[threadIdx.x] += cache[threadIdx.x ^ 1];
    cache[threadIdx.x] += cache[threadIdx.x ^ 2];
    cache[threadIdx.x] += cache[threadIdx.x ^ 4];
    cache[threadIdx.x] += cache[threadIdx.x ^ 8];
    cache[threadIdx.x] += cache[threadIdx.x ^ 16];

    __syncthreads();
    if ((threadIdx.x & 32) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 32];

    __syncthreads();
    if ((threadIdx.x & 64) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 64];

    __syncthreads();
    if ((threadIdx.x & 128) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 128];

    // Update total cost.
    if (threadIdx.x == 0) {
        atomicAdd(&cost, cache[threadIdx.x]);
    }

}
