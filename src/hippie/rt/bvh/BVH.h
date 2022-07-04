/**
 * \file	BVH.h
 * \author	Daniel Meister
 * \date	2014/08/19
 * \brief	BVH class header file.
 */

#ifndef _BVH_H_
#define _BVH_H_

#include "HipBVH.h"
#include "environment/AppEnvironment.h"
#include "util/AABB.h"
#include "util/Logger.h"
#include <QVector>

class BVH {

public:

    const static int N = 8;

    struct Node {

        Node * parent;
        AABB box;
        bool leaf;
        unsigned char depth;
        int begin;
        int end;
        int id;

        float cost[N - 1];
        Vec2i count[N - 1];

        Node(void) : parent(nullptr) {}
        Node(bool leaf) : leaf(leaf), parent(nullptr) {}
        virtual ~Node(void) {};

        bool isLeaf(void) const { return leaf; }
        int getSize(void) { return end - begin; }

        virtual const Node * getChildNode(int index) const = 0;
        virtual const int getNumberOfChildNodes(void) const = 0;

    };

    struct InteriorNode : public Node {

        int numberOfChildren;
        Node * children[N];

        InteriorNode(void) : Node(false), numberOfChildren(2) { for (int i = 0; i < N; ++i) children[i] = nullptr; }
        virtual ~InteriorNode(void) { for (int i = 0; i < N; ++i) if (children[i]) delete children[i]; }

        virtual const Node * getChildNode(int index) const { Q_ASSERT(index >= 0 && index < getNumberOfChildNodes()); return children[index]; }
        virtual const int getNumberOfChildNodes(void) const { return numberOfChildren; }

    };

    struct LeafNode : public Node {

        LeafNode(void) : Node(true) {}
        virtual ~LeafNode(void) {}

        virtual const Node * getChildNode(int index) const { return nullptr; }
        virtual const int getNumberOfChildNodes(void) const { return 0; }

    };

private:

    Node * root;
    int numberOfLeafNodes;
    int numberOfInteriorNodes;
    QVector<int> triangleIndices;
    float ct;
    float ci;

    float getCost(const Node * node) const {
        if (node->isLeaf()) {
            return ci * (node->end - node->begin);
        }
        else {
            float cost = ct;
            const InteriorNode * interior = dynamic_cast<const InteriorNode*>(node);
            if (interior->box.area() > 0.0f)
                for (int i = 0; i < interior->getNumberOfChildNodes(); ++i)
                    cost += (interior->children[i]->box.area() / interior->box.area()) * getCost(interior->children[i]);
            return cost;
        }
    }

    void recomputeCostsWide(Node * node, int n, int maxLeafSize) const {
        if (node->isLeaf()) {
            for (int j = 1; j < n; ++j) {
                node->cost[j - 1] = ci * node->box.area() / root->box.area() * divCeil(node->getSize(), n) * n;
                node->count[j - 1] = Vec2i(-1);
            }
        }
        else {
            const InteriorNode * interior = dynamic_cast<const InteriorNode*>(node);
            for (int i = 0; i < node->getNumberOfChildNodes(); ++i)
                recomputeCostsWide(interior->children[i], n, maxLeafSize);
            float leafCost = (node->end - node->begin) > maxLeafSize ? MAX_FLOAT 
                : ci * node->box.area() / root->box.area() * divCeil(node->getSize(), n) * n;
            float subtreeCost = ct * node->box.area() / root->box.area();
            float cDist = MAX_FLOAT;
            Vec2i bestCount;
            for (int k = 1; k < n; ++k) {
                float c = node->getChildNode(0)->cost[k - 1] 
                        + node->getChildNode(1)->cost[n - k - 1];
                if (cDist > c) {
                    cDist = c;
                    bestCount = Vec2i(k, n - k);
                }
            }
            subtreeCost += cDist;
            if (leafCost > subtreeCost) {
                node->cost[0] = subtreeCost;
                node->count[0] = bestCount;
            }
            else {
                node->cost[0] = leafCost;
                node->count[0] = Vec2i(-1);
            }
            for (int j = 2; j < n; ++j) {
                cDist = MAX_FLOAT;
                for (int k = 1; k < j; ++k) {
                    float c = node->getChildNode(0)->cost[k - 1]
                            + node->getChildNode(1)->cost[j - k - 1];
                    if (cDist > c) {
                        cDist = c;
                        bestCount = Vec2i(k, j - k);
                    }
                }
                if (node->cost[j - 2] <= cDist) {
                    node->cost[j - 1] = node->cost[j - 2];
                    node->count[j - 1] = node->count[j - 2];
                }
                else {
                    node->cost[j - 1] = cDist;
                    node->count[j - 1] = bestCount;
                }
            }
        }
    }

    void recomputeCosts(Node* node, int maxLeafSize) const {
        if (node->isLeaf()) {
            node->cost[0] = ci * node->box.area() / root->box.area() * node->getSize();
        }
        else {
            const InteriorNode* interior = dynamic_cast<const InteriorNode*>(node);
            for (int i = 0; i < node->getNumberOfChildNodes(); ++i)
                recomputeCosts(interior->children[i], maxLeafSize);
            float leafCost = (node->end - node->begin) > maxLeafSize ? MAX_FLOAT
                : ci * node->box.area() / root->box.area() * (node->end - node->begin);
            float subtreeCost = ct * node->box.area() / root->box.area();
            for (int i = 0; i < node->getNumberOfChildNodes(); ++i)
                subtreeCost += interior->children[i]->cost[0];
            if (leafCost > subtreeCost) {
                node->cost[0] = subtreeCost;
                node->count[0] = Vec2i(1);
            }
            else {
                node->cost[0] = leafCost;
                node->count[0] = Vec2i(-1);
            }
        }
    }

    void recomputeDepth(Node * node, int depth) {
        node->depth = depth;
        if (!node->isLeaf()) {
            const InteriorNode * interior = dynamic_cast<InteriorNode*>(node);
            for (int i = 0; i < interior->getNumberOfChildNodes(); ++i) {
                recomputeDepth(interior->children[i], depth + 1);
            }
        }
    }

    void recomputeBounds(Node * node) {
        if (!node->isLeaf()) {
            InteriorNode * interior = dynamic_cast<InteriorNode*>(node);
            node->begin = MAX_INT;
            node->end = 0;
            for (int i = 0; i < interior->getNumberOfChildNodes(); ++i) {
                recomputeBounds(interior->children[i]);
                node->begin = qMin(node->begin, interior->children[i]->begin);
                node->end = qMax(node->end, interior->children[i]->end);
            }
        }
    }

    void recomputeNumberOfNodes(Node * node) {
        if (node->isLeaf()) {
            ++numberOfLeafNodes;
        }
        else {
            ++numberOfInteriorNodes;
            const InteriorNode * interior = dynamic_cast<InteriorNode*>(node);
            for (int i = 0; i < interior->getNumberOfChildNodes(); ++i) {
                recomputeNumberOfNodes(interior->children[i]);
            }
        }
    }

    void recomputeParents(Node * node) {
        if (!node->isLeaf()) {
            InteriorNode * interior = dynamic_cast<InteriorNode*>(node);
            for (int i = 0; i < interior->getNumberOfChildNodes(); ++i) {
                interior->children[i]->parent = node;
                recomputeParents(interior->children[i]);
            }
        }
    }

    void recomputeIDs(Node * node, int & id) {
        node->id = id++;
        if (!node->isLeaf()) {
            InteriorNode * interior = dynamic_cast<InteriorNode*>(node);
            for (int i = 0; i < interior->getNumberOfChildNodes(); ++i)
                recomputeIDs(interior->children[i], id);
        }
    }

    void collapseWide(Node * node, int k, int n) {
        if (!node->isLeaf()) {

            InteriorNode * interior = dynamic_cast<InteriorNode*>(node);
            int count[N];
            count[0] = node->count[k - 1].x;
            count[1] = node->count[k - 1].y;
            while (true) {
                
                if (node->getNumberOfChildNodes() == n)
                    break;

                int s = -1;
                for (int i = 0; i < node->getNumberOfChildNodes(); ++i) {
                    int leftChildCount = interior->children[i]->count[count[i] - 1].x;
                    int rightChildCount = interior->children[i]->count[count[i] - 1].y;
                    if (leftChildCount > 0 && rightChildCount > 0 && leftChildCount + rightChildCount < n) {
                        s = i;
                        break;
                    }
                }
                if (s == -1)
                    break;

                for (int i = node->getNumberOfChildNodes(); i > s + 1; --i) {
                    interior->children[i] = interior->children[i - 1];
                    count[i] = count[i - 1];
                }

                InteriorNode * sc = dynamic_cast<InteriorNode*>(interior->children[s]);
                interior->children[s] = sc->children[0];
                interior->children[s + 1] = sc->children[1];

                int c = count[s];
                count[s] = sc->count[c - 1].x;
                count[s + 1] = sc->count[c - 1].y;
                ++interior->numberOfChildren;

                for (int i = 0; i < N; ++i)
                    sc->children[i] = nullptr;
                delete sc;

            }

            
            for (int i = 0; i < interior->getNumberOfChildNodes(); ++i) {
                Vec2i childCount = interior->children[i]->count[count[i] - 1];
                if (childCount.x == -1 && childCount.y == -1) {
                    LeafNode * leaf = new LeafNode();
                    leaf->begin = interior->children[i]->begin;
                    leaf->end = interior->children[i]->end;
                    leaf->box = interior->children[i]->box;
                    leaf->depth = interior->children[i]->depth;
                    delete interior->children[i];
                    interior->children[i] = leaf;
                }
            }

            for (int i = 0; i < interior->getNumberOfChildNodes(); ++i)
                collapseWide(interior->children[i], count[i], n);
            
        }
        
    }

    // Simple collapse.
    void collapse(Node* node, int maxLeafSize) {
        if (!node->isLeaf()) {
            InteriorNode * interior = dynamic_cast<InteriorNode*>(node);
            for (int i = 0; i < node->getNumberOfChildNodes(); ++i) {
                collapse(interior->children[i], maxLeafSize);
                Vec2i childCount = interior->children[i]->count[0];
                if (childCount.x == -1 && childCount.y == -1) {
                    LeafNode* leaf = new LeafNode();
                    leaf->begin = interior->children[i]->begin;
                    leaf->end = interior->children[i]->end;
                    leaf->box = interior->children[i]->box;
                    leaf->depth = interior->children[i]->depth;
                    delete interior->children[i];
                    interior->children[i] = leaf;
                }
            }
        }
    }

    void normalize(Node * node, const Vec3f & offset, float scale) {
        node->box.mn -= offset;
        node->box.mn *= scale;
        node->box.mx -= offset;
        node->box.mx *= scale;
        if (!node->isLeaf()) {
            InteriorNode * interior = dynamic_cast<InteriorNode*>(node);
            for (int i = 0; i < interior->getNumberOfChildNodes(); ++i)
                normalize(interior->children[i], offset, scale);
        }
    }

public:

    BVH(void) : root(nullptr), numberOfLeafNodes(0), numberOfInteriorNodes(0), ci(CI), ct(CT) {
        float _ct, _ci;
        Environment::getInstance()->getFloatValue("Bvh.ct", _ct);
        Environment::getInstance()->getFloatValue("Bvh.ci", _ci);
        if (_ct > 0.0f) ct = _ct;
        if (_ci > 0.0f) ci = _ci;
    }

    ~BVH(void) { if (root) delete root; }

    const QVector<int> & getTriangleIndices() const {
        return triangleIndices;
    }

    const Node * getRoot(void) const {
        return root;
    }

    int getNumberOfLeafNodes(void) const {
        return numberOfLeafNodes;
    }

    int getNumberOfInteriorNodes(void) const {
        return numberOfInteriorNodes;
    }

    int getNumberOfNodes(void) const {
        return getNumberOfLeafNodes() + getNumberOfInteriorNodes();
    }

    float getCi(void) {
        return ci;
    }

    float getCt(void) {
        return ct;
    }

    float getCost(void) {
        return getCost(root);
    }

    void recomputeCostsWide(int n, int maxLeafSize) {
        recomputeCostsWide(root, n, maxLeafSize);
    }

    void recomputeCosts(int maxLeafSize) {
        recomputeCosts(root, maxLeafSize);
    }

    void recomputeDepth(void) {
        recomputeDepth(root, 0);
    }

    void recomputeBounds(void) {
        recomputeBounds(root);
    }

    void recomputeNumberOfNodes(void) {
        numberOfInteriorNodes = 0;
        numberOfLeafNodes = 0;
        recomputeNumberOfNodes(root);
    }

    void recomputeParents(void) {
        recomputeParents(root);
        root->parent = nullptr;
    }

    void recomputeIDs(void) {
        int id = 0;
        recomputeIDs(root, id);
    }

    void collapseAdaptiveWide(int n) {
#if 1
        recomputeIDs();
        recomputeCostsWide(n, MAX_INT);
        collapseWide(root, 1, n);
#else
        recomputeCostsWide(n, 1);
        collapseWide(root, 1, n);
        recomputeCosts(MAX_INT);
        collapse(root, MAX_INT);
#endif
        recomputeParents();
        recomputeDepth();
        recomputeNumberOfNodes();
    }

    void collapseWide(int n, int maxLeafSize) {
        recomputeCostsWide(n, maxLeafSize);
        collapseWide(root, 1, n);
        recomputeParents();
        recomputeDepth();
        recomputeNumberOfNodes();
    }

    void collapseAdaptive(void) {
        recomputeCosts(MAX_INT);
        collapse(root, MAX_INT);
        recomputeParents();
        recomputeDepth();
        recomputeNumberOfNodes();
    }

    void collapse(int maxLeafSize) {
        recomputeCosts(maxLeafSize);
        collapse(root, maxLeafSize);
        recomputeParents();
        recomputeDepth();
        recomputeNumberOfNodes();
    }

    void normalize(void) {
        Vec3f diag = root->box.diagonal();
        Vec3f offset = root->box.mn;
        float longest = qMax(diag.x, qMax(diag.y, diag.z));
        float scale = 1.0f / longest;
        normalize(root, offset, scale);
    }

    friend class BVHCollapser;
    friend class BVHImporter;
    friend class DPBuilder;
    friend class InsertionOptimizer;
    friend class SBVHBuilder;

};

#endif /* _BVH_H_ */
