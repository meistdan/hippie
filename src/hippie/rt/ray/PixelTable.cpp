/**
  * \file	PixelTable.cpp
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	PixelTable class source file.
  */

#include "PixelTable.h"

void PixelTable::recalculate() {

    // Construct LUTs.
    indexToPixel.resizeDiscard(size.x * size.y * sizeof(int));
    pixelToIndex.resizeDiscard(size.x * size.y * sizeof(int));
    int * idxtopos = (int*)indexToPixel.getMutablePtr();
    int * postoidx = (int*)pixelToIndex.getMutablePtr();

    // Smart mode.
    int idx = 0;
    int bheight = size.y & ~7;
    int bwidth = size.x  & ~7;

    // Bulk of the image, sort blocks in in morton order.
    int maxdim = (bwidth > bheight) ? bwidth : bheight;

    // Round up to nearest power of two.
    maxdim |= maxdim >> 1;
    maxdim |= maxdim >> 2;
    maxdim |= maxdim >> 4;
    maxdim |= maxdim >> 8;
    maxdim |= maxdim >> 16;
    maxdim = (maxdim + 1) >> 1;

    int width8 = bwidth >> 3;
    int height8 = bheight >> 3;
    for (int i = 0; i < maxdim * maxdim; i++) {

        // Get interleaved bit positions.
        int tx = 0;
        int ty = 0;
        int val = i;
        int bit = 1;

        while (val) {
            if (val & 1) tx |= bit;
            if (val & 2) ty |= bit;
            bit += bit;
            val >>= 2;
        }

        if (tx < width8 && ty < height8) {
            for (int inner = 0; inner < 64; inner++) {
                // Swizzle ix and iy within blocks as well.
                int ix = ((inner & 1) >> 0) | ((inner & 4) >> 1) | ((inner & 16) >> 2);
                int iy = ((inner & 2) >> 1) | ((inner & 8) >> 2) | ((inner & 32) >> 3);
                int pos = (ty * 8 + iy) * size.x + (tx * 8 + ix);
                postoidx[pos] = idx;
                idxtopos[idx++] = pos;
            }
        }

    }

    // If height not divisible, add horizontal stripe below bulk.
    for (int px = 0; px < bwidth; px++)
        for (int py = bheight; py < size.y; py++) {
            int pos = px + py * size.x;
            postoidx[pos] = idx;
            idxtopos[idx++] = pos;
        }

    // If width not divisible, add vertical stripe and the corner.
    for (int py = 0; py < size.y; py++)
        for (int px = bwidth; px < size.x; px++) {
            int pos = px + py * size.x;
            postoidx[pos] = idx;
            idxtopos[idx++] = pos;
        }

    // Done!
}

PixelTable::PixelTable() : size(0) {
}

PixelTable::~PixelTable() {
}

void PixelTable::setSize(const Vec2i & _size) {
    bool recalc = (_size != size);
    size = _size;
    if (recalc) recalculate();
}

const Vec2i & PixelTable::getSize() {
    return size;
}

Buffer & PixelTable::getIndexToPixel() {
    return indexToPixel;
}

Buffer & PixelTable::getPixelToIndex() {
    return pixelToIndex;
}
