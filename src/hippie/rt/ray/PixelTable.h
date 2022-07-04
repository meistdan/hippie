/**
  * \file	PixelTable.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	PixelTable class header file.
  */

#ifndef _PIXEL_TABLE_H_
#define _PIXEL_TABLE_H_

#include "gpu/Buffer.h"
#include "util/Math.h"

class PixelTable {

private:

    Vec2i size;
    Buffer indexToPixel; // int
    Buffer pixelToIndex; // int

    void recalculate(void);

public:

    PixelTable(void);
    ~PixelTable(void);

    void setSize(const Vec2i & size);

    const Vec2i & getSize(void);
    Buffer & getIndexToPixel(void);
    Buffer & getPixelToIndex(void);

};

#endif /* _PIXEL_TABLE_H_ */
