/**
  * \file	Hash.cpp
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	A source file containing some useful hash functions.
  */

#include "Hash.h"

unsigned int hashString(const QString & str) {
    return hashBuffer(str.toUtf8().constData(), str.length());
}

unsigned int hashBuffer(const void * ptr, int size) {

    Q_ASSERT(size >= 0);
    Q_ASSERT(ptr || !size);

    if ((((int)(unsigned long long)ptr | size) & 3) == 0)
        return hashBufferAlign(ptr, size);

    const unsigned char * src = (const unsigned char*)ptr;
    unsigned int a = HASH_MAGIC;
    unsigned int b = HASH_MAGIC;
    unsigned int c = HASH_MAGIC;

    while (size >= 12) {
        a += src[0] + (src[1] << 8) + (src[2] << 16) + (src[3] << 24);
        b += src[4] + (src[5] << 8) + (src[6] << 16) + (src[7] << 24);
        c += src[8] + (src[9] << 8) + (src[10] << 16) + (src[11] << 24);
        JENKINS_MIX(a, b, c);
        src += 12;
        size -= 12;
    }

    switch (size) {
    case 11: c += src[10] << 16;
    case 10: c += src[9] << 8;
    case 9:  c += src[8];
    case 8:  b += src[7] << 24;
    case 7:  b += src[6] << 16;
    case 6:  b += src[5] << 8;
    case 5:  b += src[4];
    case 4:  a += src[3] << 24;
    case 3:  a += src[2] << 16;
    case 2:  a += src[1] << 8;
    case 1:  a += src[0];
    case 0:  break;
    }

    c += size;
    JENKINS_MIX(a, b, c);

    return c;
}

unsigned int hashBufferAlign(const void * ptr, int size) {

    Q_ASSERT(size >= 0);
    Q_ASSERT(ptr || !size);
    Q_ASSERT(((unsigned long long)ptr & 3) == 0);
    Q_ASSERT((size & 3) == 0);

    const unsigned int * src = (const unsigned int*)ptr;
    unsigned int a = HASH_MAGIC;
    unsigned int b = HASH_MAGIC;
    unsigned int c = HASH_MAGIC;

    while (size >= 12) {
        a += src[0];
        b += src[1];
        c += src[2];
        JENKINS_MIX(a, b, c);
        src += 3;
        size -= 12;
    }

    switch (size) {
    case 8: b += src[1];
    case 4: a += src[0];
    case 0: break;
    }

    c += size;
    JENKINS_MIX(a, b, c);

    return c;
}

unsigned int hashBits(unsigned int a, unsigned int b, unsigned int c) {
    c += HASH_MAGIC;
    JENKINS_MIX(a, b, c);
    return c;
}

unsigned int hashBits(unsigned int a, unsigned int b, unsigned int c, unsigned int d, unsigned int e, unsigned int f) {
    c += HASH_MAGIC;
    JENKINS_MIX(a, b, c);
    a += d;
    b += e;
    c += f;
    JENKINS_MIX(a, b, c);
    return c;
}
