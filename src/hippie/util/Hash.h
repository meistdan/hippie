/**
  * \file	Hash.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	A header file containing some useful hash functions.
  */

#ifndef _HASH_H_
#define _HASH_H_

#include <QString>

#define HASH_MAGIC (0x9e3779b9u)

#define JENKINS_MIX(a, b, c)   \
    a -= b; a -= c; a ^= (c>>13); \
    b -= c; b -= a; b ^= (a<<8);  \
    c -= a; c -= b; c ^= (b>>13); \
    a -= b; a -= c; a ^= (c>>12); \
    b -= c; b -= a; b ^= (a<<16); \
    c -= a; c -= b; c ^= (b>>5);  \
    a -= b; a -= c; a ^= (c>>3);  \
    b -= c; b -= a; b ^= (a<<10); \
    c -= a; c -= b; c ^= (b>>15);

unsigned int hashString(const QString & str);
unsigned int hashBuffer(const void * ptr, int size);
unsigned int hashBufferAlign(const void * ptr, int size);
unsigned int hashBits(unsigned int a, unsigned int b = HASH_MAGIC, unsigned int c = 0);
unsigned int hashBits(unsigned int a, unsigned int b, unsigned int c, unsigned int d, unsigned int e = 0, unsigned int f = 0);

#endif /* _HASH_H_ */
