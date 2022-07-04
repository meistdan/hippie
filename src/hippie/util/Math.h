/**
 * \file	Math.h
 * \author	Daniel Meister
 * \date	2021/12/08
 * \brief	Vector and matrices header file.
 */

#ifndef _MATH_H_
#define _MATH_H_

#include "Globals.h"

template <class T> HOST_DEVICE_INLINE T clamp(const T & v, const T & lo, const T & hi) { return MAX(MIN(v, hi), lo); }
template <class T, class V> HOST_DEVICE_INLINE V mix(const V& lo, const V& hi, const T & t) { return lo * ((T)1 - t) + hi * t; }

template <class T> HOST_DEVICE_INLINE T radians(const T& a) { return a * M_PIf / 180.0f; }
template <class T> HOST_DEVICE_INLINE T degrees(const T& a) { return a * 180.0f / M_PIf; }

template <class T> HOST_DEVICE_INLINE T sqr(const T& a) { return a * a; }
template <class T> HOST_DEVICE_INLINE T rcp(const T& a) { return (a) ? (T)1 / a : (T)0; }

template <class T, int L, class S>
class VecBase {

public:

	VecBase(void) = default;
	VecBase(const VecBase &) = default;

	HOST_DEVICE_INLINE T* data(void) { return (T*)this; }
	HOST_DEVICE_INLINE const T* data(void) const { return (T*)this; }

	HOST_DEVICE_INLINE const T & get(int idx) const { return data()[idx]; }
	HOST_DEVICE_INLINE T & get(int idx) { return data()[idx]; }
	HOST_DEVICE_INLINE void set(int idx, const T& a) { T & slot = get(idx); T old = slot; slot = a; return old; }
	template <class V, int K> HOST_DEVICE_INLINE void set(const VecBase<T, K, V> & v) { set(v.data()); }

	HOST_DEVICE_INLINE void set(const T & a) { T* tp = data(); for (int i = 0; i < L; i++) tp[i] = a; }
	HOST_DEVICE_INLINE void set(const T * ptr) { T * tp = data(); for (int i = 0; i < L; i++) tp[i] = ptr[i]; }
	HOST_DEVICE_INLINE void setZero(void) { set((T)0); }

	HOST_DEVICE_INLINE T & operator[](int idx) { return get(idx); }
	HOST_DEVICE_INLINE const T & operator[](int idx) const { return get(idx); }

	HOST_DEVICE_INLINE S operator+(void) const { return *this; }
	HOST_DEVICE_INLINE S operator-(void) const { S r; T* rp = r.data(); const T* tp = data(); for (int i = 0; i < L; ++i) rp[i] = -tp[i]; return r; }

	HOST_DEVICE_INLINE S operator+(const T & a) const { S r; T * rp = r.data(); const T * tp = data(); for (int i = 0; i < L; ++i) rp[i] = tp[i] + a; return r; }
	HOST_DEVICE_INLINE S operator-(const T & a) const { S r; T * rp = r.data(); const T * tp = data(); for (int i = 0; i < L; ++i) rp[i] = tp[i] - a; return r; }
	HOST_DEVICE_INLINE S operator*(const T & a) const { S r; T * rp = r.data(); const T * tp = data(); for (int i = 0; i < L; ++i) rp[i] = tp[i] * a; return r; }
	HOST_DEVICE_INLINE S operator/(const T & a) const { S r; T * rp = r.data(); const T * tp = data(); for (int i = 0; i < L; ++i) rp[i] = tp[i] / a; return r; }

	HOST_DEVICE_INLINE S & operator=(const T & a) { set(a); return *(S*)this; }
	HOST_DEVICE_INLINE S & operator+=(const T & a) { set(operator+(a)); return *(S*)this; }
	HOST_DEVICE_INLINE S & operator-=(const T & a) { set(operator-(a)); return *(S*)this; }
	HOST_DEVICE_INLINE S & operator*=(const T & a) { set(operator*(a)); return *(S*)this; }
	HOST_DEVICE_INLINE S & operator/=(const T & a) { set(operator/(a)); return *(S*)this; }

	template <class V> HOST_DEVICE_INLINE S & operator=(const VecBase<T, L, V> & v) { set(v); return *(S*)this; }
	template <class V> HOST_DEVICE_INLINE S & operator+=(const VecBase<T, L, V> & v) { set(operator+(v)); return *(S*)this; }
	template <class V> HOST_DEVICE_INLINE S & operator-=(const VecBase<T, L, V> & v) { set(operator-(v)); return *(S*)this; }
	template <class V> HOST_DEVICE_INLINE S & operator*= (const VecBase<T, L, V> & v) { set(operator*(v)); return *(S*)this; }
	template <class V> HOST_DEVICE_INLINE S & operator/=(const VecBase<T, L, V> & v) { set(operator/(v)); return *(S*)this; }

	template <class V> HOST_DEVICE_INLINE S operator+(const VecBase<T, L, V>& v) const { const T * tp = data(); const T * vp = v.data(); S r; T * rp = r.data(); for (int i = 0; i < L; ++i) rp[i] = tp[i] + vp[i]; return r; }
	template <class V> HOST_DEVICE_INLINE S operator-(const VecBase<T, L, V>& v) const { const T * tp = data(); const T * vp = v.data(); S r; T * rp = r.data(); for (int i = 0; i < L; ++i) rp[i] = tp[i] - vp[i]; return r; }
	template <class V> HOST_DEVICE_INLINE S operator*(const VecBase<T, L, V>& v) const { const T * tp = data(); const T * vp = v.data(); S r; T * rp = r.data(); for (int i = 0; i < L; ++i) rp[i] = tp[i] * vp[i]; return r; }
	template <class V> HOST_DEVICE_INLINE S operator/(const VecBase<T, L, V>& v) const { const T * tp = data(); const T * vp = v.data(); S r; T * rp = r.data(); for (int i = 0; i < L; ++i) rp[i] = tp[i] / vp[i]; return r; }

	template <class V> HOST_DEVICE_INLINE bool operator==(const VecBase<T, L, V> & v) const { const T * tp = data(); const T * vp = v.data(); for (int i = 0; i < L; ++i) if (tp[i] != vp[i]) return false; return true; }
	template <class V> HOST_DEVICE_INLINE bool operator!=(const VecBase<T, L, V> & v) const { return (!operator==(v)); }
};

template <class T, int L, class S> HOST_DEVICE_INLINE T lenSqr(const VecBase<T, L, S> & v) { const T * tp = v.data(); T r = (T)0; for (int i = 0; i < L; ++i) r += sqr(tp[i]); return r; }
template <class T, int L, class S> HOST_DEVICE_INLINE T length(const VecBase<T, L, S> & v) { return sqrt(lenSqr(v)); }
template <class T, int L, class S> HOST_DEVICE_INLINE S normalize(const VecBase<T, L, S> & v) { return rcp(length(v)) * v; }

template <class T, int L, class S> HOST_DEVICE_INLINE S abs(const VecBase<T, L, S> & v) { S r; T* rp = r.data(); const T * tp = v.data(); for (int i = 0; i < L; ++i) rp[i] = abs(tp[i]); return r; }
template <class T, int L, class S> HOST_DEVICE_INLINE T dot(const VecBase<T, L, S> & u, const VecBase<T, L, S> & v) { const T * tp = u.data(); const T* vp = v.data(); T r = (T)0; for (int i = 0; i < L; ++i) r += tp[i] * vp[i]; return r; }

template <class T, int L, class S> HOST_DEVICE_INLINE S operator+(const T & a, const VecBase<T, L, S>& b) { return b + a; }
template <class T, int L, class S> HOST_DEVICE_INLINE S operator-(const T & a, const VecBase<T, L, S>& b) { return -b + a; }
template <class T, int L, class S> HOST_DEVICE_INLINE S operator*(const T & a, const VecBase<T, L, S>& b) { return b * a; }
template <class T, int L, class S> HOST_DEVICE_INLINE S operator/(const T & a, const VecBase<T, L, S>& b) { const T * bp = b.data(); S r; T * rp = r.data(); for (int i = 0; i < L; i++) rp[i] = a / bp[i]; return r; }

template <class T, int L>
class Vec : public VecBase<T, L, Vec<T, L>> {
private:
	T values[L];
public:
	HOST_DEVICE_INLINE  Vec(void) { VecBase<T, L, Vec<T, L>>::setZero(); }
	HOST_DEVICE_INLINE  Vec(T a) { VecBase<T, L, Vec<T, L>>::set(a); }
	template <class V> HOST_DEVICE_INLINE Vec(const VecBase<T, L, V>& v) { VecBase<T, L, Vec<T, L>>::set(v); }
	template <class V> HOST_DEVICE_INLINE Vec & operator=(const VecBase<T, L, V> & v) { VecBase<T, L, Vec<T, L>>::set(v); return *this; }
};

class Vec2i : public VecBase<int, 2, Vec2i> {
public:
	int x, y;
	HOST_DEVICE_INLINE Vec2i(void) { setZero(); }
	HOST_DEVICE_INLINE Vec2i(int a) { set(a); }
	HOST_DEVICE_INLINE Vec2i(int x, int y) { this->x = x; this->y = y; }
	template <class V> HOST_DEVICE_INLINE Vec2i(const VecBase<int, 2, V> & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec2i(const VecBase<int, 3, V> & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec2i(const VecBase<int, 4, V> & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec2i & operator=(const VecBase<int, 2, V> & v) { set(v); return *this; }
};

class Vec3i : public VecBase<int, 3, Vec3i> {
public:
	int x, y, z;
	HOST_DEVICE_INLINE Vec3i(void) { setZero(); }
	HOST_DEVICE_INLINE Vec3i(int a) { set(a); }
	HOST_DEVICE_INLINE Vec3i(int x, int y, int z) { this->x = x; this->y = y; this->z = z; }
	HOST_DEVICE_INLINE Vec3i(const Vec2i & xy, int z) { x = xy.x; y = xy.y; this->z = z; }
	template <class V> HOST_DEVICE_INLINE Vec3i(const VecBase<int, 3, V> & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec3i(const VecBase<int, 4, V> & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec3i& operator=(const VecBase<int, 3, V> & v) { set(v); return *this; }
};

class Vec4i : public VecBase<int, 4, Vec4i> {
public:
	int x, y, z, w;
	HOST_DEVICE_INLINE Vec4i(void) { setZero(); }
	HOST_DEVICE_INLINE Vec4i(int a) { set(a); }
	HOST_DEVICE_INLINE Vec4i(int x, int y, int z, int w) { this->x = x; this->y = y; this->z = z; this->w = w; }
	HOST_DEVICE_INLINE Vec4i(const Vec2i & xy, int z, int w) { x = xy.x; y = xy.y; this->z = z; this->w = w; }
	HOST_DEVICE_INLINE Vec4i(const Vec3i & xyz, int w) { x = xyz.x; y = xyz.y; z = xyz.z; this->w = w; }
	HOST_DEVICE_INLINE Vec4i(const Vec2i & xy, const Vec2i & zw) { x = xy.x; y = xy.y; z = zw.x; w = zw.y; }
	template <class V> HOST_DEVICE_INLINE Vec4i(const VecBase<int, 4, V> & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec4i& operator=(const VecBase<int, 4, V> & v) { set(v); return *this; }
};

class Vec2f : public VecBase<float, 2, Vec2f> {
public:
	float x, y;
	HOST_DEVICE_INLINE Vec2f(void) { setZero(); }
	HOST_DEVICE_INLINE Vec2f(float a) { set(a); }
	HOST_DEVICE_INLINE Vec2f(float x, float y) { this->x = x; this->y = y; }
	HOST_DEVICE_INLINE Vec2f(const Vec2i & v) { x = (float)v.x; y = (float)v.y; }
	HOST_DEVICE_INLINE operator Vec2i(void) const { return Vec2i((int)x, (int)y); }
	template <class V> HOST_DEVICE_INLINE Vec2f(const VecBase <float, 2, V > & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec2f(const VecBase <float, 3, V > & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec2f(const VecBase <float, 4, V > & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec2f& operator=(const VecBase<float, 2, V> & v) { set(v); return *this; }
};

class Vec3f : public VecBase<float, 3, Vec3f> {
public:
	float x, y, z;
	HOST_DEVICE_INLINE Vec3f(void) { setZero(); }
	HOST_DEVICE_INLINE Vec3f(float a) { set(a); }
	HOST_DEVICE_INLINE Vec3f(float x, float y, float z) { this->x = x; this->y = y; this->z = z; }
	HOST_DEVICE_INLINE Vec3f(const Vec2f & xy, float z) { x = xy.x; y = xy.y; this->z = z; }
	HOST_DEVICE_INLINE Vec3f(const Vec3i & v) { x = (float)v.x; y = (float)v.y; z = (float)v.z; }
	HOST_DEVICE_INLINE operator Vec3i(void) const { return Vec3i((int)x, (int)y, (int)z); }
	template <class V> HOST_DEVICE_INLINE Vec3f(const VecBase<float, 3, V> & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec3f(const VecBase<float, 4, V> & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec3f& operator=(const VecBase<float, 3, V> & v) { set(v); return *this; }
};

class Vec4f : public VecBase<float, 4, Vec4f> {
public:
	float x, y, z, w;
	HOST_DEVICE_INLINE Vec4f(void) { setZero(); }
	HOST_DEVICE_INLINE Vec4f(float a) { set(a); }
	HOST_DEVICE_INLINE Vec4f(float x, float y, float z, float w) { this->x = x; this->y = y; this->z = z; this->w = w; }
	HOST_DEVICE_INLINE Vec4f(const Vec2f & xy, float z, float w) { x = xy.x; y = xy.y; this->z = z; this->w = w; }
	HOST_DEVICE_INLINE Vec4f(const Vec3f & xyz, float w) { x = xyz.x; y = xyz.y; z = xyz.z; this->w = w; }
	HOST_DEVICE_INLINE Vec4f(const Vec2f & xy, const Vec2f & zw) { x = xy.x; y = xy.y; z = zw.x; w = zw.y; }
	HOST_DEVICE_INLINE Vec4f(const Vec4i & v) { x = (float)v.x; y = (float)v.y; z = (float)v.z; w = (float)v.w; }
	HOST_DEVICE_INLINE operator Vec4i(void) const { return Vec4i((int)x, (int)y, (int)z, (int)w); }
	template <class V> HOST_DEVICE_INLINE Vec4f(const VecBase<float, 4, V> & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Vec4f& operator=(const VecBase<float, 4, V> & v) { set(v); return *this; }
};

HOST_DEVICE_INLINE float cross(const Vec2f & u, const Vec2f & v) { return u.x * v.y - u.y * v.x; }
HOST_DEVICE_INLINE Vec3f cross(const Vec3f & u, const Vec3f & v) { return Vec3f(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x); }

HOST_DEVICE_INLINE Vec3f byteToFloat(unsigned int bytes) { return Vec3f((float)(bytes & 0xff) * (1.0f / 255.0f), (float)((bytes >> 8) & 0xff) * (1.0f / 255.0f), (float)((bytes >> 16) & 0xff) * (1.0f / 255.0f)); }
HOST_DEVICE_INLINE unsigned int floatToByte(const Vec3f & floats) { return ((unsigned int)(fminf(fmaxf(floats.x, 0.0f), 1.0f) * 255.0f)) | ((unsigned int)(fminf(fmaxf(floats.y, 0.0f), 1.0f) * 255.0f) << 8) | ((unsigned int)(fminf(fmaxf(floats.z, 0.0f), 1.0f) * 255.0f) << 16) | (255u << 24u); }

#define MINMAX(T, L, S) \
HOST_DEVICE_INLINE S min(const S & u, const S & v) { const T * tp = u.data(); const T * vp = v.data(); S r; T* rp = r.data(); for (int i = 0; i < L; ++i) rp[i] = MIN(tp[i], vp[i]); return r; } \
HOST_DEVICE_INLINE S max(const S & u, const S & v) { const T* tp = u.data(); const T * vp = v.data(); S r; T * rp = r.data(); for (int i = 0; i < L; ++i) rp[i] = MAX(tp[i], vp[i]); return r; } \
HOST_DEVICE_INLINE S clamp(const S & v, const S & lo, const S & hi) { const T * tp = v.data(); const T * lop = lo.data(); const T * hip = hi.data(); S r; T * rp = r.data(); for (int i = 0; i < L; ++i) rp[i] = MIN(MAX(tp[i], lop[i]), hip[i]); return r; }
MINMAX(int, 2, Vec2i) MINMAX(int, 3, Vec3i) MINMAX(int, 4, Vec4i)
MINMAX(float, 2, Vec2f) MINMAX(float, 3, Vec3f) MINMAX(float, 4, Vec4f)
#undef MINMAX

template <class T, int L, class S> 
class MatBase {

public:

	MatBase(void) = default;
	MatBase(const MatBase&) = default;

	HOST_DEVICE_INLINE const T * data(void) const { return (T*)this; }
	HOST_DEVICE_INLINE T * data(void) { return (T*)this; }

	HOST_DEVICE_INLINE const T & get(int idx) const { return data()[idx]; }
	HOST_DEVICE_INLINE T & get(int idx) { return data()[idx]; }

	HOST_DEVICE_INLINE const T & get(int r, int c) const { return data()[r + c * L]; }
	HOST_DEVICE_INLINE T & get(int r, int c) { return data()[r + c * L]; }

	HOST_DEVICE_INLINE T set(int idx, const T & a) { T & slot = get(idx); T old = slot; slot = a; return old; }
	HOST_DEVICE_INLINE T set(int r, int c, const T & a) { T & slot = get(r, c); T old = slot; slot = a; return old; }

	HOST_DEVICE_INLINE void set(const T & a) { for (int i = 0; i < L * L; ++i) get(i) = a; }
	HOST_DEVICE_INLINE void set(const T * ptr) { for (int i = 0; i < L * L; ++i) get(i) = ptr[i]; }
	HOST_DEVICE_INLINE void setZero(void) { set((T)0); }
	HOST_DEVICE_INLINE void setIdentity(void) { setZero(); for (int i = 0; i < L; ++i) get(i, i) = (T)1; }

	HOST_DEVICE_INLINE const Vec<T, L> & col(int c) const { return *(const Vec<T, L>*)(data() + c * L); }
	HOST_DEVICE_INLINE Vec<T, L> & col(int c) { return *(Vec<T, L>*)(data() + c * L); }
	HOST_DEVICE_INLINE const Vec<T, L> & getCol(int c) const { return col(c); }
	HOST_DEVICE_INLINE Vec<T, L> getRow(int r) const { Vec<T, L> r; for (int i = 0; i < L; ++i) r[i] = get(idx, i); return r; }

	template <class V> HOST_DEVICE_INLINE void setCol(int c, const VecBase<T, L, V> & v) { col(c) = v; }
	template <class V> HOST_DEVICE_INLINE void setRow(int r, const VecBase<T, L, V> & v) { for (int i = 0; i < L; ++i) get(r, i) = v[i]; }
	template <class V> HOST_DEVICE_INLINE void set(const MatBase<T, L, V> & v) { set(v.data()); }

	HOST_DEVICE_INLINE const T & operator()(int r, int c) const { return get(r, c); }
	HOST_DEVICE_INLINE T & operator()(int r, int c) { return get(r, c); }

	HOST_DEVICE_INLINE Vec<T, L> & operator[](int idx) { return col(idx); }
	HOST_DEVICE_INLINE const Vec<T, L> & operator[](int idx) const { return col(idx); }

	HOST_DEVICE_INLINE S operator+(void) const { return *this; }
	HOST_DEVICE_INLINE S operator-(void) const { S r; for (int i = 0; i < L * L; ++i) r.get(i) = -get(i); return r; }

	HOST_DEVICE_INLINE S & operator=(const T & a) { set(a); return *(S*)this; }
	HOST_DEVICE_INLINE S & operator+=(const T & a) { set(operator+(a)); return *(S*)this; }
	HOST_DEVICE_INLINE S & operator-=(const T & a) { set(operator-(a)); return *(S*)this; }
	HOST_DEVICE_INLINE S & operator*=(const T & a) { set(operator*(a)); return *(S*)this; }
	HOST_DEVICE_INLINE S & operator/=(const T & a) { set(operator/(a)); return *(S*)this; }

	HOST_DEVICE_INLINE S operator+(const T & a) const { S r; for (int i = 0; i < L * L; ++i) r.get(i) = get(i) + a; return r; }
	HOST_DEVICE_INLINE S operator-(const T & a) const { S r; for (int i = 0; i < L * L; ++i) r.get(i) = get(i) - a; return r; }
	HOST_DEVICE_INLINE S operator*(const T & a) const { S r; for (int i = 0; i < L * L; ++i) r.get(i) = get(i) * a; return r; }
	HOST_DEVICE_INLINE S operator/(const T & a) const { S r; for (int i = 0; i < L * L; ++i) r.get(i) = get(i) / a; return r; }

	template <class V> HOST_DEVICE_INLINE S & operator=(const MatBase<T, L, V> & v) { set(v); return *(S*)this; }
	template <class V> HOST_DEVICE_INLINE S & operator+=(const MatBase<T, L, V> & v) { set(operator+(v)); return *(S*)this; }
	template <class V> HOST_DEVICE_INLINE S & operator-=(const MatBase<T, L, V> & v) { set(operator-(v)); return *(S*)this; }
	template <class V> HOST_DEVICE_INLINE S & operator*=(const MatBase<T, L, V> & v) { set(operator*(v)); return *(S*)this; }
	template <class V> HOST_DEVICE_INLINE S & operator/=(const MatBase<T, L, V> & v) { set(operator/(v)); return *(S*)this; }

	template <class V> HOST_DEVICE_INLINE V operator*(const VecBase<T, L, V> & v) const;

	template <class V> HOST_DEVICE_INLINE S operator+(const MatBase<T, L, V> & v) const { S r; for (int i = 0; i < L * L; ++i) r.get(i) = get(i) + v.get(i); return r; }
	template <class V> HOST_DEVICE_INLINE S operator-(const MatBase<T, L, V> & v) const { S r; for (int i = 0; i < L * L; ++i) r.get(i) = get(i) - v.get(i); return r; }
	template <class V> HOST_DEVICE_INLINE S operator*(const MatBase<T, L, V> & v) const;
	template <class V> HOST_DEVICE_INLINE S operator/(const MatBase<T, L, V> & v) const;

	template <class V> HOST_DEVICE_INLINE bool operator==(const MatBase<T, L, V> & v) const { for (int i = 0; i < L * L; i++) if (get(i) != v.get(i)) return false; return true; }
	template <class V> HOST_DEVICE_INLINE bool operator!=(const MatBase<T, L, V> & v) const { return (!operator==(v)); }
};

template <class T, int L, class S> HOST_DEVICE_INLINE S operator+(const T & a, const MatBase<T, L, S> & b) { return b + a; }
template <class T, int L, class S> HOST_DEVICE_INLINE S operator-(const T & a, const MatBase<T, L, S> & b) { return -b + a; }
template <class T, int L, class S> HOST_DEVICE_INLINE S operator*(const T & a, const MatBase<T, L, S> & b) { return b * a; }
template <class T, int L, class S> HOST_DEVICE_INLINE S operator/(const T & a, const MatBase<T, L, S> & b) { S r; for (int i = 0; i < L * L; ++i) r.get(i) = a / b.get(i); return r; }

template <class T, int L> 
class Mat : public MatBase<T, L, Mat<T, L>> {
private:
	T values[L * L];
public:
	HOST_DEVICE_INLINE Mat(void) { MatBase<T, L, Mat<T, L>>::setIdentity(); }
	HOST_DEVICE_INLINE Mat(T a) { MatBase<T, L, Mat<T, L>>::set(a); }
	template <class V> HOST_DEVICE_INLINE Mat(const MatBase<T, L, V>& v) { MatBase<T, L, Mat<T, L>>::set(v); }
	template <class V> HOST_DEVICE_INLINE Mat & operator=(const MatBase<T, L, V> & v) { MatBase<T, L, Mat<T, L>>::set(v); return *this; }
};

class Mat2f : public MatBase<float, 2, Mat2f>
{
public:
	float m00, m10, m01, m11;
	HOST_DEVICE_INLINE Mat2f(void) { setIdentity(); }
	HOST_DEVICE_INLINE Mat2f(float a) { set(a); }
	HOST_DEVICE_INLINE Mat2f(float m00, float m10, float m01, float m11) : m00(m00), m10(m10), m01(m01), m11(m11) {}
	template <class V> HOST_DEVICE_INLINE Mat2f(const MatBase<float, 2, V> & v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Mat2f & operator=(const MatBase<float, 2, V> & v) { set(v); return *this; }
};

class Mat3f : public MatBase<float, 3, Mat3f>
{
public:
	float m00, m10, m20, m01, m11, m21, m02, m12, m22;
	HOST_DEVICE_INLINE Mat3f(void) { setIdentity(); }
	HOST_DEVICE_INLINE Mat3f(float a) { set(a); }
	HOST_DEVICE_INLINE Mat3f(float m00, float m10, float m20, float m01, float m11, float m21, float m02, float m12, float m22) 
		: m00(m00), m10(m10), m20(m20), m01(m01), m11(m11), m21(m21), m02(m02), m12(m12), m22(m22) {}
	template <class V> HOST_DEVICE_INLINE Mat3f(const MatBase<float, 3, V>& v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Mat3f& operator=(const MatBase<float, 3, V>& v) { set(v); return *this; }
};

class Mat4f : public MatBase<float, 4, Mat4f>
{
public:
	float m00, m10, m20, m30, m01, m11, m21, m31, m02, m12, m22, m32, m03, m13, m23, m33;
	HOST_DEVICE_INLINE Mat4f(void) { setIdentity(); }
	HOST_DEVICE_INLINE Mat4f(float a) { set(a); }
	HOST_DEVICE_INLINE Mat4f(float m00, float m10, float m20, float m30, float m01, float m11, float m21, float m31, float m02, float m12, float m22, float m32, float m03, float m13, float m23, float m33)
		: m00(m00), m10(m10), m20(m20), m30(m30), m01(m01), m11(m11), m21(m21), m31(m31), m02(m12), m12(m12), m22(m22), m32(m32), m03(m03), m13(m13), m23(m23), m33(m33)  {}
	template <class V> HOST_DEVICE_INLINE Mat4f(const MatBase<float, 4, V>& v) { set(v); }
	template <class V> HOST_DEVICE_INLINE Mat4f& operator=(const MatBase<float, 4, V>& v) { set(v); return *this; }
};

HOST_DEVICE Mat4f perspective(float fovy, float aspect, float near, float far);
HOST_DEVICE Mat4f rotate(float angle, const Vec3f& axis);
HOST_DEVICE Mat4f translate(const Vec3f& xyz);
HOST_DEVICE Mat4f scale(const Vec3f& xyz);

template <class T, class S> HOST_DEVICE_INLINE T detImpl(const MatBase<T, 1, S>& v) {
	return v(0, 0);
}

template <class T, class S> HOST_DEVICE_INLINE T detImpl(const MatBase<T, 2, S>& v) {
	return v(0, 0) * v(1, 1) - v(0, 1) * v(1, 0);
}

template <class T, class S> HOST_DEVICE_INLINE T detImpl(const MatBase<T, 3, S>& v) {
	return v(0, 0) * v(1, 1) * v(2, 2) - v(0, 0) * v(1, 2) * v(2, 1) +
		v(1, 0) * v(2, 1) * v(0, 2) - v(1, 0) * v(2, 2) * v(0, 1) +
		v(2, 0) * v(0, 1) * v(1, 2) - v(2, 0) * v(0, 2) * v(1, 1);
}

template <class T, int L, class S> HOST_DEVICE_INLINE T detImpl(const MatBase<T, L, S> & v) {
	T r = (T)0;
	T s = (T)1;
	for (int i = 0; i < L; ++i) {
		Mat<T, L - 1> sub;
		for (int j = 0; j < L - 1; ++j)
			for (int k = 0; k < L - 1; ++k)
				sub(j, k) = v((j < i) ? j : j + 1, k + 1);
		r += determinant(sub) * v(i, 0) * s;
		s = -s;
	}
	return r;
}

template <class T, int L, class S> HOST_DEVICE_INLINE T determinant(const MatBase<T, L, S> & v) {
	return detImpl(v);
}

template <class T, int L, class S> HOST_DEVICE_INLINE S inverse(const MatBase<T, L, S> & v) {
	S r;
	T d = (T)0;
	T si = (T)1;
	for (int i = 0; i < L; ++i) {
		T sj = si;
		for (int j = 0; j < L; ++j) {
			Mat<T, L - 1> sub;
			for (int k = 0; k < L - 1; ++k)
				for (int l = 0; l < L - 1; ++l)
					sub(k, l) = v.get((k < j) ? k : k + 1, (l < i) ? l : l + 1);
			T dd = determinant(sub) * sj;
			r(i, j) = dd;
			d += dd * v.get(j, i);
			sj = -sj;
		}
		si = -si;
	}
	return r * rcp(d) * L;
}

template <class T, int L, class S> HOST_DEVICE_INLINE S transpose(const MatBase<T, L, S> & v) {
	S r;
	for (int i = 0; i < L; ++i)
		for (int j = 0; j < L; ++j)
			r(i, j) = v.get(j, i);
	return r;
}

template <class T, int L, class S> template <class V> HOST_DEVICE_INLINE V MatBase<T, L, S>::operator*(const VecBase<T, L, V> & v) const {
	V r;
	for (int i = 0; i < L; ++i) {
		T rr = (T)0;
		for (int j = 0; j < L; ++j)
			rr += get(i, j) * v[j];
		r[i] = rr;
	}
	return r;
}

template <class T, int L, class S> template <class V> HOST_DEVICE_INLINE S MatBase<T, L, S>::operator*(const MatBase<T, L, V> & v) const {
	S r;
	for (int i = 0; i < L; ++i) {
		for (int j = 0; j < L; ++j) {
			T rr = (T)0;
			for (int k = 0; k < L; ++k)
				rr += get(i, k) * v(k, j);
			r(i, j) = rr;
		}
	}
	return r;
}

template <class T, int L, class S> template <class V> HOST_DEVICE_INLINE S MatBase<T, L, S>::operator/(const MatBase<T, L, V> & v) const {
	return operator*(inverse(v))
}

#endif /* _MATH_H_ */
