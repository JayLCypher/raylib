/**********************************************************************************************
*
*   raymath v1.5 - Math functions to work with Vector2, Vector3, Matrix and Quaternions
*
*   CONVENTIONS:
*     - Matrix structure is defined as row-major (memory layout) but parameters naming AND all
*       math operations performed by the library consider the structure as it was column-major
*       It is like transposed versions of the matrices are used for all the maths
*       It benefits some functions making them cache-friendly and also avoids matrix
*       transpositions sometimes required by OpenGL
*       Example: In memory order, row0 is [m0 m4 m8 m12] but in semantic math row0 is [m0 m1 m2 m3]
*     - Functions are always self-contained, no function use another raymath function inside,
*       required code is directly re-implemented inside
*     - Functions input parameters are always received by value (2 unavoidable exceptions)
*     - Functions use always a "result" variable for return
*     - Functions are always defined inline
*     - Angles are always in radians (DEG2RAD/RAD2DEG macros provided for convenience)
*     - No compound literals used to make sure libray is compatible with C++
*
*   CONFIGURATION:
*       #define RAYMATH_IMPLEMENTATION
*           Generates the implementation of the library into the included file.
*           If not defined, the library is in header only mode and can be included in other headers
*           or source files without problems. But only ONE file should hold the implementation.
*
*       #define RAYMATH_STATIC_INLINE
*           Define static inline functions code, so #include header suffices for use.
*           This may use up lots of memory.
*
*
*   LICENSE: zlib/libpng
*
*   Copyright (c) 2015-2024 Ramon Santamaria (@raysan5)
*
*   This software is provided "as-is", without any express or implied warranty. In no event
*   will the authors be held liable for any damages arising from the use of this software.
*
*   Permission is granted to anyone to use this software for any purpose, including commercial
*   applications, and to alter it and redistribute it freely, subject to the following restrictions:
*
*     1. The origin of this software must not be misrepresented; you must not claim that you
*     wrote the original software. If you use this software in a product, an acknowledgment
*     in the product documentation would be appreciated but is not required.
*
*     2. Altered source versions must be plainly marked as such, and must not be misrepresented
*     as being the original software.
*
*     3. This notice may not be removed or altered from any source distribution.
*
**********************************************************************************************/

#ifndef RAYMATH_H
#define RAYMATH_H

#if defined(RAYMATH_IMPLEMENTATION) && defined(RAYMATH_STATIC_INLINE)
    #error "Specifying both RAYMATH_IMPLEMENTATION and RAYMATH_STATIC_INLINE is contradictory"
#endif

// Function specifiers definition
#if defined(RAYMATH_IMPLEMENTATION)
    #if defined(_WIN32) && defined(BUILD_LIBTYPE_SHARED)
        #define RMAPI __declspec(dllexport) extern inline // We are building raylib as a Win32 shared library (.dll)
    #elif defined(BUILD_LIBTYPE_SHARED)
        #define RMAPI __attribute__((visibility("default"))) // We are building raylib as a Unix shared library (.so/.dylib)
    #elif defined(_WIN32) && defined(USE_LIBTYPE_SHARED)
        #define RMAPI __declspec(dllimport)         // We are using raylib as a Win32 shared library (.dll)
    #else
        #define RMAPI extern inline // Provide external definition
    #endif
#elif defined(RAYMATH_STATIC_INLINE)
    #define RMAPI static inline // Functions may be inlined, no external out-of-line definition
#else
    #if defined(__TINYC__)
        #define RMAPI static inline // plain inline not supported by tinycc (See issue #435)
    #else
        #define RMAPI inline        // Functions may be inlined or external definition used
    #endif
#endif


//----------------------------------------------------------------------------------
// Defines and Macros
//----------------------------------------------------------------------------------
#ifndef PI
    #define PI 3.14159265358979323846f
#endif

#ifndef EPSILON
    #define EPSILON 0.000001f
#endif

#ifndef DEG2RAD
    #define DEG2RAD (PI/180.0f)
#endif

#ifndef RAD2DEG
    #define RAD2DEG (180.0f/PI)
#endif

// Get float vector for Matrix
#ifndef MatrixToFloat
    #define MatrixToFloat(mat) (MatrixToFloatV(mat).v)
#endif

// Get float vector for Vector3
#ifndef Vector3ToFloat
    #define Vector3ToFloat(vec) (Vector3ToFloatV(vec).v)
#endif

//----------------------------------------------------------------------------------
// Types and Structures Definition
//----------------------------------------------------------------------------------
#if !defined(RL_VECTOR2_TYPE)
// Vector2 type
typedef struct Vector2 {
    float x;
    float y;
} Vector2;
#define RL_VECTOR2_TYPE
#endif

#if !defined(RL_VECTOR3_TYPE)
// Vector3 type
typedef struct Vector3 {
    float x;
    float y;
    float z;
} Vector3;
#define RL_VECTOR3_TYPE
#endif

#if !defined(RL_VECTOR4_TYPE)
// Vector4 type
typedef struct Vector4 {
    float x;
    float y;
    float z;
    float w;
} Vector4;
#define RL_VECTOR4_TYPE
#endif

#if !defined(RL_QUATERNION_TYPE)
// Quaternion type
typedef Vector4 Quaternion;
#define RL_QUATERNION_TYPE
#endif

#if !defined(RL_MATRIX_TYPE)
// Matrix type (OpenGL style 4x4 - right handed, column major)
typedef struct Matrix {
    float m0, m4, m8, m12;      // Matrix first row (4 components)
    float m1, m5, m9, m13;      // Matrix second row (4 components)
    float m2, m6, m10, m14;     // Matrix third row (4 components)
    float m3, m7, m11, m15;     // Matrix fourth row (4 components)
} Matrix;
#define RL_MATRIX_TYPE
#endif

// NOTE: Helper types to be used instead of array return types for *ToFloat functions
typedef struct float3 {
    float v[3];
} float3;

typedef struct float16 {
    float v[16];
} float16;

#include <math.h>       // Required for: sinf(), cosf(), tan(), atan2f(), sqrtf(), floor(), fminf(), fmaxf(), fabsf()

//----------------------------------------------------------------------------------
// Module Functions Definition - Utils math
//----------------------------------------------------------------------------------

// Clamp float value using ternary: if vaule less than min then min elseif value greater than max then max else value.
RMAPI float Clamp(const float value, const float min, const float max)
{
    return (value < min) ? min : (value > max) ? max : value;
}

// Calculate linear interpolation between two floats
RMAPI float Lerp(const float start, const float end, const float amount)
{
    return start + (amount * (end - start));
}

// Normalize input value within input range
RMAPI float Normalize(const float value, const float start, const float end)
{
    return (value - start) / (end - start);
}

// Remap input value within input range to output range
RMAPI float Remap(const float value, const float inputStart, const float inputEnd, const  float outputStart, const float outputEnd)
{
    return ((value - inputStart) / (inputEnd - inputStart) * (outputEnd - outputStart)) + outputStart;
}

// Wrap input value from min to max
RMAPI float Wrap(const float value, const float min, const float max)
{
    return value - ((max - min) * floorf((value - min) / (max - min)));
}

// Check whether two given floats are almost equal
RMAPI int FloatEquals(float x, float y)
{
#if !defined(EPSILON)
    #define EPSILON 0.000001f
#endif
    return fabsf(x - y) <= (EPSILON * fmaxf(1.0f, fmaxf(fabsf(x), fabsf(y))));
}

//----------------------------------------------------------------------------------
// Module Functions Definition - Vector2 math
//----------------------------------------------------------------------------------

// Vector with components value 0.0f
RMAPI Vector2 Vector2Zero(void)
{
	Vector2 result = {0};
	return result;
}

// Vector with components value 1.0f
RMAPI Vector2 Vector2One(void)
{
	Vector2 result = { 1.0f, 1.0f };
	return result;
}

// Add two vectors (v1 + v2)
RMAPI Vector2 Vector2Add(const Vector2 v1, const Vector2 v2)
{
	Vector2 result = { v1.x + v2.x, v1.y + v2.y };
    return result;
}

// Add vector and float value
RMAPI Vector2 Vector2AddValue(const Vector2 v, const float add)
{

    Vector2 result = { v.x + add, v.y + add };
	return result;
}

// Subtract two vectors (v1 - v2)
RMAPI Vector2 Vector2Subtract(const Vector2 v1, const Vector2 v2)
{
	Vector2 result = { v1.x - v2.x, v1.y - v2.y };
    return result;
}

// Subtract vector by float value
RMAPI Vector2 Vector2SubtractValue(const Vector2 v, const float sub)
{
	Vector2 result = { v.x - sub, v.y - sub };
    return result;
}

// Calculate vector length
RMAPI float Vector2Length(const Vector2 v)
{
    return sqrtf((v.x * v.x) + (v.y * v.y));
}

// Calculate vector square length
RMAPI float Vector2LengthSqr(const Vector2 v)
{
    return (v.x * v.x) + (v.y * v.y);
}

// Calculate two vectors dot product
RMAPI float Vector2DotProduct(const Vector2 v1, const Vector2 v2)
{
    return (v1.x * v2.x + v1.y * v2.y);
}

// Calculate distance between two vectors
RMAPI float Vector2Distance(const Vector2 v1, const Vector2 v2)
{
    return sqrtf(((v1.x - v2.x) * (v1.x - v2.x)) + ((v1.y - v2.y) * (v1.y - v2.y)));
}

// Calculate square distance between two vectors
RMAPI float Vector2DistanceSqr(const Vector2 v1, const Vector2 v2)
{
    return (((v1.x - v2.x) * (v1.x - v2.x)) + ((v1.y - v2.y) * (v1.y - v2.y)));
}

// Calculate angle between two vectors
// NOTE: Angle is calculated from origin point (0, 0)
RMAPI float Vector2Angle(const Vector2 v1, const Vector2 v2)
{
	//							DOT							DET
	return atan2f((v1.x * v2.x) + (v1.y * v2.y), (v1.x * v2.y) - (v1.y * v2.x));
}

// Calculate angle defined by a two vectors line
// NOTE: Parameters need to be normalized
// Current implementation should be aligned with glm::angle
RMAPI float Vector2LineAngle(const Vector2 start, const Vector2 end)
{
    // TODO(10/9/2023): Currently angles move clockwise, determine if this is wanted behavior
    return -atan2f(end.y - start.y, end.x - start.x);
}

// Scale vector (multiply by value)
RMAPI Vector2 Vector2Scale(const Vector2 v, const float scale)
{
	Vector2 result = { v.x * scale, v.y * scale };
	return result;
}

// Multiply vector by vector
RMAPI Vector2 Vector2Multiply(const Vector2 v1, const Vector2 v2)
{
    Vector2 result = { (v1.x * v2.x), (v1.y * v2.y) };
	return result;
}

// Negate vector
RMAPI Vector2 Vector2Negate(const Vector2 v)
{
    Vector2 result = { -v.x, -v.y };
	return result;
}

// Divide vector by vector
RMAPI Vector2 Vector2Divide(const Vector2 v1, const Vector2 v2)
{
    Vector2 result = { (v1.x / v2.x), (v1.y / v2.y) };
    return result;
}

// Normalize provided vector
RMAPI Vector2 Vector2Normalize(const Vector2 v)
{
    const float length = sqrtf((v.x*v.x) + (v.y*v.y));
	Vector2 result = {0};
	if (length > 0.0f) {
		result.x = v.x / length;
		result.y = v.y / length;
	}
	return result;
}

// Transforms a Vector2 by a given Matrix
// NOTE: Where is mat.m2 and mat.m3?
RMAPI Vector2 Vector2Transform(const Vector2 v, const Matrix mat)
{
    Vector2 result = { 0 };
    result.x = (mat.m0 * v.x) + (mat.m4 * v.y) + (mat.m8 * 0) + mat.m12;
    result.y = (mat.m1 * v.x) + (mat.m5 * v.y) + (mat.m9 * 0) + mat.m13;
    return result;
}

// Calculate linear interpolation between two vectors
RMAPI Vector2 Vector2Lerp(const Vector2 v1, const Vector2 v2, const float amount)
{
    Vector2 result = { (v1.x + (amount * (v2.x - v1.x))), (v1.y + (amount * (v2.y - v1.y))) };
    return result;
}

// Calculate reflected vector to normal
RMAPI Vector2 Vector2Reflect(const Vector2 v, const Vector2 normal)
{
    const float dotProduct = (v.x * normal.x + v.y * normal.y); // Dot product

    Vector2 result = { 0 };
    result.x = v.x - ((2.0f * normal.x) * dotProduct);
    result.y = v.y - ((2.0f * normal.y) * dotProduct);
    return result;
}

// Get min value for each pair of components
RMAPI Vector2 Vector2Min(const Vector2 v1, const Vector2 v2)
{
    Vector2 result = { fminf(v1.x, v2.x), fminf(v1.y, v2.y) };
    return result;
}

// Get max value for each pair of components
RMAPI Vector2 Vector2Max(const Vector2 v1, const Vector2 v2)
{
    Vector2 result = { fmaxf(v1.x, v2.x), fmaxf(v1.y, v2.y) };
    return result;
}

// Rotate vector by angle
RMAPI Vector2 Vector2Rotate(const Vector2 v, const float angle)
{
    const float cosres = cosf(angle);
    const float sinres = sinf(angle);

    Vector2 result = { ((v.x * cosres) - (v.y * sinres)), ((v.x * sinres) + (v.y * cosres)) };
    return result;
}

// Move Vector towards target
RMAPI Vector2 Vector2MoveTowards(const Vector2 v, const Vector2 target, const float maxDistance)
{
    const float dx = target.x - v.x;
    const float dy = target.y - v.y;
    const float value = (dx * dx) + (dy * dy);

	// NOTE: Floating point == 0.0f is ill-advised due to inaccuracies.
    if ((value == 0.0f) || ((maxDistance >= 0.0f) && (value <= (maxDistance * maxDistance)))) { return target; }

    const float dist = sqrtf(value);

    Vector2 result = { 0 };
    result.x = v.x + (dx / dist * maxDistance);
    result.y = v.y + (dy / dist * maxDistance);
    return result;
}

// Invert the given vector
RMAPI Vector2 Vector2Invert(const Vector2 v)
{
    Vector2 result = { (1.0f / v.x) , (1.0f / v.y) };
    return result;
}

// Clamp the components of the vector between
// min and max values specified by the given vectors
RMAPI Vector2 Vector2Clamp(const Vector2 v, const Vector2 min, const Vector2 max)
{
    Vector2 result = { fminf(max.x, fmaxf(min.x, v.x)), fminf(max.y, fmaxf(min.y, v.y)) };
    return result;
}

// Clamp the magnitude of the vector between two min and max values
RMAPI Vector2 Vector2ClampValue(const Vector2 v, const float min, const float max)
{
    Vector2 result = v;
    float length = (v.x * v.x) + (v.y * v.y);
    if (length > 0.0f)
    {
        length = sqrtf(length);

        const float scale = (length < min) ? min / length : (length > max) ? max / length : 1.0f; // By default, 1 as the neutral element.
        result.x = v.x * scale;
        result.y = v.y * scale;
    }

    return result;
}

// Check whether two given vectors are almost equal
RMAPI int Vector2Equals(const Vector2 p, const Vector2 q)
{
#if !defined(EPSILON)
    #define EPSILON 0.000001f
#endif

    int result = ((fabsf(p.x - q.x)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.x), fabsf(q.x))))) &&
                  ((fabsf(p.y - q.y)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.y), fabsf(q.y)))));

    return result;
}

// Compute the direction of a refracted ray
// v: normalized direction of the incoming ray
// n: normalized normal vector of the interface of two optical media
// r: ratio of the refractive index of the medium from where the ray comes
//    to the refractive index of the medium on the other side of the surface
RMAPI Vector2 Vector2Refract(const Vector2 v, const Vector2 n, const float r)
{
    const float dot = (v.x * n.x) + (v.y * n.y);
    float d = 1.0f - (r * r * (1.0f - (dot * dot)));

    Vector2 result = { 0 };
    if (d >= 0.0f)
    {
        d = sqrtf(d);
        result.x = (r * v.x) - (((r * dot) + d) * n.x);
        result.y = (r * v.y) - (((r * dot) + d) * n.y);
    }

    return result;
}


//----------------------------------------------------------------------------------
// Module Functions Definition - Vector3 math
//----------------------------------------------------------------------------------

#define Vector3Expand(v) (v).x, (v).y, (v).z

// Vector with components value 0.0f
RMAPI Vector3 Vector3Zero(void)
{
    Vector3 result = { 0.0f, 0.0f, 0.0f };
    return result;
}

// Vector with components value 1.0f
RMAPI Vector3 Vector3One(void)
{
    Vector3 result = { 1.0f, 1.0f, 1.0f };
    return result;
}

// Add two vectors
RMAPI Vector3 Vector3Add(const Vector3 v1, const Vector3 v2)
{
    const Vector3 result = { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
    return result;
}

// Add vector and float value
RMAPI Vector3 Vector3AddValue(const Vector3 v, const float add)
{
    const Vector3 result = { v.x + add, v.y + add, v.z + add };
    return result;
}

// Subtract two vectors
RMAPI Vector3 Vector3Subtract(const Vector3 v1, const Vector3 v2)
{
    const Vector3 result = { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
    return result;
}

// Subtract vector by float value
RMAPI Vector3 Vector3SubtractValue(const Vector3 v, const float sub)
{
    const Vector3 result = { v.x - sub, v.y - sub, v.z - sub };
    return result;
}

// Multiply vector by scalar
RMAPI Vector3 Vector3Scale(const Vector3 v, const float scalar)
{
    const Vector3 result = { v.x*scalar, v.y*scalar, v.z*scalar };
    return result;
}

// Multiply vector by vector
RMAPI Vector3 Vector3Multiply(const Vector3 v1, const Vector3 v2)
{
    const Vector3 result = { v1.x*v2.x, v1.y*v2.y, v1.z*v2.z };
    return result;
}

// Calculate two vectors cross product
RMAPI Vector3 Vector3CrossProduct(Vector3 v1, Vector3 v2)
{
    const Vector3 result = { v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x };
    return result;
}

// JayLCypher:
// Utilize conditional branching to get a boolean and use that for multiplication to get a 0/1 value instead of if/else.
// Checking against godbolt gcc v14.1 (-O3) you get no branches but maybe a few more instructions.
RMAPI Vector3 Vector3Perpendicular_alt(const Vector3 v) {
	const float min_x = fabsf(v.x);
	const float min_y = fabsf(v.y);
	const float min_z = fabsf(v.z);
	const Vector3 cardinalAxis = {
		1.0f * ((min_x < (min_y < min_z ? min_y : min_z))),
		1.0f * ((min_y < (min_x < min_z ? min_x : min_z))),
		1.0f * ((min_z < (min_x < min_y ? min_x : min_y))),
	};
    // Cross product between vectors
	const Vector3 result = {
		.x = (v.y * cardinalAxis.z) - (v.z * cardinalAxis.y),
		.y = (v.z * cardinalAxis.x) - (v.x * cardinalAxis.z),
		.z = (v.x * cardinalAxis.y) - (v.y * cardinalAxis.x),
	};
    return result;
}

// Calculate one vector perpendicular vector
RMAPI Vector3 Vector3Perpendicular(const Vector3 v)
{
    const float min_x = fabsf(v.x);
    const float min_y = fabsf(v.y);
    const float min_z = fabsf(v.z);
    Vector3 cardinalAxis = { 1.0f, 0.0f, 0.0f };

    if (min_y < min_x)
    {
		cardinalAxis.x = 0.0f;
		cardinalAxis.y = 1.0f;
    }

    if (min_z < min_y)
    {
		cardinalAxis.y = 0.0f;
		cardinalAxis.z = 1.0f;
    }

    // Cross product between vectors
	const Vector3 result = {
		.x = (v.y * cardinalAxis.z) - (v.z * cardinalAxis.y),
		.y = (v.z * cardinalAxis.x) - (v.x * cardinalAxis.z),
		.z = (v.x * cardinalAxis.y) - (v.y * cardinalAxis.x),
	};
    return result;
}

// Calculate vector length
RMAPI float Vector3Length(const Vector3 v)
{
    const float result = sqrtf((v.x * v.x) + (v.y * v.y) + (v.z * v.z));
    return result;
}

// Calculate vector square length
RMAPI float Vector3LengthSqr(const Vector3 v)
{
    const float result = ((v.x * v.x) + (v.y * v.y) + (v.z * v.z));
	return result;
}

// Calculate two vectors dot product
RMAPI float Vector3DotProduct(const Vector3 v1, const Vector3 v2)
{
    const float result = ((v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z));
    return result;
}

// Calculate distance between two vectors
RMAPI float Vector3Distance(const Vector3 v1, const Vector3 v2)
{
    const float dx = v2.x - v1.x;
    const float dy = v2.y - v1.y;
    const float dz = v2.z - v1.z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

// Calculate square distance between two vectors
RMAPI float Vector3DistanceSqr(const Vector3 v1, const Vector3 v2)
{
    const float dx = v2.x - v1.x;
    const float dy = v2.y - v1.y;
    const float dz = v2.z - v1.z;
    return ((dx*dx) + (dy*dy) + (dz*dz));
}

// Calculate angle between two vectors
RMAPI float Vector3Angle(const Vector3 v1, const Vector3 v2)
{
    const Vector3 cross = { v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x };
    const float len = sqrtf(cross.x*cross.x + cross.y*cross.y + cross.z*cross.z);
    const float dot = (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
    return atan2f(len, dot);
}

// Negate provided vector (invert direction)
RMAPI Vector3 Vector3Negate(const Vector3 v)
{
    Vector3 result = { -v.x, -v.y, -v.z };
    return result;
}

// Divide vector by vector
RMAPI Vector3 Vector3Divide(const Vector3 v1, const Vector3 v2)
{
    Vector3 result = { v1.x/v2.x, v1.y/v2.y, v1.z/v2.z };
    return result;
}

// Normalize provided vector
RMAPI Vector3 Vector3Normalize(Vector3 v)
{
    const float length = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
	Vector3 result = {0};
	if (length > 0.0f) {
		result.x = v.x / length;
		result.y = v.y / length;
		result.z = v.z / length;
	}
	return result;
}

//Calculate the projection of the vector v1 on to v2
RMAPI Vector3 Vector3Project(const Vector3 v1, const Vector3 v2)
{
    const float v1dv2 = (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
    const float v2dv2 = (v2.x*v2.x + v2.y*v2.y + v2.z*v2.z);
    const float mag = v1dv2/v2dv2;

	Vector3 result = {
		.x = v2.x * mag,
		.y = v2.y * mag,
		.z = v2.z * mag,
	};
    return result;
}

//Calculate the rejection of the vector v1 on to v2
RMAPI Vector3 Vector3Reject(const Vector3 v1, const Vector3 v2)
{
    float v1dv2 = (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
    float v2dv2 = (v2.x*v2.x + v2.y*v2.y + v2.z*v2.z);
    float mag = v1dv2/v2dv2;

	Vector3 result = {
		.x = v1.x - (v2.x * mag),
		.y = v1.y - (v2.y * mag),
		.z = v1.z - (v2.z * mag),
	};
    return result;
}

// Orthonormalize provided vectors
// Makes vectors normalized and orthogonal to each other
// Implemented by using the Raylib functions instead. If these vectors should not overlap, use restrict.
RMAPI void Vector3OrthoNormalize_alt(Vector3 *v1, Vector3 *v2)
{
	const Vector3 v = Vector3Normalize(*v1);
	const Vector3 vn1 = Vector3Normalize(Vector3CrossProduct(*v1, *v2));
	*v1 = v;
	*v2 = Vector3CrossProduct(vn1, *v1);
}

// Orthonormalize provided vectors
// Makes vectors normalized and orthogonal to each other
// Gram-Schmidt function implementation
RMAPI void Vector3OrthoNormalize(Vector3 *v1, Vector3 *v2)
{
    // Vector3Normalize(*v1);
    Vector3 v = *v1;
    float length = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (length > 0.0f) {
		v1->x /= length;
		v1->y /= length;
		v1->z /= length;
	}

    // Vector3CrossProduct(*v1, *v2)
    Vector3 vn1 = { v1->y*v2->z - v1->z*v2->y, v1->z*v2->x - v1->x*v2->z, v1->x*v2->y - v1->y*v2->x };

    // Vector3Normalize(vn1);
    v = vn1;
    length = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (length != 0.0f) {
		vn1.x /= length;
		vn1.y /= length;
		vn1.z /= length;
	}

    // Vector3CrossProduct(vn1, *v1)
	Vector3 vn2 = { (vn1.y * v1->z) - (vn1.z * v1->y), (vn1.z * v1->x) - (vn1.x * v1->z), (vn1.x * v1->y) - (vn1.y * v1->x) };
    *v2 = vn2;
}

// Transforms a Vector3 by a given Matrix
RMAPI Vector3 Vector3Transform(const Vector3 v, const Matrix mat)
{
    const Vector3 result = {
		.x = (mat.m0 * v.x) + (mat.m4 * v.y) + (mat.m8  * v.z) + mat.m12,
		.y = (mat.m1 * v.x) + (mat.m5 * v.y) + (mat.m9  * v.z) + mat.m13,
		.z = (mat.m2 * v.x) + (mat.m6 * v.y) + (mat.m10 * v.z) + mat.m14,
	};
    return result;
}

// Transform a vector by quaternion rotation
RMAPI Vector3 Vector3RotateByQuaternion(const Vector3 v, const Quaternion q)
{
	Vector3 result = {
		.x = v.x * ((q.x * q.x) + (q.w * q.w) - (q.y * q.y) - (q.z * q.z)) + (v.y * ((2 * q.x *q.y) - (2 * q.w *q.z))) + (v.z * ((2 * q.x * q.z) + (2 * q.w * q.y))),
		.y = v.x * (( 2 * q.w * q.z) + (2 * q.x * q.y)) + (v.y * ((q.w * q.w) - (q.x * q.x) + (q.y * q.y) - (q.z * q.z))) + (v.z * ((-2 * q.w * q.x) + (2 * q.y * q.z))),
		.z = v.x * ((-2 * q.w * q.y) + (2 * q.x * q.z)) + (v.y * (2 * q.w * q.x + 2 * q.y * q.z)) + (v.z * ((q.w * q.w) - (q.x * q.x) - (q.y * q.y) + (q.z * q.z))),
	};
	return result;
}

// Rotates a vector around an axis
RMAPI Vector3 Vector3RotateByAxisAngle_alt(Vector3 v, Vector3 axis, float angle)
{
    // Using Euler-Rodrigues Formula
    // Ref.: https://en.wikipedia.org/w/index.php?title=Euler%E2%80%93Rodrigues_formula
	axis = Vector3Normalize(axis);
    angle /= 2.0f;
    float a = sinf(angle);
	Vector3 w = {
		.x = axis.x * a,
		.y = axis.y * a,
		.z = axis.z * a,
	};
    a = cosf(angle);

    Vector3 wv = Vector3CrossProduct(w, v);
    Vector3 wwv = Vector3CrossProduct(w, wv);
	wv = Vector3Scale(wv, 2.0f * a);
	wwv = Vector3Scale(wwv, 2.0f);

	Vector3 result = {
		.x = v.x + wv.x + wwv.x,
		.y = v.y + wv.y + wwv.y,
		.z = v.z + wv.z + wwv.z,
	};
    return result;
}

// Rotates a vector around an axis
RMAPI Vector3 Vector3RotateByAxisAngle(Vector3 v, Vector3 axis, float angle)
{
    // Using Euler-Rodrigues Formula
    // Ref.: https://en.wikipedia.org/w/index.php?title=Euler%E2%80%93Rodrigues_formula

    // Vector3Normalize(axis);
    float length = sqrtf(axis.x*axis.x + axis.y*axis.y + axis.z*axis.z);
    if (length == 0.0f) { length = 1.0f; }
    float ilength = 1.0f/length;
    axis.x *= ilength;
    axis.y *= ilength;
    axis.z *= ilength;

    angle /= 2.0f;
    float a = sinf(angle);
    float b = axis.x*a;
    float c = axis.y*a;
    float d = axis.z*a;
    a = cosf(angle);
    Vector3 w = { b, c, d };

    // Vector3CrossProduct(w, v)
    Vector3 wv = { w.y*v.z - w.z*v.y, w.z*v.x - w.x*v.z, w.x*v.y - w.y*v.x };

    // Vector3CrossProduct(w, wv)
    Vector3 wwv = { w.y*wv.z - w.z*wv.y, w.z*wv.x - w.x*wv.z, w.x*wv.y - w.y*wv.x };

    // Vector3Scale(wv, 2*a)
    a *= 2;
    wv.x *= a;
    wv.y *= a;
    wv.z *= a;

    // Vector3Scale(wwv, 2)
    wwv.x *= 2;
    wwv.y *= 2;
    wwv.z *= 2;

	Vector3 result = {
		.x = v.x + wv.x + wwv.x,
		.y = v.y + wv.y + wwv.y,
		.z = v.z + wv.z + wwv.z,
	};
    return result;
}

// Move Vector towards target
RMAPI Vector3 Vector3MoveTowards(Vector3 v, Vector3 target, float maxDistance)
{
    const float dx = target.x - v.x;
    const float dy = target.y - v.y;
    const float dz = target.z - v.z;
    const float value = (dx*dx) + (dy*dy) + (dz*dz);

    if ((value == 0.0f) || ((maxDistance >= 0.0f) && (value <= maxDistance*maxDistance))) {return target;}

    const float dist = sqrtf(value);
	const Vector3 result = {
		.x = v.x + (dx / dist * maxDistance),
		.y = v.y + (dy / dist * maxDistance),
		.z = v.z + (dz / dist * maxDistance),
	};
    return result;
}

// Calculate linear interpolation between two vectors
RMAPI Vector3 Vector3Lerp(const Vector3 v1, const Vector3 v2, const float amount)
{
    Vector3 result = {
		.x = v1.x + (amount * (v2.x - v1.x)),
		.y = v1.y + (amount * (v2.y - v1.y)),
		.z = v1.z + (amount * (v2.z - v1.z)),
	};
    return result;
}

// Calculate cubic hermite interpolation between two vectors and their tangents
// as described in the GLTF 2.0 specification: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#interpolation-cubic
RMAPI Vector3 Vector3CubicHermite(const Vector3 v1, const Vector3 tangent1, const Vector3 v2, const Vector3 tangent2, const float amount)
{
    float amountPow2 = amount * amount;
    float amountPow3 = amount * amount * amount;

    Vector3 result = {
		.x = (2 * amountPow3 - 3 * amountPow2 + 1) * v1.x + (amountPow3 - 2 * amountPow2 + amount) * tangent1.x + (-2 * amountPow3 + 3 * amountPow2) * v2.x + (amountPow3 - amountPow2) * tangent2.x,
		.y = (2 * amountPow3 - 3 * amountPow2 + 1) * v1.y + (amountPow3 - 2 * amountPow2 + amount) * tangent1.y + (-2 * amountPow3 + 3 * amountPow2) * v2.y + (amountPow3 - amountPow2) * tangent2.y,
		.z = (2 * amountPow3 - 3 * amountPow2 + 1) * v1.z + (amountPow3 - 2 * amountPow2 + amount) * tangent1.z + (-2 * amountPow3 + 3 * amountPow2) * v2.z + (amountPow3 - amountPow2) * tangent2.z,
	};
    return result;
}

// Calculate reflected vector to normal
RMAPI Vector3 Vector3Reflect(const Vector3 v, const Vector3 normal)
{
    // I is the original vector
    // N is the normal of the incident plane
    // R = I - (2*N*(DotProduct[I, N]))
    float dotProduct = (v.x*normal.x + v.y*normal.y + v.z*normal.z);
    Vector3 result = {
		.x = v.x - ((2.0f * normal.x) * dotProduct),
		.y = v.y - ((2.0f * normal.y) * dotProduct),
		.z = v.z - ((2.0f * normal.z) * dotProduct),
	};
    return result;
}

// Get min value for each pair of components
RMAPI Vector3 Vector3Min(const Vector3 v1, const Vector3 v2)
{
    Vector3 result = { .x = fminf(v1.x, v2.x), .y = fminf(v1.y, v2.y), .z = fminf(v1.z, v2.z) };
    return result;
}

// Get max value for each pair of components
RMAPI Vector3 Vector3Max(const Vector3 v1, const Vector3 v2)
{
    Vector3 result = { .x = fmaxf(v1.x, v2.x), .y = fmaxf(v1.y, v2.y), .z = fmaxf(v1.z, v2.z) };
    return result;
}

// Compute barycenter coordinates (u, v, w) for point p with respect to triangle (a, b, c)
// NOTE: Assumes P is on the plane of the triangle
RMAPI Vector3 Vector3Barycenter_alt(const Vector3 p, const Vector3 a, const Vector3 b, const Vector3 c)
{
    const Vector3 v0 = Vector3Subtract(b, a);
    const Vector3 v1 = Vector3Subtract(c, a);
    const Vector3 v2 = Vector3Subtract(p, a);
    const float d00  = Vector3DotProduct(v0, v0);
    const float d01  = Vector3DotProduct(v0, v1);
    const float d11  = Vector3DotProduct(v1, v1);
    const float d20  = Vector3DotProduct(v2, v0);
    const float d21  = Vector3DotProduct(v2, v1);

    const float denom = (d00*d11) - (d01*d01);

    Vector3 result = {
		.z = (d00*d21 - d01*d20) / denom,
		.y = (d11*d20 - d01*d21) / denom,
		.x = 1.0f,
	};
	result.x -= (result.z + result.y);
    return result;
}

// Compute barycenter coordinates (u, v, w) for point p with respect to triangle (a, b, c)
// NOTE: Assumes P is on the plane of the triangle
RMAPI Vector3 Vector3Barycenter(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
{
    const Vector3 v0 = { b.x - a.x, b.y - a.y, b.z - a.z };   // Vector3Subtract(b, a)
    const Vector3 v1 = { c.x - a.x, c.y - a.y, c.z - a.z };   // Vector3Subtract(c, a)
    const Vector3 v2 = { p.x - a.x, p.y - a.y, p.z - a.z };   // Vector3Subtract(p, a)
    const float d00 = (v0.x*v0.x + v0.y*v0.y + v0.z*v0.z);    // Vector3DotProduct(v0, v0)
    const float d01 = (v0.x*v1.x + v0.y*v1.y + v0.z*v1.z);    // Vector3DotProduct(v0, v1)
    const float d11 = (v1.x*v1.x + v1.y*v1.y + v1.z*v1.z);    // Vector3DotProduct(v1, v1)
    const float d20 = (v2.x*v0.x + v2.y*v0.y + v2.z*v0.z);    // Vector3DotProduct(v2, v0)
    const float d21 = (v2.x*v1.x + v2.y*v1.y + v2.z*v1.z);    // Vector3DotProduct(v2, v1)

    const float denom = d00*d11 - d01*d01;

    Vector3 result = {
		.y = (d11*d20 - d01*d21) / denom,
		.z = (d00*d21 - d01*d20) / denom,
		.x = 1.0f,
	};
	result.x -= (result.z + result.y);
    return result;
}

// Projects a Vector3 from screen space into object space
// NOTE: We are avoiding calling other raymath functions despite available
RMAPI Vector3 Vector3Unproject(const Vector3 source, const Matrix projection, const Matrix view)
{
    // Calculate unprojected matrix (multiply view matrix by projection matrix) and invert it
    Matrix matViewProj = {      // MatrixMultiply(view, projection);
        view.m0*projection.m0  + view.m1*projection.m4  + view.m2*projection.m8   + view.m3*projection.m12,
        view.m0*projection.m1  + view.m1*projection.m5  + view.m2*projection.m9   + view.m3*projection.m13,
        view.m0*projection.m2  + view.m1*projection.m6  + view.m2*projection.m10  + view.m3*projection.m14,
        view.m0*projection.m3  + view.m1*projection.m7  + view.m2*projection.m11  + view.m3*projection.m15,
        view.m4*projection.m0  + view.m5*projection.m4  + view.m6*projection.m8   + view.m7*projection.m12,
        view.m4*projection.m1  + view.m5*projection.m5  + view.m6*projection.m9   + view.m7*projection.m13,
        view.m4*projection.m2  + view.m5*projection.m6  + view.m6*projection.m10  + view.m7*projection.m14,
        view.m4*projection.m3  + view.m5*projection.m7  + view.m6*projection.m11  + view.m7*projection.m15,
        view.m8*projection.m0  + view.m9*projection.m4  + view.m10*projection.m8  + view.m11*projection.m12,
        view.m8*projection.m1  + view.m9*projection.m5  + view.m10*projection.m9  + view.m11*projection.m13,
        view.m8*projection.m2  + view.m9*projection.m6  + view.m10*projection.m10 + view.m11*projection.m14,
        view.m8*projection.m3  + view.m9*projection.m7  + view.m10*projection.m11 + view.m11*projection.m15,
        view.m12*projection.m0 + view.m13*projection.m4 + view.m14*projection.m8  + view.m15*projection.m12,
        view.m12*projection.m1 + view.m13*projection.m5 + view.m14*projection.m9  + view.m15*projection.m13,
        view.m12*projection.m2 + view.m13*projection.m6 + view.m14*projection.m10 + view.m15*projection.m14,
        view.m12*projection.m3 + view.m13*projection.m7 + view.m14*projection.m11 + view.m15*projection.m15 };

    // Calculate inverted matrix -> MatrixInvert(matViewProj);
    // Cache the matrix values (speed optimization)
	float a[16] = {
		matViewProj.m0,  matViewProj.m1,  matViewProj.m2,  matViewProj.m3,
		matViewProj.m4,  matViewProj.m5,  matViewProj.m6,  matViewProj.m7,
		matViewProj.m8,  matViewProj.m9,  matViewProj.m10, matViewProj.m11,
		matViewProj.m12, matViewProj.m13, matViewProj.m14, matViewProj.m15
	};

	const float b[12] = {
		a[0] * a[5]  - a[1]  * a[4],
		a[0] * a[6]  - a[2]  * a[4],
		a[0] * a[7]  - a[3]  * a[4],
		a[1] * a[6]  - a[2]  * a[5],
		a[1] * a[7]  - a[3]  * a[5],
		a[2] * a[7]  - a[3]  * a[6],
		a[8] * a[13] - a[9]  * a[12],
		a[8] * a[14] - a[10] * a[12],
		a[8] * a[15] - a[11] * a[12],
		a[9] * a[14] - a[10] * a[13],
		a[9] * a[15] - a[11] * a[13],
		a[10]* a[15] - a[11] * a[14],
	};

    // Calculate the invert determinant (inlined to avoid double-caching)
    const float invDet = 1.0f/(b[0]*b[11] - b[1]*b[10] + b[2]*b[9] + b[3]*b[8] - b[4]*b[7] + b[5]*b[6]);

    Matrix matViewProjInv = {
        ( a[5]  * b[11] - a[6]  * b[10] + a[7]  * b[9]) * invDet,
        (-a[1]  * b[11] + a[2]  * b[10] - a[3]  * b[9]) * invDet,
        ( a[13] * b[5]  - a[14] * b[4]  + a[15] * b[3]) * invDet,
        (-a[9]  * b[5]  + a[10] * b[4]  - a[11] * b[3]) * invDet,
        (-a[4]  * b[11] + a[6]  * b[8]  - a[7]  * b[7]) * invDet,
        ( a[0]  * b[11] - a[2]  * b[8]  + a[3]  * b[7]) * invDet,
        (-a[12] * b[5]  + a[14] * b[2]  - a[15] * b[1]) * invDet,
        ( a[8]  * b[5]  - a[10] * b[2]  + a[11] * b[1]) * invDet,
        ( a[4]  * b[10] - a[5]  * b[8]  + a[7]  * b[6]) * invDet,
        (-a[0]  * b[10] + a[1]  * b[8]  - a[3]  * b[6]) * invDet,
        ( a[12] * b[4]  - a[13] * b[2]  + a[15] * b[0]) * invDet,
        (-a[8]  * b[4]  + a[9]  * b[2]  - a[11] * b[0]) * invDet,
        (-a[4]  * b[9]  + a[5]  * b[7]  - a[6]  * b[6]) * invDet,
        ( a[0]  * b[9]  - a[1]  * b[7]  + a[2]  * b[6]) * invDet,
        (-a[12] * b[3]  + a[13] * b[1]  - a[14] * b[0]) * invDet,
        ( a[8]  * b[3]  - a[9]  * b[1]  + a[10] * b[0]) * invDet };

    // Create quaternion from source point
    const Quaternion quat = { source.x, source.y, source.z, 1.0f };

    // Multiply quat point by unprojecte matrix
    const Quaternion qtransformed = {     // QuaternionTransform(quat, matViewProjInv)
        matViewProjInv.m0*quat.x + matViewProjInv.m4*quat.y + matViewProjInv.m8*quat.z + matViewProjInv.m12*quat.w,
        matViewProjInv.m1*quat.x + matViewProjInv.m5*quat.y + matViewProjInv.m9*quat.z + matViewProjInv.m13*quat.w,
        matViewProjInv.m2*quat.x + matViewProjInv.m6*quat.y + matViewProjInv.m10*quat.z + matViewProjInv.m14*quat.w,
        matViewProjInv.m3*quat.x + matViewProjInv.m7*quat.y + matViewProjInv.m11*quat.z + matViewProjInv.m15*quat.w
	};

    // Normalized world points in vectors
    Vector3 result = {
		.x = qtransformed.x / qtransformed.w,
		.y = qtransformed.y / qtransformed.w,
		.z = qtransformed.z / qtransformed.w,
	};
    return result;
}

// Get Vector3 as float array
RMAPI float3 Vector3ToFloatV(const Vector3 v)
{
	const float3 result = {{v.x, v.y, v.z}};
	return result;
}

// Invert the given vector
RMAPI Vector3 Vector3Invert(const Vector3 v)
{
	Vector3 result = {0};
	if (v.x != 0.0f && v.y != 0.0f && v.z != 0.0f) { return result; }
	result.x = 1.0f / v.x;
	result.y = 1.0f / v.y;
	result.z = 1.0f / v.z;
	return result;
}

// Clamp the components of the vector between
// min and max values specified by the given vectors
RMAPI Vector3 Vector3Clamp(const Vector3 v, const Vector3 min, const Vector3 max)
{
	Vector3 result = { fminf(max.x, fmaxf(min.x, v.x)), fminf(max.y, fmaxf(min.y, v.y)), fminf(max.z, fmaxf(min.z, v.z)) };
	return result;
}

// Clamp the magnitude of the vector between two values
RMAPI Vector3 Vector3ClampValue(const Vector3 v, const float min, const float max)
{
    Vector3 result = v;

    float length = (v.x * v.x) + (v.y * v.y) + (v.z * v.z);
    if (length > 0.0f)
    {
        length = sqrtf(length);

        const float scale = (length < min) ? min / length : (length > max) ? max / length : 1.0f;    // By default, 1 as the neutral element.
        result.x = v.x * scale;
        result.y = v.y * scale;
        result.z = v.z * scale;
    }

    return result;
}

// Check whether two given vectors are almost equal
RMAPI int Vector3Equals(const Vector3 p, const Vector3 q)
{
#if !defined(EPSILON)
    #define EPSILON 0.000001f
#endif

    int result = ((fabsf(p.x - q.x)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.x), fabsf(q.x))))) &&
                 ((fabsf(p.y - q.y)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.y), fabsf(q.y))))) &&
                 ((fabsf(p.z - q.z)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.z), fabsf(q.z)))));

    return result;
}

// Compute the direction of a refracted ray
// v: normalized direction of the incoming ray
// n: normalized normal vector of the interface of two optical media
// r: ratio of the refractive index of the medium from where the ray comes
//    to the refractive index of the medium on the other side of the surface
RMAPI Vector3 Vector3Refract(const Vector3 v, const Vector3 n, const float r)
{
    Vector3 result = { 0 };

    const float dot = (v.x * n.x) + (v.y * n.y) + (v.z * n.z);
    float d = 1.0f - (r * r * (1.0f - (dot * dot)));

    if (d >= 0.0f)
    {
        d = sqrtf(d);
        result.x = (r * v.x) - (((r * dot) + d) * n.x);
        result.y = (r * v.y) - (((r * dot) + d) * n.y);
        result.z = (r * v.z) - (((r * dot) + d) * n.z);
    }

    return result;
}


//----------------------------------------------------------------------------------
// Module Functions Definition - Vector4 math
//----------------------------------------------------------------------------------

RMAPI Vector4 Vector4Zero(void)
{
    const Vector4 result = { 0.0f, 0.0f, 0.0f, 0.0f };
    return result;
}

RMAPI Vector4 Vector4One(void)
{
    const Vector4 result = { 1.0f, 1.0f, 1.0f, 1.0f };
    return result;
}

RMAPI Vector4 Vector4Add(const Vector4 v1, const Vector4 v2)
{
    const Vector4 result = {
        .x = v1.x + v2.x,
        .y = v1.y + v2.y,
        .z = v1.z + v2.z,
        .w = v1.w + v2.w
    };
    return result;
}

RMAPI Vector4 Vector4AddValue(const Vector4 v, const float add)
{
    const Vector4 result = {
		.x = v.x + add,
		.y = v.y + add,
		.z = v.z + add,
		.w = v.w + add
    };
    return result;
}

RMAPI Vector4 Vector4Subtract(const Vector4 v1, const Vector4 v2)
{
    const Vector4 result = {
        .x = v1.x - v2.x,
        .y = v1.y - v2.y,
        .z = v1.z - v2.z,
        .w = v1.w - v2.w
    };
    return result;
}

RMAPI Vector4 Vector4SubtractValue(const Vector4 v, const float add)
{
    const Vector4 result = {
        .x = v.x - add,
        .y = v.y - add,
        .z = v.z - add,
        .w = v.w - add
    };
    return result;
}

RMAPI float Vector4Length(const Vector4 v)
{
    const float result = sqrtf((v.x*v.x) + (v.y*v.y) + (v.z*v.z) + (v.w*v.w));
    return result;
}

RMAPI float Vector4LengthSqr(const Vector4 v)
{
    const float result = (v.x*v.x) + (v.y*v.y) + (v.z*v.z) + (v.w*v.w);
    return result;
}

RMAPI float Vector4DotProduct(const Vector4 v1, const Vector4 v2)
{
    const float result = (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w);
    return result;
}

// Calculate distance between two vectors
RMAPI float Vector4Distance(const Vector4 v1, const Vector4 v2)
{
    const float result = sqrtf(
        (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) +
        (v1.z - v2.z)*(v1.z - v2.z) + (v1.w - v2.w)*(v1.w - v2.w));
    return result;
}

// Calculate square distance between two vectors
RMAPI float Vector4DistanceSqr(const Vector4 v1, const Vector4 v2)
{
    const float result =
        (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) +
        (v1.z - v2.z)*(v1.z - v2.z) + (v1.w - v2.w)*(v1.w - v2.w);

    return result;
}

RMAPI Vector4 Vector4Scale(const Vector4 v, const float scale)
{
    const Vector4 result = { v.x*scale, v.y*scale, v.z*scale, v.w*scale };
    return result;
}

// Multiply vector by vector
RMAPI Vector4 Vector4Multiply(const Vector4 v1, const Vector4 v2)
{
    const Vector4 result = { v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v1.w*v2.w };
    return result;
}

// Negate vector
RMAPI Vector4 Vector4Negate(const Vector4 v)
{
    const Vector4 result = { -v.x, -v.y, -v.z, -v.w };
    return result;
}

// Divide vector by vector
RMAPI Vector4 Vector4Divide(const Vector4 v1, const Vector4 v2)
{
    const Vector4 result = { v1.x/v2.x, v1.y/v2.y, v1.z/v2.z, v1.w/v2.w };
    return result;
}

// Normalize provided vector
RMAPI Vector4 Vector4Normalize(const Vector4 v)
{
	Vector4 result = {0};
    const float length = sqrtf((v.x*v.x) + (v.y*v.y) + (v.z*v.z) + (v.w*v.w));
	if (length > 0.0f) {
		result.x = v.x / length;
		result.y = v.y / length;
		result.z = v.z / length;
		result.w = v.w / length;
	}
	return result;
}

// Get min value for each pair of components
RMAPI Vector4 Vector4Min(const Vector4 v1, const Vector4 v2)
{
    const Vector4 result = { fminf(v1.x, v2.x), fminf(v1.y, v2.y), fminf(v1.z, v2.z), fminf(v1.w, v2.w) };
    return result;
}

// Get max value for each pair of components
RMAPI Vector4 Vector4Max(const Vector4 v1, const Vector4 v2)
{
    const Vector4 result = { fmaxf(v1.x, v2.x), fmaxf(v1.y, v2.y), fmaxf(v1.z, v2.z), fmaxf(v1.w, v2.w) };
    return result;
}

// Calculate linear interpolation between two vectors
RMAPI Vector4 Vector4Lerp(const Vector4 v1, const Vector4 v2, const float amount)
{
    const Vector4 result = {
		v1.x + amount*(v2.x - v1.x),
		v1.y + amount*(v2.y - v1.y),
		v1.z + amount*(v2.z - v1.z),
		v1.w + amount*(v2.w - v1.w)
	};
    return result;
}

// Move Vector towards target
RMAPI Vector4 Vector4MoveTowards(const Vector4 v, const Vector4 target, const float maxDistance)
{
    const float dx = target.x - v.x;
    const float dy = target.y - v.y;
    const float dz = target.z - v.z;
    const float dw = target.w - v.w;
    const float value = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw);

    if ((value == 0) || ((maxDistance >= 0) && (value <= maxDistance*maxDistance))) { return target; }

    const float dist = sqrtf(value);

    const Vector4 result = {
		.x = v.x + dx/dist*maxDistance,
		.y = v.y + dy/dist*maxDistance,
		.z = v.z + dz/dist*maxDistance,
		.w = v.w + dw/dist*maxDistance
	};
    return result;
}

// Invert the given vector
RMAPI Vector4 Vector4Invert(const Vector4 v)
{
    const Vector4 result = { 1.0f / v.x, 1.0f / v.y, 1.0f / v.z, 1.0f / v.w };
    return result;
}

// Check whether two given vectors are almost equal
RMAPI int Vector4Equals(const Vector4 p, const Vector4 q)
{
#if !defined(EPSILON)
    #define EPSILON 0.000001f
#endif

    return ((fabsf(p.x - q.x)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.x), fabsf(q.x))))) &&
           ((fabsf(p.y - q.y)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.y), fabsf(q.y))))) &&
           ((fabsf(p.z - q.z)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.z), fabsf(q.z))))) &&
           ((fabsf(p.w - q.w)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.w), fabsf(q.w)))));
}


//----------------------------------------------------------------------------------
// Module Functions Definition - Matrix math
//----------------------------------------------------------------------------------

// Compute matrix determinant
RMAPI float MatrixDeterminant(const Matrix mat)
{
    // Cache the matrix values (speed optimization)
	const float a[16] = {
		mat.m0,  mat.m1,  mat.m2,  mat.m3,
		mat.m4,  mat.m5,  mat.m6,  mat.m7,
		mat.m8,  mat.m9,  mat.m10, mat.m11,
		mat.m12, mat.m13, mat.m14, mat.m15,
	};

	return (a[12] * a[9] * a[6]  * a[3] ) - (a[8] * a[13] * a[6]  * a[3] ) - (a[12] * a[5] * a[10] * a[3] ) + (a[4] * a[13] *a[10]*a[3] ) +
	       (a[8]  * a[5] * a[14] * a[3] ) - (a[4] * a[9]  * a[14] * a[3] ) - (a[12] * a[9] * a[2]  * a[7] ) + (a[8] * a[13] *a[2] *a[7] ) +
	       (a[12] * a[1] * a[10] * a[7] ) - (a[0] * a[13] * a[10] * a[7] ) - (a[8]  * a[1] * a[14] * a[7] ) + (a[0] * a[9]  *a[14]*a[7] ) +
	       (a[12] * a[5] * a[2]  * a[12]) - (a[4] * a[13] * a[2]  * a[12]) - (a[12] * a[1] * a[6]  * a[12]) + (a[0] * a[13] *a[6] *a[12]) +
	       (a[4]  * a[1] * a[14] * a[12]) - (a[0] * a[5]  * a[14] * a[12]) - (a[8]  * a[5] * a[2]  * a[15]) + (a[4] * a[9]  *a[2] *a[15]) +
	       (a[8]  * a[1] * a[6]  * a[15]) - (a[0] * a[9]  * a[6]  * a[15]) - (a[4]  * a[1] * a[10] * a[15]) + (a[0] * a[5]  *a[10]*a[15]);
}

// Get the trace of the matrix (sum of the values along the diagonal)
RMAPI float MatrixTrace(const Matrix mat)
{
    return (mat.m0 + mat.m5 + mat.m10 + mat.m15);
}

// Transposes provided matrix
RMAPI Matrix MatrixTranspose(Matrix mat)
{
	const Matrix result = {
		.m0  = mat.m0,
		.m1  = mat.m4,
		.m2  = mat.m8,
		.m3  = mat.m12,
		.m4  = mat.m1,
		.m5  = mat.m5,
		.m6  = mat.m9,
		.m7  = mat.m13,
		.m8  = mat.m2,
		.m9  = mat.m6,
		.m10 = mat.m10,
		.m11 = mat.m14,
		.m12 = mat.m3,
		.m13 = mat.m7,
		.m14 = mat.m11,
		.m15 = mat.m15,
	};
    return result;
}

// Invert provided matrix
RMAPI Matrix MatrixInvert(const Matrix mat)
{
    // Cache the matrix values (speed optimization)
	const float a[16] = {
		mat.m0,  mat.m1,  mat.m2,  mat.m3,
		mat.m4,  mat.m5,  mat.m6,  mat.m7,
		mat.m8,  mat.m9,  mat.m10, mat.m11,
		mat.m12, mat.m13, mat.m14, mat.m15,
	};

	const float b[12] = {
		(a[0] * a[5] ) - (a[1]  * a[4] ),
		(a[0] * a[6] ) - (a[2]  * a[4] ),
		(a[0] * a[7] ) - (a[3]  * a[4] ),
		(a[1] * a[6] ) - (a[2]  * a[5] ),
		(a[1] * a[7] ) - (a[3]  * a[5] ),
		(a[2] * a[7] ) - (a[3]  * a[6] ),
		(a[8] * a[13]) - (a[9]  * a[12]),
		(a[8] * a[14]) - (a[10] * a[12]),
		(a[8] * a[15]) - (a[11] * a[12]),
		(a[9] * a[14]) - (a[10] * a[13]),
		(a[9] * a[15]) - (a[11] * a[13]),
		(a[10]* a[15]) - (a[11] * a[14]),
	};

    // Calculate the invert determinant (inlined to avoid double-caching)
    const float invDet = 1.0f/(b[0]*b[11] - b[1]*b[10] + b[2]*b[9] + b[3]*b[8] - b[4]*b[7] + b[5]*b[6]);

	const Matrix result = {
		.m0  = (( a[5]  * b[11]) - (a[6]  * b[10]) + (a[7]  * b[9])) * invDet,
		.m1  = ((-a[1]  * b[11]) + (a[2]  * b[10]) - (a[3]  * b[9])) * invDet,
		.m2  = (( a[13] * b[5] ) - (a[14] * b[4] ) + (a[15] * b[3])) * invDet,
		.m3  = ((-a[9]  * b[5] ) + (a[10] * b[4] ) - (a[11] * b[3])) * invDet,
		.m4  = ((-a[4]  * b[11]) + (a[6]  * b[8] ) - (a[7]  * b[7])) * invDet,
		.m5  = (( a[0]  * b[11]) - (a[2]  * b[8] ) + (a[3]  * b[7])) * invDet,
		.m6  = ((-a[12] * b[5] ) + (a[14] * b[2] ) - (a[15] * b[1])) * invDet,
		.m7  = (( a[8]  * b[5] ) - (a[10] * b[2] ) + (a[11] * b[1])) * invDet,
		.m8  = (( a[4]  * b[10]) - (a[5]  * b[8] ) + (a[7]  * b[6])) * invDet,
		.m9  = ((-a[0]  * b[10]) + (a[1]  * b[8] ) - (a[3]  * b[6])) * invDet,
		.m10 = (( a[12] * b[4] ) - (a[13] * b[2] ) + (a[15] * b[0])) * invDet,
		.m11 = ((-a[8]  * b[4] ) + (a[9]  * b[2] ) - (a[11] * b[0])) * invDet,
		.m12 = ((-a[4]  * b[9] ) + (a[5]  * b[7] ) - (a[6]  * b[6])) * invDet,
		.m13 = (( a[0]  * b[9] ) - (a[1]  * b[7] ) + (a[2]  * b[6])) * invDet,
		.m14 = ((-a[12] * b[3] ) + (a[13] * b[1] ) - (a[14] * b[0])) * invDet,
		.m15 = (( a[8]  * b[3] ) - (a[9]  * b[1] ) + (a[10] * b[0])) * invDet,
	};
    return result;
}

// Get identity matrix
RMAPI Matrix MatrixIdentity(void)
{
    const Matrix result = { 1.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 1.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 1.0f };
    return result;
}

// Add two matrices
RMAPI Matrix MatrixAdd(const Matrix left, const Matrix right)
{
    const Matrix result = {
		.m0  = left.m0  + right.m0,
		.m1  = left.m1  + right.m1,
		.m2  = left.m2  + right.m2,
		.m3  = left.m3  + right.m3,
		.m4  = left.m4  + right.m4,
		.m5  = left.m5  + right.m5,
		.m6  = left.m6  + right.m6,
		.m7  = left.m7  + right.m7,
		.m8  = left.m8  + right.m8,
		.m9  = left.m9  + right.m9,
		.m10 = left.m10 + right.m10,
		.m11 = left.m11 + right.m11,
		.m12 = left.m12 + right.m12,
		.m13 = left.m13 + right.m13,
		.m14 = left.m14 + right.m14,
		.m15 = left.m15 + right.m15,
	};
    return result;
}

// Subtract two matrices (left - right)
RMAPI Matrix MatrixSubtract(const Matrix left, const Matrix right)
{
    const Matrix result = {
		.m0  = left.m0  + right.m0,
		.m1  = left.m1  + right.m1,
		.m2  = left.m2  + right.m2,
		.m3  = left.m3  + right.m3,
		.m4  = left.m4  + right.m4,
		.m5  = left.m5  + right.m5,
		.m6  = left.m6  + right.m6,
		.m7  = left.m7  + right.m7,
		.m8  = left.m8  + right.m8,
		.m9  = left.m9  + right.m9,
		.m10 = left.m10 + right.m10,
		.m11 = left.m11 + right.m11,
		.m12 = left.m12 + right.m12,
		.m13 = left.m13 + right.m13,
		.m14 = left.m14 + right.m14,
		.m15 = left.m15 + right.m15,
	};
    return result;
}

// Get two matrix multiplication
// NOTE: When multiplying matrices... the order matters!
RMAPI Matrix MatrixMultiply(const Matrix left, const Matrix right)
{
    const Matrix result = {
		.m0  = (left.m0  * right.m0) + (left.m1  * right.m4) + (left.m2  * right.m8 ) + (left.m3  * right.m12),
		.m1  = (left.m0  * right.m1) + (left.m1  * right.m5) + (left.m2  * right.m9 ) + (left.m3  * right.m13),
		.m2  = (left.m0  * right.m2) + (left.m1  * right.m6) + (left.m2  * right.m10) + (left.m3  * right.m14),
		.m3  = (left.m0  * right.m3) + (left.m1  * right.m7) + (left.m2  * right.m11) + (left.m3  * right.m15),
		.m4  = (left.m4  * right.m0) + (left.m5  * right.m4) + (left.m6  * right.m8 ) + (left.m7  * right.m12),
		.m5  = (left.m4  * right.m1) + (left.m5  * right.m5) + (left.m6  * right.m9 ) + (left.m7  * right.m13),
		.m6  = (left.m4  * right.m2) + (left.m5  * right.m6) + (left.m6  * right.m10) + (left.m7  * right.m14),
		.m7  = (left.m4  * right.m3) + (left.m5  * right.m7) + (left.m6  * right.m11) + (left.m7  * right.m15),
		.m8  = (left.m8  * right.m0) + (left.m9  * right.m4) + (left.m10 * right.m8 ) + (left.m11 * right.m12),
		.m9  = (left.m8  * right.m1) + (left.m9  * right.m5) + (left.m10 * right.m9 ) + (left.m11 * right.m13),
		.m10 = (left.m8  * right.m2) + (left.m9  * right.m6) + (left.m10 * right.m10) + (left.m11 * right.m14),
		.m11 = (left.m8  * right.m3) + (left.m9  * right.m7) + (left.m10 * right.m11) + (left.m11 * right.m15),
		.m12 = (left.m12 * right.m0) + (left.m13 * right.m4) + (left.m14 * right.m8 ) + (left.m15 * right.m12),
		.m13 = (left.m12 * right.m1) + (left.m13 * right.m5) + (left.m14 * right.m9 ) + (left.m15 * right.m13),
		.m14 = (left.m12 * right.m2) + (left.m13 * right.m6) + (left.m14 * right.m10) + (left.m15 * right.m14),
		.m15 = (left.m12 * right.m3) + (left.m13 * right.m7) + (left.m14 * right.m11) + (left.m15 * right.m15),
	};
    return result;
}

// Get translation matrix
RMAPI Matrix MatrixTranslate(const float x, const float y, const float z)
{
    const Matrix result = { 1.0f, 0.0f, 0.0f, x,
                            0.0f, 1.0f, 0.0f, y,
                            0.0f, 0.0f, 1.0f, z,
                            0.0f, 0.0f, 0.0f, 1.0f };
	return result;
}

// Create rotation matrix from axis and angle
// NOTE: Angle should be provided in radians
RMAPI Matrix MatrixRotate(Vector3 axis, const float angle)
{
    const float length = sqrtf((axis.x * axis.x) + (axis.y * axis.y) + (axis.z * axis.z));
	if (length != 0.0f) {
		axis.x /= length;
		axis.y /= length;
		axis.z /= length;
	}

    const float sinres = sinf(angle);
    const float cosres = cosf(angle);
    const float t = 1.0f - cosres;

    const Matrix result = {
		.m0 = (axis.x * axis.x * t) + cosres,
		.m1 = (axis.y * axis.x * t) + (axis.z * sinres),
		.m2 = (axis.z * axis.x * t) - (axis.y * sinres),
		.m3 = 0.0f,

		.m4 = (axis.x * axis.y * t) - (axis.z * sinres),
		.m5 = (axis.y * axis.y * t) + cosres,
		.m6 = (axis.z * axis.y * t) + (axis.x * sinres),
		.m7 = 0.0f,

		.m8  = (axis.x * axis.z * t) + (axis.y * sinres),
		.m9  = (axis.y * axis.z * t) - (axis.x * sinres),
		.m10 = (axis.z * axis.z * t) + cosres,
		.m11 = 0.0f,

		.m12 = 0.0f,
		.m13 = 0.0f,
		.m14 = 0.0f,
		.m15 = 1.0f,
};

    return result;
}

// Get x-rotation matrix
// NOTE: Angle must be provided in radians
RMAPI Matrix MatrixRotateX(const float angle)
{
    const float cosres = cosf(angle);
    const float sinres = sinf(angle);

    Matrix result = { 1.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 1.0f }; // MatrixIdentity()

    result.m5  = cosres;
    result.m6  = sinres;
    result.m9  = -sinres;
    result.m10 = cosres;

    return result;
}

// Get y-rotation matrix
// NOTE: Angle must be provided in radians
RMAPI Matrix MatrixRotateY(const float angle)
{
    const float cosres = cosf(angle);
    const float sinres = sinf(angle);

    Matrix result = { 1.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 1.0f }; // MatrixIdentity()

    result.m0  = cosres;
    result.m2  = -sinres;
    result.m8  = sinres;
    result.m10 = cosres;

    return result;
}

// Get z-rotation matrix
// NOTE: Angle must be provided in radians
RMAPI Matrix MatrixRotateZ(const float angle)
{
    const float cosres = cosf(angle);
    const float sinres = sinf(angle);

    Matrix result = { 1.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 1.0f }; // MatrixIdentity()

    result.m0 = cosres;
    result.m1 = sinres;
    result.m4 = -sinres;
    result.m5 = cosres;

    return result;
}


// Get xyz-rotation matrix
// NOTE: Angle must be provided in radians
RMAPI Matrix MatrixRotateXYZ(const Vector3 angle)
{
    const float cosz = cosf(-angle.z);
    const float sinz = sinf(-angle.z);
    const float cosy = cosf(-angle.y);
    const float siny = sinf(-angle.y);
    const float cosx = cosf(-angle.x);
    const float sinx = sinf(-angle.x);

    Matrix result = { 1.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 1.0f }; // MatrixIdentity()

    result.m0  = (cosz * cosy);
    result.m1  = (cosz * siny * sinx) - (sinz * cosx);
    result.m2  = (cosz * siny * cosx) + (sinz * sinx);

    result.m4  = (sinz * cosy);
    result.m5  = (sinz * siny * sinx) + (cosz * cosx);
    result.m6  = (sinz * siny * cosx) - (cosz * sinx);

    result.m8  = (-siny);
    result.m9  = (cosy * sinx);
    result.m10 = (cosy * cosx);

    return result;
}

// Get zyx-rotation matrix
// NOTE: Angle must be provided in radians
RMAPI Matrix MatrixRotateZYX(const Vector3 angle)
{
    const float cz = cosf(angle.z);
    const float sz = sinf(angle.z);
    const float cy = cosf(angle.y);
    const float sy = sinf(angle.y);
    const float cx = cosf(angle.x);
    const float sx = sinf(angle.x);

    const Matrix result = {
		.m0  = (cz * cy),
		.m4  = (cz * sy * sx) - (cx * sz),
		.m8  = (sz * sx) + (cz * cx * sy),
		.m12 = 0.0f,

		.m1  = (cy * sz),
		.m5  = (cz * cx) + (sz * sy * sx),
		.m9  = (cx * sz * sy) - (cz * sx),
		.m13 = 0.0f,

		.m2  = -sy,
		.m6  = (cy * sx),
		.m10 = (cy * cx),
		.m14 = 0.0f,

		.m3  = 0.0f,
		.m7  = 0.0f,
		.m11 = 0.0f,
		.m15 = 1.0f,
	};
    return result;
}

// Get scaling matrix
RMAPI Matrix MatrixScale(const float x, const float y, const float z)
{
    const Matrix result = { x, 0.0f, 0.0f, 0.0f,
                            0.0f, y, 0.0f, 0.0f,
                            0.0f, 0.0f, z, 0.0f,
                            0.0f, 0.0f, 0.0f, 1.0f };
    return result;
}

// Get perspective projection matrix
RMAPI Matrix MatrixFrustum(const double left, const double right, const double bottom, const double top, const double nearPlane, const double farPlane)
{
    const float rl = (float)(right - left);
    const float tb = (float)(top - bottom);
    const float fn = (float)(farPlane - nearPlane);

    const Matrix result = {
		.m0  = ((float)nearPlane * 2.0f) / rl,
		.m1  = 0.0f,
		.m2  = 0.0f,
		.m3  = 0.0f,

		.m4  = 0.0f,
		.m5  = ((float)nearPlane * 2.0f) / tb,
		.m6  = 0.0f,
		.m7  = 0.0f,

		.m8  = ((float)right + (float)left)   / rl,
		.m9  = ((float)top   + (float)bottom) / tb,
		.m10 = -((float)farPlane  + (float)nearPlane) / fn,
		.m11 = -1.0f,

		.m12 = 0.0f,
		.m13 = 0.0f,
		.m14 = -((float)farPlane * (float)nearPlane * 2.0f) / fn,
		.m15 = 0.0f,
	};
    return result;
}

// Get perspective projection matrix
// NOTE: Fovy angle must be provided in radians
RMAPI Matrix MatrixPerspective(const double fovY, const double aspect, const double nearPlane, const double farPlane)
{
    const double top    =  nearPlane * tan(fovY*0.5);
    const double bottom = -top;
    const double right  =  top * aspect;
    const double left   = -right;

    // MatrixFrustum(-right, right, -top, top, near, far);
    const float rl = (float)(right - left);
    const float tb = (float)(top - bottom);
    const float fn = (float)(farPlane - nearPlane);

    const Matrix result = {
		.m0  =  ((float)nearPlane * 2.0f) / rl,
		.m5  =  ((float)nearPlane * 2.0f) / tb,
		.m8  =  ((float)right + (float)left) / rl,
		.m9  =  ((float)top   + (float)bottom) / tb,
		.m10 = -((float)farPlane + (float)nearPlane) / fn,
		.m11 = -1.0f,
		.m14 = -((float)farPlane * (float)nearPlane * 2.0f) / fn,
	};
    return result;
}

// Get orthographic projection matrix
RMAPI Matrix MatrixOrtho(const double left, const double right, const double bottom, const double top, const double nearPlane, const double farPlane)
{

    float rl = (float)(right - left);
    float tb = (float)(top - bottom);
    float fn = (float)(farPlane - nearPlane);

    const Matrix result = {
		.m0  =  2.0f / rl,
		.m1  =  0.0f,
		.m2  =  0.0f,
		.m3  =  0.0f,
		.m4  =  0.0f,
		.m5  =  2.0f / tb,
		.m6  =  0.0f,
		.m7  =  0.0f,
		.m8  =  0.0f,
		.m9  =  0.0f,
		.m10 = -2.0f / fn,
		.m11 =  0.0f,
		.m12 = -((float)left     + (float)right)     / rl,
		.m13 = -((float)top      + (float)bottom)    / tb,
		.m14 = -((float)farPlane + (float)nearPlane) / fn,
		.m15 =  1.0f,
	};
    return result;
}

// Get camera look-at matrix (view matrix)
RMAPI Matrix MatrixLookAt(const Vector3 eye, const Vector3 target, const Vector3 up)
{
    // Vector3Subtract(eye, target)
    Vector3 vz = { eye.x - target.x, eye.y - target.y, eye.z - target.z };

    // Vector3Normalize(vz)
    Vector3 v = vz;
    float length = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (length != 0.0f) {
		vz.x /= length;
		vz.y /= length;
		vz.z /= length;
	}

    // Vector3CrossProduct(up, vz)
    Vector3 vx = { up.y*vz.z - up.z*vz.y, up.z*vz.x - up.x*vz.z, up.x*vz.y - up.y*vz.x };

    // Vector3Normalize(x)
    v = vx;
    length = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (length != 0.0f) {
		vx.x /= length;
		vx.y /= length;
		vx.z /= length;
	}

    // Vector3CrossProduct(vz, vx)
    const Vector3 vy = { vz.y*vx.z - vz.z*vx.y, vz.z*vx.x - vz.x*vx.z, vz.x*vx.y - vz.y*vx.x };

    const Matrix result = {
		.m0  = vx.x,
		.m1  = vy.x,
		.m2  = vz.x,
		.m3  = 0.0f,
		.m4  = vx.y,
		.m5  = vy.y,
		.m6  = vz.y,
		.m7  = 0.0f,
		.m8  = vx.z,
		.m9  = vy.z,
		.m10 = vz.z,
		.m11 = 0.0f,
		.m12 = -((vx.x * eye.x) + (vx.y * eye.y) + (vx.z * eye.z)),   // Vector3DotProduct(vx, eye)
		.m13 = -((vy.x * eye.x) + (vy.y * eye.y) + (vy.z * eye.z)),   // Vector3DotProduct(vy, eye)
		.m14 = -((vz.x * eye.x) + (vz.y * eye.y) + (vz.z * eye.z)),   // Vector3DotProduct(vz, eye)
		.m15 = 1.0f,
	};

    return result;
}

// Get float array of matrix data
RMAPI float16 MatrixToFloatV(const Matrix mat)
{
    const float16 result = {
		.v[0] = mat.m0,
		.v[1] = mat.m1,
		.v[2] = mat.m2,
		.v[3] = mat.m3,
		.v[4] = mat.m4,
		.v[5] = mat.m5,
		.v[6] = mat.m6,
		.v[7] = mat.m7,
		.v[8] = mat.m8,
		.v[9] = mat.m9,
		.v[10] = mat.m10,
		.v[11] = mat.m11,
		.v[12] = mat.m12,
		.v[13] = mat.m13,
		.v[14] = mat.m14,
		.v[15] = mat.m15,
	};
    return result;
}

//----------------------------------------------------------------------------------
// Module Functions Definition - Quaternion math
//----------------------------------------------------------------------------------

RMAPI Quaternion QuaternionZero(void) {
	const Quaternion result = {0};
	return result;
}

RMAPI Quaternion QuaternionOne(void) {
	const Quaternion result = {1.0f, 1.0f, 1.0f, 1.0f};
	return result;
}

// Add two quaternions
RMAPI Quaternion QuaternionAdd(const Quaternion q1, const Quaternion q2)
{
    const Quaternion result = {q1.x + q2.x, q1.y + q2.y, q1.z + q2.z, q1.w + q2.w};
    return result;
}

// Add quaternion and float value
RMAPI Quaternion QuaternionAddValue(const Quaternion q, const float add)
{
	const Quaternion result = {q.x + add, q.y + add, q.z + add, q.w + add};
    return result;
}

// Subtract two quaternions
RMAPI Quaternion QuaternionSubtract(const Quaternion q1, const Quaternion q2)
{
    const Quaternion result = {q1.x - q2.x, q1.y - q2.y, q1.z - q2.z, q1.w - q2.w};
    return result;
}

// Subtract quaternion and float value
RMAPI Quaternion QuaternionSubtractValue(const Quaternion q, const float sub)
{
    const Quaternion result = {q.x - sub, q.y - sub, q.z - sub, q.w - sub};
    return result;
}

// Get identity quaternion
RMAPI Quaternion QuaternionIdentity(void)
{
    const Quaternion result = { 0.0f, 0.0f, 0.0f, 1.0f };
    return result;
}

// Computes the length of a quaternion
RMAPI float QuaternionLength(const Quaternion q)
{
    const float result = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    return result;
}

// Normalize provided quaternion
RMAPI Quaternion QuaternionNormalize(const Quaternion q)
{
	Quaternion result = {0};
    const float length = sqrtf((q.x * q.x) + q.y*q.y + q.z*q.z + q.w*q.w);
	if (length > 0.0f) {
		result.x = q.x / length;
		result.y = q.y / length;
		result.z = q.z / length;
		result.w = q.w / length;
	}
	return result;
}

// Invert provided quaternion
RMAPI Quaternion QuaternionInvert(Quaternion q)
{
    Quaternion result = q;
    const float lengthSq = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    if (lengthSq != 0.0f)
    {
        const float invLength = 1.0f/lengthSq;
        result.x *= -invLength;
        result.y *= -invLength;
        result.z *= -invLength;
        result.w *= invLength;
    }
    return result;
}

// Calculate two quaternion multiplication
RMAPI Quaternion QuaternionMultiply(const Quaternion q1, const Quaternion q2)
{
    const Quaternion result = {
		.x = (q1.x * q2.w) + (q1.w * q2.x) + (q1.y * q2.z) - (q1.z * q2.y),
		.y = (q1.y * q2.w) + (q1.w * q2.y) + (q1.z * q2.x) - (q1.x * q2.z),
		.z = (q1.z * q2.w) + (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x),
		.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z),
	};
    return result;
}

// Scale quaternion by float value
RMAPI Quaternion QuaternionScale(const Quaternion q, const float mul)
{
    const Quaternion result = {
		.x = q.x * mul,
		.y = q.y * mul,
		.z = q.z * mul,
		.w = q.w * mul,
	};
    return result;
}

// Divide two quaternions
RMAPI Quaternion QuaternionDivide(const Quaternion q1, const Quaternion q2)
{
    const Quaternion result = { q1.x/q2.x, q1.y/q2.y, q1.z/q2.z, q1.w/q2.w };
    return result;
}

// Calculate linear interpolation between two quaternions
RMAPI Quaternion QuaternionLerp(const Quaternion q1, const Quaternion q2, const float amount)
{
    Quaternion result = {
		.x = q1.x + (amount * (q2.x - q1.x)),
		.y = q1.y + (amount * (q2.y - q1.y)),
		.z = q1.z + (amount * (q2.z - q1.z)),
		.w = q1.w + (amount * (q2.w - q1.w)),
	};
    return result;
}

// Calculate slerp-optimized interpolation between two quaternions
RMAPI Quaternion QuaternionNlerp(const Quaternion q1, const Quaternion q2, const float amount)
{
    // QuaternionLerp(q1, q2, amount)
    const Quaternion q = {
		.x = q1.x + (amount * (q2.x - q1.x)),
		.y = q1.y + (amount * (q2.y - q1.y)),
		.z = q1.z + (amount * (q2.z - q1.z)),
		.w = q1.w + (amount * (q2.w - q1.w)),
	};

    // QuaternionNormalize(q);
    float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
	length = (length != 0.0f) ? length : 1.0f; // Here we could remove this line if we had access to the Elvis Operator :?
	// float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w) :? 1.0f; // This would check and assign length on true.

    const Quaternion result = {
		.x = q.x / length,
		.y = q.y / length,
		.z = q.z / length,
		.w = q.w / length,
	};
    return result;
}

// Calculates spherical linear interpolation between two quaternions
RMAPI Quaternion QuaternionSlerp(const Quaternion q1, Quaternion q2, const float amount)
{
#if !defined(EPSILON)
    #define EPSILON 0.000001f
#endif

    float cosHalfTheta = (q1.x * q2.x) + (q1.y * q2.y) + (q1.z * q2.z) + (q1.w * q2.w);

    if (cosHalfTheta < 0.0f)
    {
        q2.x = -q2.x;
		q2.y = -q2.y;
		q2.z = -q2.z;
		q2.w = -q2.w;
        cosHalfTheta = -cosHalfTheta;
    }

    Quaternion result = { 0 };
    if (fabsf(cosHalfTheta) >= 1.0f) { result = q1; }
    else if (cosHalfTheta > 0.95f) { result = QuaternionNlerp(q1, q2, amount); }
    else
    {
        const float halfTheta = acosf(cosHalfTheta);
        const float sinHalfTheta = sqrtf(1.0f - cosHalfTheta*cosHalfTheta);

        if (fabsf(sinHalfTheta) < EPSILON)
        {
            result.x = (q1.x*0.5f + q2.x*0.5f);
            result.y = (q1.y*0.5f + q2.y*0.5f);
            result.z = (q1.z*0.5f + q2.z*0.5f);
            result.w = (q1.w*0.5f + q2.w*0.5f);
        }
        else
        {
            const float ratioA = sinf((1 - amount)*halfTheta)/sinHalfTheta;
            const float ratioB = sinf(amount*halfTheta)/sinHalfTheta;

            result.x = (q1.x*ratioA + q2.x*ratioB);
            result.y = (q1.y*ratioA + q2.y*ratioB);
            result.z = (q1.z*ratioA + q2.z*ratioB);
            result.w = (q1.w*ratioA + q2.w*ratioB);
        }
    }

    return result;
}

// Calculate quaternion cubic spline interpolation using Cubic Hermite Spline algorithm
// as described in the GLTF 2.0 specification: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#interpolation-cubic
RMAPI Quaternion QuaternionCubicHermiteSpline(const Quaternion q1, const Quaternion outTangent1, const Quaternion q2, const Quaternion inTangent2, const float t)
{
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float h00 = (2 * t3) - (3 * t2) + 1;
    const float h10 = t3 - (2 * t2) + t;
    const float h01 = (-2 * t3) + (3 * t2);
    const float h11 = t3 - t2;

    const Quaternion p0 = QuaternionScale(q1, h00);
    const Quaternion m0 = QuaternionScale(outTangent1, h10);
    const Quaternion p1 = QuaternionScale(q2, h01);
    const Quaternion m1 = QuaternionScale(inTangent2, h11);

    Quaternion result = { 0 };
    result = QuaternionAdd(p0, m0);
    result = QuaternionAdd(result, p1);
    result = QuaternionAdd(result, m1);
    result = QuaternionNormalize(result);
    return result;
}

// Calculate quaternion based on the rotation from one vector to another
RMAPI Quaternion QuaternionFromVector3ToVector3(const Vector3 from, const Vector3 to)
{

    const float cos2Theta = (from.x*to.x + from.y*to.y + from.z*to.z);    // Vector3DotProduct(from, to)
    const Vector3 cross = { from.y*to.z - from.z*to.y, from.z*to.x - from.x*to.z, from.x*to.y - from.y*to.x }; // Vector3CrossProduct(from, to)

    const Quaternion q = {
		.x = cross.x,
		.y = cross.y,
		.z = cross.z,
		.w = 1.0f + cos2Theta,
	};

    // QuaternionNormalize(q);
    // NOTE: Normalize to essentially nlerp the original and identity to 0.5
    float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
	length = (length != 0.0f) ? length : 1.0f;

    const Quaternion result = {
		.x = q.x / length,
		.y = q.y / length,
		.z = q.z / length,
		.w = q.w / length,
	};
    return result;
}

// Get a quaternion for a given rotation matrix
RMAPI Quaternion QuaternionFromMatrix(const Matrix mat)
{
    const float fourWSquaredMinus1 = mat.m0  + mat.m5 + mat.m10;
    const float fourXSquaredMinus1 = mat.m0  - mat.m5 - mat.m10;
    const float fourYSquaredMinus1 = mat.m5  - mat.m0 - mat.m10;
    const float fourZSquaredMinus1 = mat.m10 - mat.m0 - mat.m5;

    int biggestIndex = 0;
    float fourBiggestSquaredMinus1 = fourWSquaredMinus1;
    if (fourXSquaredMinus1 > fourBiggestSquaredMinus1)
    {
        fourBiggestSquaredMinus1 = fourXSquaredMinus1;
        biggestIndex = 1;
    }

    if (fourYSquaredMinus1 > fourBiggestSquaredMinus1)
    {
        fourBiggestSquaredMinus1 = fourYSquaredMinus1;
        biggestIndex = 2;
    }

    if (fourZSquaredMinus1 > fourBiggestSquaredMinus1)
    {
        fourBiggestSquaredMinus1 = fourZSquaredMinus1;
        biggestIndex = 3;
    }

    const float biggestVal = sqrtf(fourBiggestSquaredMinus1 + 1.0f) * 0.5f;
    const float mult = (0.25f / biggestVal); // 1/(4 * biggestVal);

    Quaternion result = { 0 };
    switch (biggestIndex)
    {
        case 0:
            result.w = biggestVal;
            result.x = (mat.m6 - mat.m9)*mult;
            result.y = (mat.m8 - mat.m2)*mult;
            result.z = (mat.m1 - mat.m4)*mult;
            break;
        case 1:
            result.x = biggestVal;
            result.w = (mat.m6 - mat.m9)*mult;
            result.y = (mat.m1 + mat.m4)*mult;
            result.z = (mat.m8 + mat.m2)*mult;
            break;
        case 2:
            result.y = biggestVal;
            result.w = (mat.m8 - mat.m2)*mult;
            result.x = (mat.m1 + mat.m4)*mult;
            result.z = (mat.m6 + mat.m9)*mult;
            break;
        case 3:
            result.z = biggestVal;
            result.w = (mat.m1 - mat.m4)*mult;
            result.x = (mat.m8 + mat.m2)*mult;
            result.y = (mat.m6 + mat.m9)*mult;
            break;
		default: break;
    }

    return result;
}

// Get a matrix for a given quaternion
RMAPI Matrix QuaternionToMatrix(const Quaternion q)
{
    const float a2 = q.x*q.x;
    const float b2 = q.y*q.y;
    const float c2 = q.z*q.z;
    const float ac = q.x*q.z;
    const float ab = q.x*q.y;
    const float bc = q.y*q.z;
    const float ad = q.w*q.x;
    const float bd = q.w*q.y;
    const float cd = q.w*q.z;

    Matrix result = { 1.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 1.0f }; // MatrixIdentity()

    result.m0 = 1 - 2*(b2 + c2);
    result.m1 = 2*(ab + cd);
    result.m2 = 2*(ac - bd);

    result.m4 = 2*(ab - cd);
    result.m5 = 1 - 2*(a2 + c2);
    result.m6 = 2*(bc + ad);

    result.m8 = 2*(ac + bd);
    result.m9 = 2*(bc - ad);
    result.m10 = 1 - 2*(a2 + b2);

    return result;
}

// Get rotation quaternion for an angle and axis
// NOTE: Angle must be provided in radians
RMAPI Quaternion QuaternionFromAxisAngle(Vector3 axis, float angle)
{
    const float axisLength = sqrtf(axis.x*axis.x + axis.y*axis.y + axis.z*axis.z);

    Quaternion result = { 0.0f, 0.0f, 0.0f, 1.0f };
    if (axisLength != 0.0f)
    {
        angle *= 0.5f;

        // Vector3Normalize(axis)
        float length = axisLength;
        if (length != 0.0f) {
			axis.x /= length;
			axis.y /= length;
			axis.z /= length;
		}

        const float sinres = sinf(angle);
        const float cosres = cosf(angle);

        result.x = axis.x*sinres;
        result.y = axis.y*sinres;
        result.z = axis.z*sinres;
        result.w = cosres;

        // QuaternionNormalize(q);
        length = sqrtf(result.x*result.x + result.y*result.y + result.z*result.z + result.w*result.w);
        if (length != 0.0f) {
			result.x = result.x / length;
			result.y = result.y / length;
			result.z = result.z / length;
			result.w = result.w / length;
		}
    }

    return result;
}

// Get the rotation angle and axis for a given quaternion
RMAPI void QuaternionToAxisAngle(Quaternion q, Vector3 *outAxis, float *outAngle)
{
    if (fabsf(q.w) > 1.0f)
    {
        // QuaternionNormalize(q);
        float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
        if (length != 0.0f) {
			q.x /= length;
			q.y /= length;
			q.z /= length;
			q.w /= length;
		}
    }

    Vector3 resAxis = { 0.0f, 0.0f, 0.0f };
    const float resAngle = 2.0f*acosf(q.w);
    const float den = sqrtf(1.0f - q.w*q.w);

    if (den > EPSILON)
    {
        resAxis.x = q.x / den;
        resAxis.y = q.y / den;
        resAxis.z = q.z / den;
    }
    else
    {
        // This occurs when the angle is zero.
        // Not a problem: just set an arbitrary normalized axis.
        resAxis.x = 1.0f;
    }

    *outAxis = resAxis;
    *outAngle = resAngle;
}

// Do a Euler Quaternion roll!
// Get the quaternion equivalent to Euler angles
// NOTE: Rotation order is ZYX
RMAPI Quaternion QuaternionFromEuler(const float pitch, const float yaw, const float roll)
{
    const float x0 = cosf(pitch*0.5f);
    const float x1 = sinf(pitch*0.5f);
    const float y0 = cosf(yaw*0.5f);
    const float y1 = sinf(yaw*0.5f);
    const float z0 = cosf(roll*0.5f);
    const float z1 = sinf(roll*0.5f);

    Quaternion result = {
		.x = x1*y0*z0 - x0*y1*z1,
		.y = x0*y1*z0 + x1*y0*z1,
		.z = x0*y0*z1 - x1*y1*z0,
		.w = x0*y0*z0 + x1*y1*z1,
	};
    return result;
}

// Get the Euler angles equivalent to quaternion (roll, pitch, yaw)
// NOTE: Angles are returned in a Vector3 struct in radians
RMAPI Vector3 QuaternionToEuler(const Quaternion q)
{
    // Roll (x-axis rotation)
    const float x0 = 2.0f*(q.w*q.x + q.y*q.z);
    const float x1 = 1.0f - 2.0f*(q.x*q.x + q.y*q.y);

    // Pitch (y-axis rotation)
    float y0 = 2.0f*(q.w*q.y - q.z*q.x);
    y0 = y0 > 1.0f ? 1.0f : y0;
    y0 = y0 < -1.0f ? -1.0f : y0;

    // Yaw (z-axis rotation)
    const float z0 = 2.0f*(q.w*q.z + q.x*q.y);
    const float z1 = 1.0f - 2.0f*(q.y*q.y + q.z*q.z);

    const Vector3 result = {
		.x = atan2f(x0, x1),
		.y = asinf(y0),
		.z = atan2f(z0, z1),
	};
    return result;
}

// Transform a quaternion given a transformation matrix
RMAPI Quaternion QuaternionTransform(const Quaternion q, const Matrix mat)
{
    const Quaternion result = {
		.x = mat.m0*q.x + mat.m4*q.y + mat.m8*q.z + mat.m12*q.w,
		.y = mat.m1*q.x + mat.m5*q.y + mat.m9*q.z + mat.m13*q.w,
		.z = mat.m2*q.x + mat.m6*q.y + mat.m10*q.z + mat.m14*q.w,
		.w = mat.m3*q.x + mat.m7*q.y + mat.m11*q.z + mat.m15*q.w,
	};
    return result;
}

// Check whether two given quaternions are almost equal
RMAPI int QuaternionEquals(const Quaternion p, const Quaternion q)
{
#if !defined(EPSILON)
    #define EPSILON 0.000001f
#endif

    return (((fabsf(p.x - q.x)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.x), fabsf(q.x))))) &&
            ((fabsf(p.y - q.y)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.y), fabsf(q.y))))) &&
            ((fabsf(p.z - q.z)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.z), fabsf(q.z))))) &&
            ((fabsf(p.w - q.w)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.w), fabsf(q.w)))))) ||

		   (((fabsf(p.x + q.x)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.x), fabsf(q.x))))) &&
            ((fabsf(p.y + q.y)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.y), fabsf(q.y))))) &&
            ((fabsf(p.z + q.z)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.z), fabsf(q.z))))) &&
            ((fabsf(p.w + q.w)) <= (EPSILON*fmaxf(1.0f, fmaxf(fabsf(p.w), fabsf(q.w))))));
}

#endif  // RAYMATH_H
