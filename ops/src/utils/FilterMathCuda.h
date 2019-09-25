#pragma once
#include "LinAlgCuda.h"

//=============================================================

#define BATCH_BLOCK_SIZE 1
#define POINT_BLOCK_SIZE 256

//=============================================================

template <int dim>
__device__ static Vector<dim> toroidalMinShift(const Vector<dim>& v, const Vector<dim>& reference)
{
    Vector<dim> ret(v);
    for (int d=0; d<dim; d++)
    {
        if (abs(reference(d) - v(d)) > 0.5f)
        {
            ret(d) += (reference(d) < 0.5f) ? -1.0f : 1.0f;
        }
    }
    return ret;
}

//=============================================================

__device__ static float lerpGPU(const float a, const float b, const float w)
{
	return (1.0f - w) * a + w * b;
}

//=============================================================

template <int dim>
__device__ static Vector<dim> oneHot(const int k)
{
	Vector<dim> v;
    v(k) = 1.0f;
    return v;
}

//=============================================================

__device__ static int get2DIndex(const int b, const int i, const int elementCount)
    {
        return b * elementCount + i;
    }


