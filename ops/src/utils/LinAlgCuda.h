#pragma once

//=============================================================

template <int dim>
class Vector 
{

private:

    float elements[dim];

    //-----------------------------------

    // data indexing
    __device__ int get3DIndex(const int b, const int i, const int pointCount) const
    {
        return b * pointCount * dim + i * dim;
    }
    
public:

    //-----------------------------------

    // constructors
    __device__ Vector<dim>()
    {
        for (int i=0; i<dim; i++) elements[i] = 0.f;
    }

    __device__ Vector<dim>(const float c)
    {
        for (int i=0; i<dim; i++) elements[i] = c;
    }

    __device__ Vector<dim>(const Vector<dim>& v)
    {
        for (int i=0; i<dim; i++) elements[i] = v(i);
    }

    //-----------------------------------

    // slice from data
    __device__ void slice(const float* data, const int b, const int i, const int pointCount)
    {
        const int startIndex = get3DIndex(b, i, pointCount);
        for (int i=0; i<dim; i++) elements[i] = data[startIndex + i];
    }

    //-----------------------------------

    // write to data
    __device__ void store(float* data, const int b, const int i, const int pointCount)
    {
        int startIndex = get3DIndex(b, i, pointCount);
        for (int i=0; i<dim; i++) data[startIndex + i] = elements[i];
    }

    //-----------------------------------

    // dimensionality
    __device__ int size() const
    {
        return dim;
    }

    //-----------------------------------

    // indexing
    __device__ float operator() (const int index) const 
    {
        return elements[index];
    }

    __device__ float& operator() (const int index)
    {
        return elements[index];
    }

    //-----------------------------------

    // addition
    __device__ Vector<dim> operator+ (const Vector<dim>& v) const 
    {
        Vector<dim> diff;
        for (int i=0; i<dim; i++) diff(i) = elements[i] + v(i);
        return diff;
    }

    //-----------------------------------

    // subtraction
    __device__ Vector<dim> operator- (const Vector<dim>& v) const 
    {
        Vector<dim> diff;
        for (int i=0; i<dim; i++) diff(i) = elements[i] - v(i);
        return diff;
    }

    //-----------------------------------

    // scalar multiplication
    __device__ Vector<dim> operator* (const float& f) const 
    {
        Vector<dim> scaled;
        for (int i=0; i<dim; i++) scaled(i) = elements[i] * f;
        return scaled;
    }

    //-----------------------------------

    // scalar division
    __device__ Vector<dim> operator/ (const float& f) const 
    {
        Vector<dim> scaled;
        for (int i=0; i<dim; i++) scaled(i) = elements[i] / f;
        return scaled;
    }

    //-----------------------------------

    // 2-norm
    __device__ float norm() const
    {
        float n = 0.f;
        for (int i=0; i<dim; i++) n += elements[i] * elements[i];
        return sqrt(n);
    }

    //-----------------------------------

    // dot product
    __device__ float dot(const Vector& v) const
    {
        float d = 0.f;
        for (int i=0; i<dim; i++) d += elements[i] * v(i);
        return d;
    }

};


//=============================================================

template <int dim>
class Matrix
{

private:

    Vector<dim> cols[dim];

public:

    //-----------------------------------

    // slice column
    __device__ Vector<dim> col(const int k) const
    {   
        return cols[k];
    }

    __device__ Vector<dim>& col(const int k)
    {   
        return cols[k];
    }

    //-----------------------------------

    // matrix-vector multiply
    __device__ Vector<dim> operator* (const Vector<dim>& v) const
    {
        Vector<dim> res;
        for (int i=0; i<dim; i++) res(i) = col(i).dot(v);
        return res;
    }

    //-----------------------------------

    // transpose
    __device__ Matrix<dim> transpose() const
    {
        Matrix<dim> ret;
        for (int i=0; i<dim; i++)
        {
            for (int j=0; j<dim; j++) ret.col(i)(j) = cols[j](i);
        }
        return ret;
    }

};