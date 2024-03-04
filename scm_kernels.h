#ifndef SCM_KERNELS_H
#define SCM_KERNELS_H
#pragma once

#include <algorithm> //std::stable_sort
#include <assert.h>
#include <numeric> //std::iota
#include <queue>
#include <sstream>
#include <stdint.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector>

__global__ void intersect_helper_cuda(int *d_intersectTriangle, float *d_baryT,
                                      float *d_baryU, float *d_baryV,
                                      float *d_vertices, int *d_triangles,
                                      float *d_intersect, int nRays) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nRays) {
    if (d_intersectTriangle[i] != -1) {
      int f = d_intersectTriangle[i];
      int i0 = d_triangles[3 * f];
      int i1 = d_triangles[3 * f + 1];
      int i2 = d_triangles[3 * f + 2];
      //float t = d_baryT[i];
      float u = d_baryU[i];
      float v = d_baryV[i];
      float x = (1 - u - v) * d_vertices[3 * i0] + u * d_vertices[3 * i1] +
                v * d_vertices[3 * i2];
      float y = (1 - u - v) * d_vertices[3 * i0 + 1] +
                u * d_vertices[3 * i1 + 1] + v * d_vertices[3 * i2 + 1];
      float z = (1 - u - v) * d_vertices[3 * i0 + 2] +
                u * d_vertices[3 * i1 + 2] + v * d_vertices[3 * i2 + 2];
      d_intersect[3 * i] = x;
      d_intersect[3 * i + 1] = y;
      d_intersect[3 * i + 2] = z;
    }
  }
}

#endif // RAY_TRACING_H
