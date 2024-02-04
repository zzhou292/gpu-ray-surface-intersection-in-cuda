// Copyright (c) 2022, Raymond Leung
// All rights reserved.
//
// This source code is licensed under the BSD-3-clause license found
// in the LICENSE.md file in the root directory of this source tree.
//
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <vector>

#include <stdint.h>
#include "bvh_structure.h"
#include "rsi_geometry.h"

using namespace std;
using namespace lib_bvh;
using namespace lib_rsi;

//-------------------------------------------------
// This implementation corresponds to version v3
// with support for barycentric mode and the
// intercept_count experimental feature
//-------------------------------------------------

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void CheckSyncAsyncErrors(const char* file, int line) {
    // Inspired from https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
    cudaError_t errSync =
        cudaGetLastError();  // returns the value of the latest asynchronous error and also resets it to cudaSuccess.
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) {
        printf("Sync kernel error\n");
        HandleError(errSync, file, line);
    }
    if (errAsync != cudaSuccess) {
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        HandleError(errAsync, file, line);
    }
}

#define CUDA_SYNCHRO_CHECK() (CheckSyncAsyncErrors(__FILE__, __LINE__))

template <class T>
int readData(string fname, vector<T>& v, int dim = 1, bool silent = false) {
    ifstream infile(fname.c_str(), ios::binary | ios::ate);
    if (!infile) {
        cerr << "File " << fname << " not found" << endl;
        exit(1);
    }
    ifstream::pos_type nbytes = infile.tellg();
    infile.seekg(0, infile.beg);
    const int elements = nbytes / sizeof(T);
    v.resize(elements);
    infile.read(reinterpret_cast<char*>(v.data()), nbytes);
    if (!silent) {
        cout << fname << " contains " << nbytes << " bytes, " << v.size() << " <" << typeid(v.front()).name() << ">, "
             << v.size() / dim << " elements" << endl;
    }
    return elements / dim;
}

template <class T>
void writeData(string fname, vector<T>& v) {
    ofstream outfile(fname.c_str(), ios::out | ios::binary);
    if (!outfile) {
        cerr << "Cannot create " << fname << " for writing" << endl;
        exit(1);
    }
    outfile.write(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
    outfile.close();
}

// =================== JSON added helper function =======================
void readMeshData(string fname, vector<float>& v, vector<int>& surf) {
    std::ifstream file(fname);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float x,y,z;
            iss >> x >> y >> z;
            v.push_back(x);
            v.push_back(y);
            v.push_back(z);
        } else if (prefix == "f") {
            std::string vertex;
            for (int i = 0; i < 3; ++i) {
                iss >> vertex;
                size_t slash_pos = vertex.find('/');
                if (slash_pos != std::string::npos) {
                    vertex = vertex.substr(0, slash_pos);
                }
                int idx = std::stoi(vertex) - 1;
                surf.push_back(idx);
            }
        }
    }

}

// ======================================================================
// this helper converts intersection points from Baycentric coordinates to Cartesian coordinates
void intersect_helper(vector<int>& h_intersectTriangle,
    vector<float>& h_baryT, vector<float>& h_baryU, vector<float>& h_baryV, const int& nVertices, const int& nTriangles, const int& nRays, vector<float>& h_vertices, vector<int>& h_triangles, vector<float>& p_intersect){
        for(int i = 0; i < nRays; i++){
            if(h_intersectTriangle[i] != -1){
                int f = h_intersectTriangle[i];
                int i0 = h_triangles[3*f];
                int i1 = h_triangles[3*f+1];
                int i2 = h_triangles[3*f+2];
                float t = h_baryT[i];
                float u = h_baryU[i];
                float v = h_baryV[i];
                float x = (1-u-v)*h_vertices[3*i0] + u*h_vertices[3*i1] + v*h_vertices[3*i2];
                float y = (1-u-v)*h_vertices[3*i0+1] + u*h_vertices[3*i1+1] + v*h_vertices[3*i2+1];
                float z = (1-u-v)*h_vertices[3*i0+2] + u*h_vertices[3*i1+2] + v*h_vertices[3*i2+2];
                p_intersect[3*i] = x;
                p_intersect[3*i+1] = y;
                p_intersect[3*i+2] = z;
            }
        }
    }  

void ray_generation_helper(vector<float>& h_rayFrom, vector<float>& h_rayTo, int& nRays){
    // h_rayFrom = {0.0, 0.0, -0.4, 10.0, 0.0, 0.0};
    // h_rayTo = {0.0, 0.0, 5.0, 10.0, 0.0, 10.0};
    // nRays = 2;

    //a dummy ray generation of SCM grid 1m x 0.5 m, centered at 0, with mesh resolution 0.01.
    float x_dim = 2.0; float y_dim = 1.0;
    float x_c = 0.0; float y_c = 0.0;
    float resolution = 0.01;
    int n_x = int(x_dim / resolution)+1;
    int n_y = int(y_dim / resolution)+1;
    nRays = n_x * n_y;

    for(int i = 0; i < n_x; i++){
        for(int j = 0; j < n_y; j++){
            h_rayFrom.push_back(x_c - x_dim/2 + i*resolution);
            h_rayFrom.push_back(y_c - y_dim/2 + j*resolution);
            h_rayFrom.push_back(-0.4);
            h_rayTo.push_back(x_c - x_dim/2 + i*resolution);
            h_rayTo.push_back(y_c - y_dim/2 + j*resolution);
            h_rayTo.push_back(5.0);
        }
    }
}

// ======================================================================

int main(int argc, char* argv[]) {
    const bool checkEnabled(true);
    const float largePosVal(2.5e+8);
    vector<float> h_vertices;
    vector<int> h_triangles;
    vector<float> h_rayFrom;
    vector<float> h_rayTo;
    vector<int> h_crossingDetected;
    vector<int> h_intersectTriangle;
    vector<float> h_baryT, h_baryU, h_baryV;
    int nVertices, nTriangles, nRays;

    bool quietMode=false;
    /*
    Ray-surface intersection results are reported as follows:
      barycentric = false
      |  if interceptsCount is false (by default)
      |     return boolean array, h_crossingDetected[r] is set to 0 or 1
      |  else report the number of surface intersections for each ray
      |     return integer array, h_crossingDetected[r] >= 0
      barycentric = true
      |  return index of intersecting triangle (f) via h_intersectTriangle[r]
      |  (-1 if none) and the intersecting point P via barycentric coordinates
      |  (t[r], u[r], v[r]) where t = distance(rayFrom, surface), P =
      |  (1-u-v)*V[0] + u*V[1] + v*V[2], V[i] = vertices[triangles[f][i]].
    */
    bool barycentric = true;
    bool interceptsCount(argc > 6 ? strcmp(argv[6], "intercept_count") == 0 : false);

    // read input data into host memory
    readMeshData("cobra_wheel.obj", h_vertices, h_triangles);
    nVertices = h_vertices.size() / 3;
    nTriangles = h_triangles.size() / 3;
    cout << "v size:"<< nVertices << " t size:" << nTriangles << endl;
    //nVertices = readData(fileVertices, h_vertices, 3, quietMode);
    //nTriangles = readData(fileTriangles, h_triangles, 3, quietMode);

    if (h_triangles.size() == 3) {
        // Add an extra triangle so that BVH traversal works in an
        // uncomplicated way without throwing an exception. It
        // expects at least one split node at the top of the binary
        // radix tree where the left and right child nodes are defined.
        for (int i = 0; i < 3; i++)
            h_triangles.push_back(0);
        nTriangles += 1;
    }

    //nRays = readData(fileFrom, h_rayFrom, 3, quietMode);
    //int nRaysTo = readData(fileTo, h_rayTo, 3, quietMode);
    //assert(nRaysTo == nRays);

    ray_generation_helper(h_rayFrom, h_rayTo, nRays);
    std::cout << "nRays: " << nRays << std::endl;

    h_crossingDetected.resize(nRays);

    cudaEvent_t start, end;
    float time = 0;
    float *d_vertices, *d_rayFrom, *d_rayTo;
    int *d_triangles, *d_crossingDetected, *d_intersectTriangle;
    float *d_baryT, *d_baryU, *d_baryV;
    AABB* d_rayBox;
    int sz_vertices(3 * nVertices * sizeof(float)), sz_triangles(3 * nTriangles * sizeof(int)),
        sz_rays(3 * nRays * sizeof(float)), sz_rbox(nRays * sizeof(AABB)), sz_id(nRays * sizeof(int)),
        sz_bary(nRays * sizeof(float));
    HANDLE_ERROR(cudaMalloc(&d_vertices, sz_vertices));
    HANDLE_ERROR(cudaMalloc(&d_triangles, sz_triangles));
    HANDLE_ERROR(cudaMalloc(&d_rayFrom, sz_rays));
    HANDLE_ERROR(cudaMalloc(&d_rayTo, sz_rays));
    HANDLE_ERROR(cudaMalloc(&d_rayBox, sz_rbox));

    if (!barycentric) {
        HANDLE_ERROR(cudaMalloc(&d_crossingDetected, sz_id));
        HANDLE_ERROR(cudaMemset(d_crossingDetected, 0, sz_id));
    } else {
        h_intersectTriangle.resize(nRays);
        h_baryT.resize(nRays);
        h_baryU.resize(nRays);
        h_baryV.resize(nRays);
        HANDLE_ERROR(cudaMalloc(&d_intersectTriangle, sz_id));
        HANDLE_ERROR(cudaMalloc(&d_baryT, sz_bary));
        HANDLE_ERROR(cudaMalloc(&d_baryU, sz_bary));
        HANDLE_ERROR(cudaMalloc(&d_baryV, sz_bary));
    }
    HANDLE_ERROR(cudaMemcpy(d_vertices, h_vertices.data(), sz_vertices, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_triangles, h_triangles.data(), sz_triangles, cudaMemcpyHostToDevice));

    // grid partitions
    int blockX = 1024, gridXr = (int)ceil((float)nRays / blockX), gridXt = (int)ceil((float)nTriangles / blockX),
        gridXLambda = 16;  // N_{grids}
    if (!quietMode) {
        cout << blockX << " threads/block, grids: {triangles: " << gridXt << ", rays: " << gridXLambda << "}" << endl;
    }
    float minval[3], maxval[3], half_delta[3], inv_delta[3];
    vector<uint64_t> h_morton;
    vector<int> h_sortedTriangleIDs;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));
    HANDLE_ERROR(cudaEventRecord(start));
    HANDLE_ERROR(cudaMemcpy(d_rayFrom, h_rayFrom.data(), sz_rays, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_rayTo, h_rayTo.data(), sz_rays, cudaMemcpyHostToDevice));

    // initialise arrays
    if (barycentric) {
        initArrayKernel<<<gridXr, blockX>>>(d_intersectTriangle, -1, nRays);
        initArrayKernel<<<gridXr, blockX>>>(d_baryT, largePosVal, nRays);
    }
    HANDLE_ERROR(cudaDeviceSynchronize());

    // compute ray-segment bounding boxes
    rbxKernel<<<gridXr, blockX>>>(d_rayFrom, d_rayTo, d_rayBox, nRays);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // order triangles using Morton code
    //- normalise surface vertices to canvas coords
    getMinMaxExtentOfSurface<float>(h_vertices, minval, maxval, half_delta, inv_delta, nVertices, quietMode);
    //- convert centroid of triangles to morton code
    createMortonCode<float, uint64_t>(h_vertices, h_triangles, minval, half_delta, inv_delta, h_morton, nTriangles);
    //- sort before constructing binary radix tree
    sortMortonCode<uint64_t>(h_morton, h_sortedTriangleIDs);
    if (!quietMode && checkEnabled) {
        cout << "checking sortMortonCode" << endl;
        for (int j = 0; j < min(12, nTriangles); j++) {
            cout << j << ": (" << h_sortedTriangleIDs[j] << ") " << h_morton[j] << endl;
        }
    }
    // data structures used in agglomerative LBVH construction
    BVHNode *d_leafNodes, *d_internalNodes;
    uint64_t* d_morton;
    int* d_sortedTriangleIDs;
    CollisionList* d_hitIDs;
    int sz_morton(nTriangles * sizeof(uint64_t)), sz_sortedIDs(nTriangles * sizeof(int)),
        sz_hitIDs(gridXLambda * blockX * sizeof(CollisionList));
    InterceptDistances* d_interceptDists;
    int sz_interceptDists(gridXLambda * blockX * sizeof(InterceptDistances));
    HANDLE_ERROR(cudaMalloc(&d_leafNodes, nTriangles * sizeof(BVHNode)));
    HANDLE_ERROR(cudaMalloc(&d_internalNodes, nTriangles * sizeof(BVHNode)));
    HANDLE_ERROR(cudaMalloc(&d_morton, sz_morton));
    HANDLE_ERROR(cudaMalloc(&d_sortedTriangleIDs, sz_sortedIDs));
    HANDLE_ERROR(cudaMalloc(&d_hitIDs, sz_hitIDs));
    if (interceptsCount) {
        HANDLE_ERROR(cudaMalloc(&d_interceptDists, sz_interceptDists));
    }
    HANDLE_ERROR(cudaMemcpy(d_morton, h_morton.data(), sz_morton, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_sortedTriangleIDs, h_sortedTriangleIDs.data(), sz_sortedIDs, cudaMemcpyHostToDevice));
    std::vector<uint64_t>().swap(h_morton);
    std::vector<int>().swap(h_sortedTriangleIDs);

    bvhResetKernel<<<gridXt, blockX>>>(d_vertices, d_triangles, d_internalNodes, d_leafNodes, d_sortedTriangleIDs,
                                       nTriangles);
    HANDLE_ERROR(cudaDeviceSynchronize());

    bvhConstruct<uint64_t><<<gridXt, blockX>>>(d_internalNodes, d_leafNodes, d_morton, nTriangles);
    // HANDLE_ERROR(cudaDeviceSynchronize());
    CUDA_SYNCHRO_CHECK();

    if (barycentric) {
        bvhIntersectionKernel<<<gridXLambda, blockX>>>(d_vertices, d_triangles, d_rayFrom, d_rayTo, d_internalNodes,
                                                       d_rayBox, d_hitIDs, d_intersectTriangle, d_baryT, d_baryU,
                                                       d_baryV, nTriangles, nRays);
    } else if (interceptsCount) {
        bvhIntersectionKernel<<<gridXLambda, blockX>>>(d_vertices, d_triangles, d_rayFrom, d_rayTo, d_internalNodes,
                                                       d_rayBox, d_hitIDs, d_interceptDists, d_crossingDetected,
                                                       nTriangles, nRays);
    } else {
        bvhIntersectionKernel<<<gridXLambda, blockX>>>(d_vertices, d_triangles, d_rayFrom, d_rayTo, d_internalNodes,
                                                       d_rayBox, d_hitIDs, d_crossingDetected, nTriangles, nRays);
    }
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaEventRecord(end));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, end));

    if (!barycentric) {
        HANDLE_ERROR(cudaMemcpy(h_crossingDetected.data(), d_crossingDetected, sz_id, cudaMemcpyDeviceToHost));
        writeData("results_i32", h_crossingDetected);
    } else {
        HANDLE_ERROR(cudaMemcpy(h_intersectTriangle.data(), d_intersectTriangle, sz_id, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(h_baryT.data(), d_baryT, sz_bary, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(h_baryU.data(), d_baryU, sz_bary, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(h_baryV.data(), d_baryV, sz_bary, cudaMemcpyDeviceToHost));
        writeData("intersectTriangle_i32", h_intersectTriangle);
        writeData("barycentricT_f32", h_baryT);
        writeData("barycentricU_f32", h_baryU);
        writeData("barycentricV_f32", h_baryV);
    }

    vector<float> p_intersect(nRays * 3, -1.0);
    // calculate the intersection point based on the barycentric coordinates
    intersect_helper(h_intersectTriangle,h_baryT, h_baryU, h_baryV, nVertices, nTriangles, nRays, h_vertices, h_triangles, p_intersect);

    // sanity check
    vector<int>& outcome = !barycentric ? h_crossingDetected : h_intersectTriangle;
    if (!quietMode) {
        cout << "Results for all intersection elements:" << endl;
        for (int i = 0; i < nRays; i++) {
            if(outcome[i] != -1){
                cout << i << ": " << outcome[i] << "," << p_intersect[3*i] << "," << p_intersect[3*i+1] << "," << p_intersect[3*i+2] << endl;
            }

        }
        cout << "Processing time: ";
        cout << time << " ms" << endl;
    }
}
