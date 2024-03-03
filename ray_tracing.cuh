#ifndef RAY_TRACING_H
#define RAY_TRACING_H

#include <vector>

// Declare the function
void gpu_trace_ray(std::vector<float> h_vertices, std::vector<int> h_triangles, std::vector<float> h_rayFrom, std::vector<float> h_rayTo, std::vector<int>& valid_outcome_idx, std::vector<float>& valid_outcome);

#endif // RAY_TRACING_H
