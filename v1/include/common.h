#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
  std::exit(1); } } while(0)

static inline double ms_since(const std::chrono::high_resolution_clock::time_point& t0,
                              const std::chrono::high_resolution_clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
