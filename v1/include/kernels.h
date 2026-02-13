#pragma once

#include "common.h"

struct Pair { float d2; int idx; };

__device__ __forceinline__ void topk_insert(Pair* best, int k, float d2, int idx) {
    if (d2 >= best[k-1].d2) return;
    int pos = k - 1;
    best[pos].d2 = d2;
    best[pos].idx = idx;
    while (pos > 0 && best[pos].d2 < best[pos-1].d2) {
        Pair tmp = best[pos-1];
        best[pos-1] = best[pos];
        best[pos] = tmp;
        pos--;
    }
}

__global__ void kernel_dist2qc_and_seed(const float* __restrict__ d_queries,
                                        const float* __restrict__ d_centroids,
                                        float* __restrict__ d_dist2qc,
                                        int* __restrict__ d_seed,
                                        int Q, int C, int D)
{
    int q = blockIdx.x;
    int tid = threadIdx.x;

    float qv[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < D) qv[j] = d_queries[(size_t)q * D + j];
    }

    float bestv = 1e30f;
    int bestc = 0;

    for (int c = tid; c < C; c += blockDim.x) {
        const float* cc = d_centroids + (size_t)c * D;
        float s = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < D) {
                float d = qv[j] - cc[j];
                s += d*d;
            }
        }
        d_dist2qc[(size_t)q * C + c] = s;
        if (s < bestv) { bestv = s; bestc = c; }
    }

    extern __shared__ unsigned char smem[];
    float* shv = (float*)smem;
    int*   shi = (int*)(shv + blockDim.x);

    shv[tid] = bestv;
    shi[tid] = bestc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v2 = shv[tid + s];
            int   i2 = shi[tid + s];
            if (v2 < shv[tid]) { shv[tid] = v2; shi[tid] = i2; }
        }
        __syncthreads();
    }

    if (tid == 0) d_seed[q] = shi[0];
}

__global__ void kernel_dk_from_seed(const float* __restrict__ d_queries,
                                   const float* __restrict__ d_points_packed,
                                   const int* __restrict__ d_offsets,
                                   const int* __restrict__ d_sizes,
                                   const int* __restrict__ d_seed,
                                   float* __restrict__ d_dk,
                                   int Q, int D, int k)
{
    int q = blockIdx.x;
    int tid = threadIdx.x;

    int c = d_seed[q];
    int off = d_offsets[c];
    int sz  = d_sizes[c];

    float qv[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < D) qv[j] = d_queries[(size_t)q * D + j];
    }

    Pair best_local[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        best_local[i].d2 = 1e30f;
        best_local[i].idx = -1;
    }

    for (int i = tid; i < sz; i += blockDim.x) {
        const float* x = d_points_packed + (size_t)(off + i) * D;

        float s = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < D) {
                float d = qv[j] - x[j];
                s += d*d;
            }
        }
        topk_insert(best_local, k, s, off + i);
    }

    extern __shared__ Pair shpairs[];
    for (int i = 0; i < k; ++i) shpairs[(size_t)tid * k + i] = best_local[i];
    __syncthreads();

    if (tid == 0) {
        Pair best[32];
        for (int i = 0; i < k; ++i) { best[i].d2 = 1e30f; best[i].idx = -1; }

        for (int t = 0; t < blockDim.x; ++t) {
            for (int i = 0; i < k; ++i) {
                Pair p = shpairs[(size_t)t * k + i];
                if (p.idx >= 0) topk_insert(best, k, p.d2, p.idx);
            }
        }
        d_dk[q] = sqrtf(best[k-1].d2);
    }
}

template<int MAX_ACTIVE>
__global__ void kernel_build_active_from_dist2(const float* __restrict__ d_dist2qc,
                                               const float* __restrict__ d_radii,
                                               const float* __restrict__ d_dk,
                                               int* __restrict__ d_active_counts,
                                               int* __restrict__ d_active_ids,
                                               int Q, int C)
{
    int q = blockIdx.x;
    int tid = threadIdx.x;

    if (tid == 0) d_active_counts[q] = 0;
    __syncthreads();

    float dk = d_dk[q];

    int lane = tid & 31;
    int warp = tid >> 5;

    for (int c = tid; c < C; c += blockDim.x) {
        float dist2 = d_dist2qc[(size_t)q * C + c];
        float thr = dk + d_radii[c];
        float thr2 = thr * thr;
        bool pred = (dist2 <= thr2);

        unsigned mask = __ballot_sync(0xffffffff, pred);
        int n = __popc(mask);
        if (n == 0) continue;

        int base = 0;
        if (lane == 0) {
            base = atomicAdd(&d_active_counts[q], n);
        }
        base = __shfl_sync(0xffffffff, base, 0);

        if (pred) {
            int rank = __popc(mask & ((1u << lane) - 1u));
            int pos = base + rank;
            if (pos < MAX_ACTIVE) {
                d_active_ids[(size_t)q * MAX_ACTIVE + pos] = c;
            }
        }
    }
}

template<int MAX_ACTIVE, int MAX_K>
__global__ void kernel_topk_from_active(const float* __restrict__ d_queries,
                                        const float* __restrict__ d_points_packed,
                                        const int* __restrict__ d_offsets,
                                        const int* __restrict__ d_sizes,
                                        const int* __restrict__ d_active_counts,
                                        const int* __restrict__ d_active_ids,
                                        int* __restrict__ d_out_idx,
                                        float* __restrict__ d_out_dist,
                                        int Q, int D, int k)
{
    int q = blockIdx.x;
    int tid = threadIdx.x;

    float qv[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < D) qv[j] = d_queries[(size_t)q * D + j];
    }

    Pair best_local[MAX_K];
    for (int i = 0; i < k; ++i) { best_local[i].d2 = 1e30f; best_local[i].idx = -1; }

    int ac = d_active_counts[q];
    if (ac > MAX_ACTIVE) ac = MAX_ACTIVE;

    for (int ai = 0; ai < ac; ++ai) {
        int cid = d_active_ids[(size_t)q * MAX_ACTIVE + ai];
        int off = d_offsets[cid];
        int sz  = d_sizes[cid];

        for (int p = tid; p < sz; p += blockDim.x) {
            int global_idx = off + p;
            const float* x = d_points_packed + (size_t)global_idx * D;

            float s = 0.f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                if (j < D) {
                    float d = qv[j] - x[j];
                    s += d*d;
                }
            }
            topk_insert(best_local, k, s, global_idx);
        }
    }

    extern __shared__ Pair shpairs[];
    for (int i = 0; i < k; ++i) shpairs[(size_t)tid * k + i] = best_local[i];
    __syncthreads();

    if (tid == 0) {
        Pair best[MAX_K];
        for (int i = 0; i < k; ++i) { best[i].d2 = 1e30f; best[i].idx = -1; }

        for (int t = 0; t < blockDim.x; ++t) {
            for (int i = 0; i < k; ++i) {
                Pair p = shpairs[(size_t)t * k + i];
                if (p.idx >= 0) topk_insert(best, k, p.d2, p.idx);
            }
        }

        for (int i = 0; i < k; ++i) {
            d_out_idx[(size_t)q * k + i]  = best[i].idx;
            d_out_dist[(size_t)q * k + i] = sqrtf(best[i].d2);
        }
    }
}
