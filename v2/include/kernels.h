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

__device__ __forceinline__ void warp_argmin(float& v, int& lane_src, int& idx) {
    for (int off = 16; off > 0; off >>= 1) {
        float v2 = __shfl_down_sync(0xffffffff, v, off);
        int   l2 = __shfl_down_sync(0xffffffff, lane_src, off);
        int   i2 = __shfl_down_sync(0xffffffff, idx, off);
        if (v2 < v) { v = v2; lane_src = l2; idx = i2; }
    }
}

template<int MAXK>
__device__ __forceinline__ void warp_merge_k_lists(const Pair* __restrict__ my_list, int k,
                                                   Pair* __restrict__ out_list) {
    int lane = (int)(threadIdx.x & 31);
    int ptr = 0;

    float cur = (ptr < k) ? my_list[ptr].d2 : 1e30f;
    int   cur_idx = (ptr < k) ? my_list[ptr].idx : -1;

    #pragma unroll
    for (int it = 0; it < MAXK; ++it) {
        if (it >= k) break;

        float v = cur;
        int   src_lane = lane;
        int   idx = cur_idx;

        warp_argmin(v, src_lane, idx);
        float minv = __shfl_sync(0xffffffff, v, 0);
        int   win_lane = __shfl_sync(0xffffffff, src_lane, 0);
        int   min_idx = __shfl_sync(0xffffffff, idx, 0);

        if (lane == 0) { out_list[it].d2 = minv; out_list[it].idx = min_idx; }

        if (lane == win_lane) {
            ptr++;
            cur = (ptr < k) ? my_list[ptr].d2 : 1e30f;
            cur_idx = (ptr < k) ? my_list[ptr].idx : -1;
        }
    }
}

__global__ void kernel_dist2qc_and_seed(const float* __restrict__ d_queries,
                                        const float* __restrict__ d_centroids,
                                        float* __restrict__ d_dist2qc,
                                        int* __restrict__ d_seed,
                                        int Q, int C, int D)
{
    int q = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    float qv[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) if (j < D) qv[j] = d_queries[(size_t)q * D + j];

    float bestv = 1e30f;
    int bestc = 0;

    for (int c = tid; c < C; c += (int)blockDim.x) {
        const float* cc = d_centroids + (size_t)c * D;
        float s = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < D) { float d = qv[j] - cc[j]; s += d*d; }
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

    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v2 = shv[tid + s];
            int   i2 = shi[tid + s];
            if (v2 < shv[tid]) { shv[tid] = v2; shi[tid] = i2; }
        }
        __syncthreads();
    }
    if (tid == 0) d_seed[q] = shi[0];
}

template<int MAXK>
__global__ void kernel_dk_from_seed_warpreduce_soa(const float* __restrict__ d_queries,
                                                  const float* __restrict__ d_points_soa,
                                                  const int* __restrict__ d_offsets,
                                                  const int* __restrict__ d_sizes,
                                                  const int* __restrict__ d_seed,
                                                  float* __restrict__ d_dk,
                                                  int Q, int D, int k, int N)
{
    int q = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int numWarps = (int)blockDim.x >> 5;

    int c = d_seed[q];
    int off = d_offsets[c];
    int sz  = d_sizes[c];

    float qv[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) if (j < D) qv[j] = d_queries[(size_t)q * D + j];

    Pair best_local[MAXK];
    for (int i = 0; i < k; ++i) { best_local[i].d2 = 1e30f; best_local[i].idx = -1; }

    for (int i = tid; i < sz; i += (int)blockDim.x) {
        int global_idx = off + i;
        float s = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < D) {
                float xj = d_points_soa[(size_t)j * (size_t)N + (size_t)global_idx];
                float d = qv[j] - xj;
                s += d*d;
            }
        }
        topk_insert(best_local, k, s, global_idx);
    }

    __shared__ Pair sh_warp[8 * MAXK];
    Pair warp_best[MAXK];
    warp_merge_k_lists<MAXK>(best_local, k, warp_best);

    if (lane == 0) {
        for (int i = 0; i < k; ++i) sh_warp[warp_id * MAXK + i] = warp_best[i];
    }
    __syncthreads();

    if (warp_id == 0) {
        Pair my_list[MAXK];
        if (lane < numWarps) {
            for (int i = 0; i < k; ++i) my_list[i] = sh_warp[lane * MAXK + i];
        } else {
            for (int i = 0; i < k; ++i) { my_list[i].d2 = 1e30f; my_list[i].idx = -1; }
        }

        Pair merged[MAXK];
        warp_merge_k_lists<MAXK>(my_list, k, merged);

        if (lane == 0) {
            d_dk[q] = sqrtf(merged[k-1].d2);
        }
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
    int q = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    if (tid == 0) d_active_counts[q] = 0;
    __syncthreads();

    float dk = d_dk[q];
    int lane = tid & 31;

    for (int c = tid; c < C; c += (int)blockDim.x) {
        float dist2 = d_dist2qc[(size_t)q * (size_t)C + (size_t)c];
        float thr = dk + d_radii[c];
        float thr2 = thr * thr;
        bool pred = (dist2 <= thr2);

        unsigned mask = __ballot_sync(0xffffffff, pred);
        int n = __popc(mask);
        if (n == 0) continue;

        int base = 0;
        if (lane == 0) base = atomicAdd(&d_active_counts[q], n);
        base = __shfl_sync(0xffffffff, base, 0);

        if (pred) {
            int rank = __popc(mask & ((1u << lane) - 1u));
            int pos = base + rank;
            if (pos < MAX_ACTIVE) d_active_ids[(size_t)q * (size_t)MAX_ACTIVE + (size_t)pos] = c;
        }
    }
}

template<int MAX_ACTIVE, int MAXK>
__global__ void kernel_topk_from_active_warpreduce_soa(const float* __restrict__ d_queries,
                                                      const float* __restrict__ d_points_soa,
                                                      const int* __restrict__ d_offsets,
                                                      const int* __restrict__ d_sizes,
                                                      const int* __restrict__ d_active_counts,
                                                      const int* __restrict__ d_active_ids,
                                                      int* __restrict__ d_out_idx,
                                                      float* __restrict__ d_out_dist,
                                                      int Q, int D, int k, int N)
{
    int q = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int numWarps = (int)blockDim.x >> 5;

    float qv[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) if (j < D) qv[j] = d_queries[(size_t)q * D + j];

    Pair best_local[MAXK];
    for (int i = 0; i < k; ++i) { best_local[i].d2 = 1e30f; best_local[i].idx = -1; }

    int ac = d_active_counts[q];
    if (ac > MAX_ACTIVE) ac = MAX_ACTIVE;

    for (int ai = 0; ai < ac; ++ai) {
        int cid = d_active_ids[(size_t)q * (size_t)MAX_ACTIVE + (size_t)ai];
        int off = d_offsets[cid];
        int sz  = d_sizes[cid];

        for (int p = tid; p < sz; p += (int)blockDim.x) {
            int global_idx = off + p;

            float s = 0.f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                if (j < D) {
                    float xj = d_points_soa[(size_t)j * (size_t)N + (size_t)global_idx];
                    float d = qv[j] - xj;
                    s += d*d;
                }
            }
            topk_insert(best_local, k, s, global_idx);
        }
    }

    __shared__ Pair sh_warp[8 * MAXK];
    Pair warp_best[MAXK];
    warp_merge_k_lists<MAXK>(best_local, k, warp_best);

    if (lane == 0) {
        for (int i = 0; i < k; ++i) sh_warp[warp_id * MAXK + i] = warp_best[i];
    }
    __syncthreads();

    if (warp_id == 0) {
        Pair my_list[MAXK];
        if (lane < numWarps) {
            for (int i = 0; i < k; ++i) my_list[i] = sh_warp[lane * MAXK + i];
        } else {
            for (int i = 0; i < k; ++i) { my_list[i].d2 = 1e30f; my_list[i].idx = -1; }
        }

        Pair merged[MAXK];
        warp_merge_k_lists<MAXK>(my_list, k, merged);

        if (lane == 0) {
            for (int i = 0; i < k; ++i) {
                d_out_idx[(size_t)q * (size_t)k + (size_t)i]  = merged[i].idx;
                d_out_dist[(size_t)q * (size_t)k + (size_t)i] = sqrtf(merged[i].d2);
            }
        }
    }
}
