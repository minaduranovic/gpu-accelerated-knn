#include "kmeans_cpu.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>

static inline float dist2(const float* a, const float* b, int D) {
    float s = 0.0f;
    for (int d = 0; d < D; ++d) {
        float diff = a[d] - b[d];
        s += diff * diff;
    }
    return s;
}

void kmeans_fit(
    const std::vector<float>& points,
    int N, int D, int K,
    int iters,
    std::vector<int>& labels,
    std::vector<float>& centroids
) {
    labels.assign(N, 0);
    centroids.assign(K * D, 0.0f);

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> uid(0, N - 1);
    for (int k = 0; k < K; ++k) {
        int idx = uid(gen);
        std::copy(points.begin() + idx * D, points.begin() + (idx + 1) * D,
                  centroids.begin() + k * D);
    }

    std::vector<float> new_centroids(K * D, 0.0f);
    std::vector<int> counts(K, 0);

    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < N; ++i) {
            const float* p = &points[i * D];
            int best_k = 0;
            float best = std::numeric_limits<float>::infinity();

            for (int k = 0; k < K; ++k) {
                const float* c = &centroids[k * D];
                float d2 = dist2(p, c, D);
                if (d2 < best) { best = d2; best_k = k; }
            }
            labels[i] = best_k;
        }

        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (int i = 0; i < N; ++i) {
            int k = labels[i];
            const float* p = &points[i * D];
            float* acc = &new_centroids[k * D];
            for (int d = 0; d < D; ++d) acc[d] += p[d];
            counts[k]++;
        }

        for (int k = 0; k < K; ++k) {
            float* c = &new_centroids[k * D];
            if (counts[k] > 0) {
                float inv = 1.0f / counts[k];
                for (int d = 0; d < D; ++d) c[d] *= inv;
            } else {
                int idx = uid(gen);
                std::copy(points.begin() + idx * D, points.begin() + (idx + 1) * D, c);
            }
        }

        centroids.swap(new_centroids);
    }
}

void compute_cluster_radii(
    const std::vector<float>& points,
    int N, int D, int K,
    const std::vector<int>& labels,
    const std::vector<float>& centroids,
    std::vector<float>& radii
) {
    radii.assign(K, 0.0f);
    for (int i = 0; i < N; ++i) {
        int k = labels[i];
        const float* p = &points[i * D];
        const float* c = &centroids[k * D];
        float d = std::sqrt(dist2(p, c, D));
        if (d > radii[k]) radii[k] = d;
    }
}

void pack_points_by_cluster(
    const std::vector<float>& points,
    int N, int D, int K,
    const std::vector<int>& labels,
    std::vector<float>& packed_points,
    std::vector<int>& cluster_offsets,
    std::vector<int>& cluster_sizes
) {
    cluster_sizes.assign(K, 0);
    for (int i = 0; i < N; ++i) cluster_sizes[labels[i]]++;

    cluster_offsets.assign(K, 0);
    int sum = 0;
    for (int k = 0; k < K; ++k) {
        cluster_offsets[k] = sum;
        sum += cluster_sizes[k];
    }

    packed_points.assign((size_t)N * (size_t)D, 0.0f);

    std::vector<int> cursor = cluster_offsets;

    for (int i = 0; i < N; ++i) {
        int k = labels[i];
        int pos = cursor[k]++;
        std::copy(points.begin() + i * D, points.begin() + (i + 1) * D,
                  packed_points.begin() + pos * D);
    }
}
