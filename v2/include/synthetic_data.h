#pragma once

#include "common.h"

static float l2_host(const float* a, const float* b, int dim) {
    float s = 0.f;
    for (int i = 0; i < dim; ++i) { float d = a[i] - b[i]; s += d*d; }
    return std::sqrt(s);
}

static void generate_centroids(std::vector<float>& centroids,
                               int num_clusters, int dim,
                               float min_dist,
                               std::mt19937& gen) {
    std::uniform_real_distribution<float> uni(0.0f, 20.0f);
    centroids.assign((size_t)num_clusters * (size_t)dim, 0.f);

    for (int c = 0; c < num_clusters; ++c) {
        bool ok = false;
        for (int tries = 0; tries < 20000 && !ok; ++tries) {
            std::vector<float> cand(dim);
            for (int j = 0; j < dim; ++j) cand[j] = uni(gen);

            ok = true;
            for (int p = 0; p < c; ++p) {
                const float* prev = centroids.data() + (size_t)p * (size_t)dim;
                if (l2_host(prev, cand.data(), dim) < min_dist) { ok = false; break; }
            }
            if (ok) {
                float* dst = centroids.data() + (size_t)c * (size_t)dim;
                for (int j = 0; j < dim; ++j) dst[j] = cand[j];
            }
        }
        if (!ok) {
            float* dst = centroids.data() + (size_t)c * (size_t)dim;
            for (int j = 0; j < dim; ++j) dst[j] = uni(gen);
        }
    }
}

static void generate_blobs(std::vector<float>& points,
                           const std::vector<float>& centroids,
                           int num_clusters, int dim,
                           int points_per_cluster,
                           float sigma,
                           std::mt19937& gen) {
    std::normal_distribution<float> n01(0.0f, sigma);
    const int total_points = num_clusters * points_per_cluster;
    points.resize((size_t)total_points * (size_t)dim);

    int idx = 0;
    for (int c = 0; c < num_clusters; ++c) {
        const float* mu = centroids.data() + (size_t)c * (size_t)dim;
        for (int t = 0; t < points_per_cluster; ++t) {
            float* x = points.data() + (size_t)idx * (size_t)dim;
            for (int j = 0; j < dim; ++j) x[j] = mu[j] + n01(gen);
            idx++;
        }
    }
}
