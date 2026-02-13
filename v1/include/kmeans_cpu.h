#pragma once
#include <vector>

void kmeans_fit(
    const std::vector<float>& points,
    int N, int D, int K,
    int iters,
    std::vector<int>& labels,
    std::vector<float>& centroids
);

void compute_cluster_radii(
    const std::vector<float>& points,
    int N, int D, int K,
    const std::vector<int>& labels,
    const std::vector<float>& centroids,
    std::vector<float>& radii
);

void pack_points_by_cluster(
    const std::vector<float>& points,
    int N, int D, int K,
    const std::vector<int>& labels,
    std::vector<float>& packed_points,
    std::vector<int>& cluster_offsets,
    std::vector<int>& cluster_sizes
);
