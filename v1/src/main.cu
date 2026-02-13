#include "common.h"
#include "kmeans_cpu.h"
#include "kernels.h"
#include "synthetic_data.h"

int main() {
    const int dim = 8;
    const int num_clusters = 256;
    const int points_per_cluster = 512;
    const int total_points = num_clusters * points_per_cluster;

    const int batchQ = 1024;
    const int k = 20;
    const int kmeans_iters = 6;

    const float sigma = 0.08f;
    const float min_center_dist = 4.0f;

    constexpr int MAX_ACTIVE = 32;
    constexpr int MAX_K = 32;

    std::mt19937 gen(42);

    std::cout << "Batch GPU-only STEP synthetic (OPT)\n";
    std::cout << "dim=" << dim
              << " clusters=" << num_clusters
              << " points=" << total_points
              << " per_cluster=" << points_per_cluster
              << " batchQ=" << batchQ
              << " k=" << k
              << " sigma=" << sigma
              << " min_center_dist=" << min_center_dist
              << "\n\n";

    std::vector<float> true_centroids;
    generate_centroids(true_centroids, num_clusters, dim, min_center_dist, gen);

    std::vector<float> h_points;
    generate_blobs(h_points, true_centroids, num_clusters, dim, points_per_cluster, sigma, gen);

    std::normal_distribution<float> qnoise(0.0f, sigma * 0.25f);
    std::vector<float> h_queries((size_t)batchQ * (size_t)dim);
    for (int q = 0; q < batchQ; ++q) {
        int base_c = (q & 1);
        const float* mu = true_centroids.data() + (size_t)base_c * (size_t)dim;
        float* dst = h_queries.data() + (size_t)q * (size_t)dim;
        for (int j = 0; j < dim; ++j) dst[j] = mu[j] + qnoise(gen);
    }

    std::vector<int> labels;
    std::vector<float> h_centroids;
    std::vector<float> h_radii;
    std::vector<float> h_packed_points;
    std::vector<int> h_offsets;
    std::vector<int> h_sizes;

    auto t0_km = std::chrono::high_resolution_clock::now();
    kmeans_fit(h_points, total_points, dim, num_clusters, kmeans_iters, labels, h_centroids);
    compute_cluster_radii(h_points, total_points, dim, num_clusters, labels, h_centroids, h_radii);
    pack_points_by_cluster(h_points, total_points, dim, num_clusters, labels, h_packed_points, h_offsets, h_sizes);
    auto t1_km = std::chrono::high_resolution_clock::now();

    std::cout << "Offline k-means+pack (CPU): " << std::fixed << std::setprecision(3)
              << ms_since(t0_km, t1_km) << " ms\n";

    float* d_points = nullptr;
    float* d_queries = nullptr;
    float* d_centroids = nullptr;
    float* d_radii = nullptr;
    int* d_offsets = nullptr;
    int* d_sizes = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points,    (size_t)total_points * (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_queries,   (size_t)batchQ * (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)num_clusters * (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radii,     (size_t)num_clusters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_offsets,   (size_t)num_clusters * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sizes,     (size_t)num_clusters * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points,    h_packed_points.data(),
                          (size_t)total_points * (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries,   h_queries.data(),
                          (size_t)batchQ * (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(),
                          (size_t)num_clusters * (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radii,     h_radii.data(),
                          (size_t)num_clusters * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets,   h_offsets.data(),
                          (size_t)num_clusters * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes,     h_sizes.data(),
                          (size_t)num_clusters * sizeof(int), cudaMemcpyHostToDevice));

    float* d_dist2qc = nullptr;
    int*   d_seed = nullptr;
    float* d_dk = nullptr;

    int* d_active_counts = nullptr;
    int* d_active_ids = nullptr;

    int* d_out_idx = nullptr;
    float* d_out_dist = nullptr;

    CUDA_CHECK(cudaMalloc(&d_dist2qc, (size_t)batchQ * (size_t)num_clusters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_seed,    (size_t)batchQ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dk,      (size_t)batchQ * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_active_counts, (size_t)batchQ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active_ids,    (size_t)batchQ * (size_t)MAX_ACTIVE * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_out_idx,  (size_t)batchQ * (size_t)k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_dist, (size_t)batchQ * (size_t)k * sizeof(float)));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    int blockAB = 256;
    size_t shAB = (size_t)blockAB * (sizeof(float) + sizeof(int));
    kernel_dist2qc_and_seed<<<batchQ, blockAB, shAB>>>(d_queries, d_centroids, d_dist2qc, d_seed, batchQ, num_clusters, dim);
    CUDA_CHECK(cudaGetLastError());

    int blockC = 128;
    size_t shC = (size_t)blockC * (size_t)k * sizeof(Pair);
    kernel_dk_from_seed<<<batchQ, blockC, shC>>>(d_queries, d_points, d_offsets, d_sizes, d_seed, d_dk, batchQ, dim, k);
    CUDA_CHECK(cudaGetLastError());

    int blockD = 256;
    kernel_build_active_from_dist2<MAX_ACTIVE><<<batchQ, blockD>>>(
        d_dist2qc, d_radii, d_dk, d_active_counts, d_active_ids, batchQ, num_clusters
    );
    CUDA_CHECK(cudaGetLastError());

    int blockE = 128;
    size_t shE = (size_t)blockE * (size_t)k * sizeof(Pair);
    kernel_topk_from_active<MAX_ACTIVE, MAX_K><<<batchQ, blockE, shE>>>(
        d_queries, d_points, d_offsets, d_sizes,
        d_active_counts, d_active_ids,
        d_out_idx, d_out_dist,
        batchQ, dim, k
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double gpu_ms = ms_since(t0, t1);
    std::cout << "GPU-only batch pipeline time (OPT): " << std::fixed << std::setprecision(3) << gpu_ms << " ms\n";

    std::vector<int> h_active_counts(batchQ);
    CUDA_CHECK(cudaMemcpy(h_active_counts.data(), d_active_counts, (size_t)batchQ * sizeof(int), cudaMemcpyDeviceToHost));

    long long sum_active = 0;
    int max_active = 0;
    for (int q = 0; q < batchQ; ++q) {
        sum_active += h_active_counts[q];
        max_active = std::max(max_active, h_active_counts[q]);
    }
    double avg_active = (double)sum_active / (double)batchQ;

    std::cout << "Active clusters: avg=" << std::fixed << std::setprecision(2) << avg_active
              << " max=" << max_active << " (cap=" << MAX_ACTIVE << ")\n";

    std::vector<int> h_idx((size_t)k);
    std::vector<float> h_dist((size_t)k);
    CUDA_CHECK(cudaMemcpy(h_idx.data(), d_out_idx, (size_t)k * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_out_dist, (size_t)k * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\nFirst query top-5 (packed indices):\n";
    for (int i = 0; i < std::min(5, k); ++i) {
        std::cout << "  #" << i << " idx=" << h_idx[i] << " dist=" << std::fixed << std::setprecision(6) << h_dist[i] << "\n";
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_radii));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_sizes));

    CUDA_CHECK(cudaFree(d_dist2qc));
    CUDA_CHECK(cudaFree(d_seed));
    CUDA_CHECK(cudaFree(d_dk));
    CUDA_CHECK(cudaFree(d_active_counts));
    CUDA_CHECK(cudaFree(d_active_ids));
    CUDA_CHECK(cudaFree(d_out_idx));
    CUDA_CHECK(cudaFree(d_out_dist));

    return 0;
}
