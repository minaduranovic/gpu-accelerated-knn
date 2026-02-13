# v1

Verzija podijeljena na:
- `src/main.cu` (main + pipeline orkestracija)
- `include/kernels.h` (CUDA kerneli)
- `include/synthetic_data.h` (CPU generacija sintetičkih podataka)
- `include/common.h` (zajednički include-i, makroi i util funkcije)
- `src/kmeans_cpu.cpp` + `include/kmeans_cpu.h` (offline indeksiranje)

Build:

```bash
nvcc -O3 -std=c++17 v1/src/main.cu v1/src/kmeans_cpu.cpp -I v1/include -o v1/main
```
