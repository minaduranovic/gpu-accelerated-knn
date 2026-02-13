# v2

Verzija podijeljena na:
- `src/main.cu` (main + CLI parametri)
- `include/kernels.h` (CUDA kerneli, SoA + warp-reduce)
- `include/synthetic_data.h` (CPU generacija sintetičkih podataka)
- `include/common.h` (zajednički include-i, makroi i util funkcije)
- `src/kmeans_cpu.cpp` + `include/kmeans_cpu.h` (offline indeksiranje)
- `tests/run_scenarios.sh` (pripadajući test scenariji)

Buld:

```bash
nvcc -O3 -std=c++17 v2/src/main.cu v2/src/kmeans_cpu.cpp -I v2/include -o v2/main
```

Testovi:

```bash
v2/tests/run_scenarios.sh ./v2/main
```
