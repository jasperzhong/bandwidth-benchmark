#include <chrono>
#include <cstdio>
#include <cstring>

void Memcpy(void* dst, void* src, std::size_t size) {
  float* out = static_cast<float*>(dst);
  float* in = static_cast<float*>(src);
#pragma omp parallel for
  for (std::size_t i = 0; i < size / 4; ++i) {
    out[i] = in[i];
  }

  if (size % 4) {
    std::memcpy(out + size / 4, in + size / 4, size % 4);
  }
}

float ProfileCopyH2H(void* dst, void* src, std::size_t size) {
  auto start = std::chrono::steady_clock::now();
  Memcpy(dst, src, size);
  auto end = std::chrono::steady_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void CPUBandwidthBenchmark(std::size_t size) {
  float *dst, *src;
  dst = new float[size / sizeof(float)];
  src = new float[size / sizeof(float)];

  float time = ProfileCopyH2H(dst, src, size);
  printf("\nCPU memory bandwidth: %.2f (GB/s)\n", size / 1e6 / time);

  delete[] dst;
  delete[] src;
}
