#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

#define CUDA_CHECK(err)                                                   \
  {                                                                       \
    if (err != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorName(err)); \
    }                                                                     \
  }

float ProfileCopyH2D(void* dev, void* host, std::size_t size) {
  cudaEvent_t start, end;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaEventRecord(start, 0));
  CUDA_CHECK(cudaMemcpy(dev, host, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));

  float time;  // ms
  CUDA_CHECK(cudaEventElapsedTime(&time, start, end));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  return time;
}

float ProfileCopyD2D(void* dst, void* src, std::size_t size) {
  cudaEvent_t start, end;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaEventRecord(start, 0));
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));

  float time;  // ms
  CUDA_CHECK(cudaEventElapsedTime(&time, start, end));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  return time;
}

float ProfileCopyP2P(void* dst, int dst_device, void* src, int src_device,
                     std::size_t size) {
  cudaEvent_t start, end;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaEventRecord(start, 0));
  CUDA_CHECK(cudaMemcpyPeer(dst, dst_device, src, src_device, size));
  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));

  float time;  // ms
  CUDA_CHECK(cudaEventElapsedTime(&time, start, end));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  return time;
}

void GPUBandwidthBenchmark(std::size_t size) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("Device: %s\n", prop.name);

  float *host_pageable, *host_pinned;
  float *dev1, *dev2, *dev3;
  int dst_device = 0, src_device = 1;

  host_pageable = new float[size / sizeof(float)];
  CUDA_CHECK(cudaSetDevice(dst_device));
  CUDA_CHECK(cudaMallocHost(&host_pinned, size));
  CUDA_CHECK(cudaMalloc(&dev1, size));
  CUDA_CHECK(cudaMalloc(&dev2, size));

  float time;
  time = ProfileCopyH2D(dev1, host_pageable, size);
  printf("\nHost to Device (Pageable) bandwidth: %.2f (GB/s)\n",
         size / 1e6 / time);

  time = ProfileCopyH2D(dev1, host_pinned, size);
  printf("\nHost to Device (Pinned) bandwidth: %.2f (GB/s)\n",
         size / 1e6 / time);

  time = ProfileCopyD2D(dev1, dev2, size);
  printf("\nDevice memory bandwidth: %.2f (GB/s)\n", size / 1e6 / time);

  CUDA_CHECK(cudaSetDevice(src_device));
  CUDA_CHECK(cudaMalloc(&dev3, size));

  time = ProfileCopyP2P(dev1, dst_device, dev3, src_device, size);
  printf("\nDevice to Device (GPU0-GPU1) bandwidth: %.2f (GB/s)\n",
         size / 1e6 / time);

  CUDA_CHECK(cudaFree(dev3));
  CUDA_CHECK(cudaSetDevice(dst_device));

  delete[] host_pageable;
  CUDA_CHECK(cudaFreeHost(host_pinned));
  CUDA_CHECK(cudaFree(dev1));
  CUDA_CHECK(cudaFree(dev2));
}
