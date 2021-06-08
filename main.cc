#include <cstdio>
#include <cstring>
#include <string>

extern void GPUBandwidthBenchmark(std::size_t);
extern void CPUBandwidthBenchmark(std::size_t);

int main(int argc, char* argv[]) {
  std::size_t size = 100;
  if (argc == 2) {
    size = std::stol(argv[1]);  // MB
  }

  printf("Transfer Size = %lu MB\n", size);

  size *= 1e6;
  GPUBandwidthBenchmark(size);
  CPUBandwidthBenchmark(size);
  return 0;
}
