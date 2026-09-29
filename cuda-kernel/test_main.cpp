#include <dlfcn.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define WARMUP_TESTS 5000
#define TARGET_RUNTIME_SEC 10.0

// Define target library interface
typedef void (*libafl_target_t)(uint8_t *, size_t);
typedef void (*libafl_wait_t)();

static void append_le_u64(std::vector<uint8_t> &out, uint64_t value) {
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    out.push_back((value >> (i * 8)) & 0xFF);
  }
}

static void append_le_size_t(std::vector<uint8_t> &out, size_t value) {
  for (size_t i = 0; i < sizeof(size_t); ++i) {
    out.push_back((value >> (i * 8)) & 0xFF);
  }
}

static size_t align_up(size_t value, size_t alignment) {
  if (alignment <= 1) {
    return value;
  }
  size_t rem = value % alignment;
  return rem == 0 ? value : value + (alignment - rem);
}

static std::vector<uint8_t> make_arg_pack_v1_payload(const uint8_t *raw,
                                                      size_t raw_size) {
  size_t split = raw_size / 2;
  const uint8_t *input = raw;
  size_t input_len = split;
  const uint8_t *output = raw + split;
  size_t output_len = raw_size - split;
  size_t kernel_size = (input_len < output_len) ? input_len : output_len;

  const size_t output_align = 4;
  const size_t size_align = sizeof(size_t);
  // Second length field must be at 8-byte aligned offset, so
  // input_encoded_len must be a multiple of 8.  Output data must also
  // satisfy output_align, so align to max(8, output_align).
  const size_t seg1_content_align =
      (sizeof(uint64_t) > output_align) ? sizeof(uint64_t) : output_align;

  size_t output_start_unaligned = sizeof(uint64_t) + input_len + sizeof(uint64_t);
  size_t output_start_aligned = align_up(output_start_unaligned, seg1_content_align);
  size_t input_encoded_len = output_start_aligned - sizeof(uint64_t) - sizeof(uint64_t);

  size_t size_start_unaligned = output_start_aligned + output_len;
  size_t size_start_aligned = align_up(size_start_unaligned, size_align);
  size_t output_encoded_len = size_start_aligned - output_start_aligned;

  std::vector<uint8_t> packed;
  packed.reserve(sizeof(uint64_t) + input_encoded_len + sizeof(uint64_t) +
                 output_encoded_len + sizeof(size_t));

  append_le_u64(packed, input_encoded_len);
  packed.insert(packed.end(), input, input + input_len);
  while (packed.size() < sizeof(uint64_t) + input_encoded_len) {
    packed.push_back(0);
  }

  append_le_u64(packed, output_encoded_len);
  packed.insert(packed.end(), output, output + output_len);
  while (packed.size() < sizeof(uint64_t) + input_encoded_len +
                             sizeof(uint64_t) + output_encoded_len) {
    packed.push_back(0);
  }

  append_le_size_t(packed, kernel_size);
  return packed;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr,
            "Usage: %s <libphase2_origin_target.so|libphase2_rapid_target.so|librapid2_target.so>\n",
            argv[0]);
    return 1;
  }

  // Open shared library
  char lib_path[256];
  snprintf(lib_path, sizeof(lib_path), "./%s", argv[1]);
  void *lib = dlopen(lib_path, RTLD_LAZY);
  if (!lib) {
    fprintf(stderr, "Error loading library %s: %s\n", lib_path, dlerror());
    return 1;
  }

  // Load required symbols
  libafl_target_t libafl_target = (libafl_target_t)dlsym(lib, "libafl_target");
  uint8_t *libafl_cov_map = (uint8_t *)dlsym(lib, "libafl_cov_map");

  if (!libafl_target || !libafl_cov_map) {
    fprintf(stderr, "Error loading required symbols: %s\n", dlerror());
    dlclose(lib);
    return 1;
  }

  // Load optional symbol
  libafl_wait_t libafl_wait = (libafl_wait_t)dlsym(lib, "libafl_wait");

  printf("== Performance Test (%s) ==\n", argv[1]);
  printf("libafl_wait: %s\n", libafl_wait ? "Available" : "Not available");

  // Create test input
  const size_t test_size = 4000;
  uint8_t *test_input = (uint8_t *)malloc(test_size);
  memset(test_input, 0x42, test_size);
  auto packed_input = make_arg_pack_v1_payload(test_input, test_size);
  /* init env */
  libafl_target(packed_input.data(), packed_input.size());
  if (libafl_wait) {
    libafl_wait();
  }
  // Warmup and estimate timing
  printf("\nPhase 1: Warmup and calibration (%d iterations)...\n",
         WARMUP_TESTS);
  auto warmup_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < WARMUP_TESTS; ++i) {
    libafl_target(packed_input.data(), packed_input.size());
  }

  if (libafl_wait) {
    libafl_wait();
  }

  auto warmup_end = std::chrono::high_resolution_clock::now();
  double warmup_duration =
      std::chrono::duration<double>(warmup_end - warmup_start).count();

  printf("  Warmup completed in %.3f seconds\n", warmup_duration);
  printf("  Warmup speed: %.0f ops/sec\n", WARMUP_TESTS / warmup_duration);

  // Calculate iterations for target runtime
  long long target_iterations =
      (long long)(TARGET_RUNTIME_SEC * WARMUP_TESTS / warmup_duration);
  if (target_iterations < 1000)
    target_iterations = 1000; // Minimum iterations

  printf("\nPhase 2: Performance test\n");
  printf("  Target runtime: %.1f seconds\n", TARGET_RUNTIME_SEC);
  printf("  Estimated iterations: %lld\n", target_iterations);

  // Main performance test
  printf("  Running...\n");
  auto test_start = std::chrono::high_resolution_clock::now();

  for (long long i = 0; i < target_iterations; ++i) {
    libafl_target(packed_input.data(), packed_input.size());
  }

  if (libafl_wait) {
    libafl_wait();
  }

  auto test_end = std::chrono::high_resolution_clock::now();
  double test_duration =
      std::chrono::duration<double>(test_end - test_start).count();

  // Calculate and print results
  double ops_per_second = target_iterations / test_duration;

  printf("\n== Performance Results ==\n");
  printf("  Library: %s\n", argv[1]);
  printf("  Total iterations: %lld\n", target_iterations);
  printf("  Total time: %.3f seconds\n", test_duration);
  printf("  Performance: %.0f ops/sec\n", ops_per_second);
  printf("  Average latency: %.3f us/op\n",
         test_duration * 1000000.0 / target_iterations);

  // Coverage summary
  printf("\nCoverage Summary:\n");
  int total_edges = 0;
  for (int i = 0; i < 64; ++i) {
    if (libafl_cov_map[i] > 0)
      total_edges++;
  }
  printf("  Total edges covered: %d\n", total_edges);

  // Cleanup
  free(test_input);
  dlclose(lib);

  return 0;
}
