#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kLineSizeBytes = 128;
constexpr int kElementSizeBytes = sizeof(int);
constexpr int kElementsPerLine = kLineSizeBytes / kElementSizeBytes;
constexpr const char *kDefaultMetadataPath = "./ima_tiny_case_metadata.json";

struct options_t {
  int ctas = 1;
  int warps_per_cta = 8;
  int warmup_iters = 256;
  int ima_iters = 4096;
  int tail_iters = 256;
  int data_lines = 4096;
  int mapping_seed = 1;
  int compute_gap = 8;
  std::string metadata_out = kDefaultMetadataPath;
};

struct device_buffers_t {
  int *warmup = nullptr;
  int *index = nullptr;
  int *data = nullptr;
  int *tail = nullptr;
  int *output = nullptr;
};

int parse_int_arg(const char *flag, const char *value);

const char *read_env(const char *name) { return std::getenv(name); }

void apply_env_override(const char *name, int *target) {
  const char *value = read_env(name);
  if (value == nullptr || *value == '\0') {
    return;
  }
  *target = parse_int_arg(name, value);
}

void apply_env_override(const char *name, std::string *target) {
  const char *value = read_env(name);
  if (value == nullptr || *value == '\0') {
    return;
  }
  *target = value;
}

void usage(const char *prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "  --ctas N\n"
      << "  --warps-per-cta N\n"
      << "  --warmup-iters N\n"
      << "  --ima-iters N\n"
      << "  --tail-iters N\n"
      << "  --data-lines N\n"
      << "  --mapping-seed N\n"
      << "  --compute-gap N\n"
      << "  --metadata-out PATH\n";
}

int parse_int_arg(const char *flag, const char *value) {
  if (value == nullptr) {
    throw std::runtime_error(std::string("missing value for ") + flag);
  }
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') {
    throw std::runtime_error(std::string("invalid integer for ") + flag + ": " +
                             value);
  }
  return static_cast<int>(parsed);
}

options_t parse_args(int argc, char **argv) {
  options_t opts;
  apply_env_override("IMA_TINY_CTAS", &opts.ctas);
  apply_env_override("IMA_TINY_WARPS_PER_CTA", &opts.warps_per_cta);
  apply_env_override("IMA_TINY_WARMUP_ITERS", &opts.warmup_iters);
  apply_env_override("IMA_TINY_IMA_ITERS", &opts.ima_iters);
  apply_env_override("IMA_TINY_TAIL_ITERS", &opts.tail_iters);
  apply_env_override("IMA_TINY_DATA_LINES", &opts.data_lines);
  apply_env_override("IMA_TINY_MAPPING_SEED", &opts.mapping_seed);
  apply_env_override("IMA_TINY_COMPUTE_GAP", &opts.compute_gap);
  apply_env_override("IMA_TINY_METADATA_OUT", &opts.metadata_out);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--ctas") {
      opts.ctas = parse_int_arg("--ctas", argv[++i]);
    } else if (arg == "--warps-per-cta") {
      opts.warps_per_cta = parse_int_arg("--warps-per-cta", argv[++i]);
    } else if (arg == "--warmup-iters") {
      opts.warmup_iters = parse_int_arg("--warmup-iters", argv[++i]);
    } else if (arg == "--ima-iters") {
      opts.ima_iters = parse_int_arg("--ima-iters", argv[++i]);
    } else if (arg == "--tail-iters") {
      opts.tail_iters = parse_int_arg("--tail-iters", argv[++i]);
    } else if (arg == "--data-lines") {
      opts.data_lines = parse_int_arg("--data-lines", argv[++i]);
    } else if (arg == "--mapping-seed") {
      opts.mapping_seed = parse_int_arg("--mapping-seed", argv[++i]);
    } else if (arg == "--compute-gap") {
      opts.compute_gap = parse_int_arg("--compute-gap", argv[++i]);
    } else if (arg == "--metadata-out") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --metadata-out");
      }
      opts.metadata_out = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (opts.ctas <= 0 || opts.warps_per_cta <= 0 || opts.warmup_iters < 0 ||
      opts.ima_iters <= 0 || opts.tail_iters < 0 || opts.data_lines <= 0 ||
      opts.compute_gap < 0) {
    throw std::runtime_error("all sizes must be positive, except warmup/tail/compute-gap can be zero");
  }
  if (opts.warps_per_cta * kWarpSize > 1024) {
    throw std::runtime_error("warps-per-cta exceeds 1024-thread block limit");
  }
  return opts;
}

inline void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::ostringstream oss;
    oss << what << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(oss.str());
  }
}

int gcd_int(int a, int b) {
  while (b != 0) {
    int t = a % b;
    a = b;
    b = t;
  }
  return a < 0 ? -a : a;
}

int choose_step(int data_lines, int mapping_seed) {
  int step = (mapping_seed * 2) + 1;
  while (gcd_int(step, data_lines) != 1) {
    step += 2;
  }
  return step;
}

std::string hex_string(std::uintptr_t value) {
  std::ostringstream oss;
  oss << "0x" << std::hex << value;
  return oss.str();
}

unsigned long long checksum_vector(const std::vector<int> &values) {
  unsigned long long checksum = 0;
  for (int value : values) {
    checksum += static_cast<unsigned long long>(static_cast<uint32_t>(value));
  }
  return checksum;
}

void write_metadata(const options_t &opts, const device_buffers_t &bufs,
                    std::size_t total_threads, std::size_t warmup_count,
                    std::size_t index_count, std::size_t data_count,
                    std::size_t tail_count, int line_step_lane,
                    int line_step_warp, int line_step_iter,
                    int element_step,
                    unsigned long long warmup_checksum,
                    unsigned long long index_checksum,
                    unsigned long long data_checksum,
                    unsigned long long tail_checksum,
                    unsigned long long output_checksum) {
  std::ofstream out(opts.metadata_out);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open metadata file: " + opts.metadata_out);
  }

  auto emit_array = [&](const char *name, const int *ptr, std::size_t count) {
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(ptr);
    const std::size_t size_bytes = count * sizeof(int);
    out << "  \"" << name << "\": {\n"
        << "    \"device_base_hex\": \"" << hex_string(begin) << "\",\n"
        << "    \"bytes\": " << static_cast<unsigned long long>(size_bytes) << ",\n"
        << "    \"element_bytes\": " << kElementSizeBytes << "\n"
        << "  }";
  };

  out << "{\n";
  out << "  \"benchmark\": {\n";
  out << "    \"name\": \"ima_tiny_case\",\n";
  out << "    \"version\": 1\n";
  out << "  },\n";
  out << "  \"args\": {\n";
  out << "    \"ctas\": " << opts.ctas << ",\n";
  out << "    \"warps_per_cta\": " << opts.warps_per_cta << ",\n";
  out << "    \"warmup_iters\": " << opts.warmup_iters << ",\n";
  out << "    \"ima_iters\": " << opts.ima_iters << ",\n";
  out << "    \"tail_iters\": " << opts.tail_iters << ",\n";
  out << "    \"data_lines\": " << opts.data_lines << ",\n";
  out << "    \"mapping_seed\": " << opts.mapping_seed << ",\n";
  out << "    \"compute_gap\": " << opts.compute_gap << ",\n";
  out << "    \"metadata_out\": \"" << opts.metadata_out << "\"\n";
  out << "  },\n";
  out << "  \"launch\": {\n";
  out << "    \"ctas\": " << opts.ctas << ",\n";
  out << "    \"warps_per_cta\": " << opts.warps_per_cta << ",\n";
  out << "    \"threads_per_cta\": " << opts.warps_per_cta * kWarpSize << ",\n";
  out << "    \"total_threads\": " << static_cast<unsigned long long>(total_threads)
      << "\n";
  out << "  },\n";
  out << "  \"phases\": {\n";
  out << "    \"warmup\": " << opts.warmup_iters << ",\n";
  out << "    \"ima\": " << opts.ima_iters << ",\n";
  out << "    \"tail\": " << opts.tail_iters << "\n";
  out << "  },\n";
  out << "  \"arrays\": {\n";
  emit_array("warmup", bufs.warmup, warmup_count);
  out << ",\n";
  emit_array("index", bufs.index, index_count);
  out << ",\n";
  emit_array("data", bufs.data, data_count);
  out << ",\n";
  emit_array("tail", bufs.tail, tail_count);
  out << ",\n";
  emit_array("output", bufs.output, total_threads);
  out << "\n  },\n";
  out << "  \"mapping\": {\n";
  out << "    \"seed\": " << opts.mapping_seed << ",\n";
  out << "    \"data_lines\": " << opts.data_lines << ",\n";
  out << "    \"elements_per_line\": " << kElementsPerLine << ",\n";
  out << "    \"compute_gap\": " << opts.compute_gap << ",\n";
  out << "    \"line_bytes\": " << kLineSizeBytes << ",\n";
  out << "    \"type\": \"deterministic_scatter_mix\",\n";
  out << "    \"line_step_lane\": " << line_step_lane << ",\n";
  out << "    \"line_step_warp\": " << line_step_warp << ",\n";
  out << "    \"line_step_iter\": " << line_step_iter << ",\n";
  out << "    \"element_step\": " << element_step << "\n";
  out << "  },\n";
  out << "  \"checksums\": {\n";
  out << "    \"warmup_sum_u64\": " << warmup_checksum << ",\n";
  out << "    \"index_sum_u64\": " << index_checksum << ",\n";
  out << "    \"data_sum_u64\": " << data_checksum << ",\n";
  out << "    \"tail_sum_u64\": " << tail_checksum << ",\n";
  out << "    \"output_sum_u64\": " << output_checksum << "\n";
  out << "  }\n";
  out << "}\n";
}

__global__ void ima_tiny_case_kernel(const int *warmup, const int *index,
                                     const int *data, const int *tail,
                                     int *output, int warmup_iters,
                                     int ima_iters, int tail_iters,
                                     int total_threads, int warmup_count,
                                     int tail_count, int compute_gap) {
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_launch_threads = gridDim.x * blockDim.x;
  if (global_tid >= total_threads) return;

  int acc = global_tid;

  #pragma unroll 1
  for (int iter = 0; iter < warmup_iters; ++iter) {
    const int pos = (iter * total_threads + global_tid) % warmup_count;
    acc += warmup[pos];
    acc ^= (acc << 5);
  }

  #pragma unroll 1
  for (int iter = 0; iter < ima_iters; ++iter) {
    const int pos = iter * total_threads + global_tid;
    const int indirect = index[pos];
    acc += data[indirect];
    for (int gap = 0; gap < compute_gap; ++gap) {
      acc = acc * 1664525 + 1013904223;
    }
  }

  #pragma unroll 1
  for (int iter = 0; iter < tail_iters; ++iter) {
    const int pos = (iter * total_threads + global_tid) % tail_count;
    acc += tail[pos];
    acc ^= (acc >> 3);
  }

  output[global_tid] = acc + total_launch_threads;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const options_t opts = parse_args(argc, argv);
    const std::size_t threads_per_cta =
        static_cast<std::size_t>(opts.warps_per_cta) * kWarpSize;
    const std::size_t total_threads =
        static_cast<std::size_t>(opts.ctas) * threads_per_cta;
    const std::size_t warmup_count =
        std::max<std::size_t>(total_threads * std::max(opts.warmup_iters, 1), 1024);
    const std::size_t index_count =
        std::max<std::size_t>(total_threads * static_cast<std::size_t>(opts.ima_iters), 1);
    const std::size_t data_count =
        std::max<std::size_t>(static_cast<std::size_t>(opts.data_lines) * kElementsPerLine, 1);
    const std::size_t tail_count =
        std::max<std::size_t>(total_threads * std::max(opts.tail_iters, 1), 1024);

    std::vector<int> warmup_h(warmup_count);
    std::vector<int> index_h(index_count);
    std::vector<int> data_h(data_count);
    std::vector<int> tail_h(tail_count);
    std::vector<int> output_h(total_threads, 0);

    for (std::size_t i = 0; i < warmup_h.size(); ++i) {
      warmup_h[i] = static_cast<int>((i * 17 + 3) & 0x7fff);
    }
    for (std::size_t i = 0; i < tail_h.size(); ++i) {
      tail_h[i] = static_cast<int>((i * 29 + 7) & 0x7fff);
    }
    for (std::size_t i = 0; i < data_h.size(); ++i) {
      data_h[i] = static_cast<int>((i * 1315423911ULL + 0x9e3779b9ULL) & 0x7fffffff);
    }

    const int line_step_lane = choose_step(opts.data_lines, opts.mapping_seed);
    const int line_step_warp = choose_step(opts.data_lines, opts.mapping_seed * 3 + 1);
    const int line_step_iter = choose_step(opts.data_lines, opts.mapping_seed * 5 + 3);
    const int element_step = choose_step(kElementsPerLine, opts.mapping_seed * 7 + 5);
    for (std::size_t pos = 0; pos < index_h.size(); ++pos) {
      const int iter = static_cast<int>(pos / total_threads);
      const int thread = static_cast<int>(pos % total_threads);
      const int warp = thread / kWarpSize;
      const int lane = thread % kWarpSize;
      const int line =
          (opts.mapping_seed + lane * line_step_lane + warp * line_step_warp +
           iter * line_step_iter) %
          opts.data_lines;
      const int element =
          (opts.mapping_seed + lane + warp * 3 + iter * element_step) %
          kElementsPerLine;
      index_h[pos] = line * kElementsPerLine + element;
    }

    device_buffers_t bufs;
    check_cuda(cudaMalloc(&bufs.warmup, warmup_h.size() * sizeof(int)), "cudaMalloc(warmup)");
    check_cuda(cudaMalloc(&bufs.index, index_h.size() * sizeof(int)), "cudaMalloc(index)");
    check_cuda(cudaMalloc(&bufs.data, data_h.size() * sizeof(int)), "cudaMalloc(data)");
    check_cuda(cudaMalloc(&bufs.tail, tail_h.size() * sizeof(int)), "cudaMalloc(tail)");
    check_cuda(cudaMalloc(&bufs.output, output_h.size() * sizeof(int)), "cudaMalloc(output)");

    check_cuda(cudaMemcpy(bufs.warmup, warmup_h.data(), warmup_h.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(warmup)");
    check_cuda(cudaMemcpy(bufs.index, index_h.data(), index_h.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(index)");
    check_cuda(cudaMemcpy(bufs.data, data_h.data(), data_h.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(data)");
    check_cuda(cudaMemcpy(bufs.tail, tail_h.data(), tail_h.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(tail)");

    dim3 grid(opts.ctas);
    dim3 block(static_cast<unsigned>(threads_per_cta));
    ima_tiny_case_kernel<<<grid, block>>>(
        bufs.warmup, bufs.index, bufs.data, bufs.tail, bufs.output,
        opts.warmup_iters, opts.ima_iters, opts.tail_iters,
        static_cast<int>(total_threads), static_cast<int>(warmup_h.size()),
        static_cast<int>(tail_h.size()), opts.compute_gap);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check_cuda(cudaMemcpy(output_h.data(), bufs.output,
                          output_h.size() * sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(output)");

    const unsigned long long output_checksum = checksum_vector(output_h);

    write_metadata(opts, bufs, total_threads, warmup_h.size(), index_h.size(),
                   data_h.size(), tail_h.size(), line_step_lane,
                   line_step_warp, line_step_iter, element_step,
                   checksum_vector(warmup_h), checksum_vector(index_h),
                   checksum_vector(data_h), checksum_vector(tail_h),
                   output_checksum);

    std::cout << "ima_tiny_case completed"
              << " ctas=" << opts.ctas
              << " warps_per_cta=" << opts.warps_per_cta
              << " total_threads=" << total_threads
              << " warmup_iters=" << opts.warmup_iters
              << " ima_iters=" << opts.ima_iters
              << " tail_iters=" << opts.tail_iters
              << " data_lines=" << opts.data_lines
              << " mapping_seed=" << opts.mapping_seed
              << " compute_gap=" << opts.compute_gap
              << " checksum=" << output_checksum
              << " metadata=" << opts.metadata_out << "\n";

    cudaFree(bufs.warmup);
    cudaFree(bufs.index);
    cudaFree(bufs.data);
    cudaFree(bufs.tail);
    cudaFree(bufs.output);
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "ima_tiny_case error: " << ex.what() << "\n";
    return 1;
  }
}
