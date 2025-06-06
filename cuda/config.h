#ifndef CONFIG_H
#define CONFIG_H

// CUDA Configuration
#define BLOCK_SIZE 256
#define GRID_SIZE(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)

// Memory optimization settings
#define USE_SHARED_MEMORY 1
#define SHARED_HISTOGRAM_SIZE 512

// Precision settings (keeping double for now as requested)
typedef double real_t;

// Performance monitoring
#define ENABLE_TIMING 1
#define ENABLE_MEMORY_TRACKING 1

// Debug settings
#ifdef DEBUG
#define CUDA_DEBUG_SYNC() cudaDeviceSynchronize()
#else
#define CUDA_DEBUG_SYNC()
#endif

// Compilation flags for different GPU architectures
// This should be set in Makefile based on target GPU
#ifndef GPU_ARCH
#define GPU_ARCH 60  // Default to sm_60 (Pascal)
#endif

// Random number generation settings
#define RNG_SEED_OFFSET 12345
#define RNG_THREADS_PER_BLOCK BLOCK_SIZE

#endif 