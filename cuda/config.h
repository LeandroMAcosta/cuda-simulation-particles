#ifndef CONFIG_H
#define CONFIG_H

// ============================================================================
// CUDA OPTIMIZATION CONFIGURATION
// ============================================================================

// Kernel optimization strategy selection
#define USE_KERNEL_V2 1         // Highly optimized with fast math


// ============================================================================
// CUDA KERNEL CONFIGURATION
// ============================================================================

// Default CUDA configuration
#define BLOCK_SIZE 256
#define MAX_GRID_SIZE 65535
#define GRID_SIZE(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)

// Memory alignment
#define MEMORY_ALIGNMENT 256

// Random number generation
#define RNG_SEED_OFFSET 12345

// ============================================================================
// PERFORMANCE MONITORING
// ============================================================================


// Debug settings
#ifdef DEBUG
#define CUDA_DEBUG_SYNC() cudaDeviceSynchronize()
#else
#define CUDA_DEBUG_SYNC()
#endif

// Compilation flags for different GPU architectures
// This should be set in Makefile based on target GPU
#ifndef GPU_ARCH
#define GPU_ARCH 75
#endif

#endif 