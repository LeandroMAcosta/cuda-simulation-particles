#include "../include/kernels.h"
#include "../include/types.h"
#include "../config.h"

// ============================================================================
// HISTOGRAM KERNELS
// ============================================================================

// Clear histogram arrays before computation
__global__ void clear_histograms(int *h, int *g, int *hg, int BINS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Clear position histogram h
    int h_size = 2 * BINS + 4;
    if (idx < h_size) {
        h[idx] = 0;
    }
    
    // Clear momentum histogram g  
    int g_size = 2 * BINS;
    if (idx < g_size) {
        g[idx] = 0;
    }
    
    // Clear combined histogram hg
    int hg_size = h_size * g_size;
    if (idx < hg_size) {
        hg[idx] = 0;
    }
}

// Compute histograms from particle data using atomic operations
__global__ void compute_histograms(float *x, float *p, int *h, int *g, int *hg, 
                                  int N_PART, int BINS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_PART) return;
    
    // Calculate histogram bin indices
    // Position histogram: map [-0.5, 0.5] to histogram bins
    int h_idx = floorf((x[idx] + 0.5f) * (1.99999999999999f * BINS) + 2.0f);
    
    // Momentum histogram: map momentum range to bins  
    int g_idx = floorf((p[idx] / 3.0e-23f + 1.0f) * (0.999999999999994f * BINS));
    
    // Bounds checking to prevent array overflows
    h_idx = max(0, min(h_idx, 2 * BINS + 3));
    g_idx = max(0, min(g_idx, 2 * BINS - 1));
    
    // Combined histogram index
    int hg_idx = (2 * BINS) * h_idx + g_idx;
    
    // Thread-safe atomic increments for histogram accumulation
    atomicAdd(&h[h_idx], 1);
    atomicAdd(&g[g_idx], 1);
    atomicAdd(&hg[hg_idx], 1);
}

// Optimized histogram computation with shared memory for better performance
__global__ void compute_histograms_shared(float *x, float *p, int *h, int *g, int *hg, 
                                         int N_PART, int BINS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory histograms for this block
    extern __shared__ int shared_hist[];
    int *sh_h = shared_hist;
    int *sh_g = &shared_hist[2 * BINS + 4];
    
    // Initialize shared memory
    if (tid < 2 * BINS + 4) {
        sh_h[tid] = 0;
    }
    if (tid < 2 * BINS) {
        sh_g[tid] = 0;
    }
    __syncthreads();
    
    // Compute histogram for this particle
    if (idx < N_PART) {
        // Calculate bin indices
        int h_idx = floorf((x[idx] + 0.5f) * (1.99999999999999f * BINS) + 2.0f);
        int g_idx = floorf((p[idx] / 3.0e-23f + 1.0f) * (0.999999999999994f * BINS));
        
        // Bounds checking
        h_idx = max(0, min(h_idx, 2 * BINS + 3));
        g_idx = max(0, min(g_idx, 2 * BINS - 1));
        
        // Atomic increment in shared memory
        atomicAdd(&sh_h[h_idx], 1);
        atomicAdd(&sh_g[g_idx], 1);
    }
    
    __syncthreads();
    
    // Copy shared memory results to global memory
    if (tid < 2 * BINS + 4) {
        atomicAdd(&h[tid], sh_h[tid]);
    }
    if (tid < 2 * BINS) {
        atomicAdd(&g[tid], sh_g[tid]);
    }
} 