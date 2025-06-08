#include "../include/kernels.h"
#include "../include/types.h"
#include "../config.h"

// ============================================================================
// REDUCTION KERNELS
// ============================================================================

// Energy sum kernel with efficient block-level reduction
__global__ void energy_sum_kernel(float *p, double *partial_sums, int N_PART, double M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ double shared_sums[BLOCK_SIZE];
    
    // Each thread computes local kinetic energy - convert to double for accuracy
    double local_sum = 0.0;
    if (idx < N_PART) {
        double p_double = (double)p[idx];
        local_sum = p_double * p_double;  // Kinetic energy proportional to p^2
    }
    shared_sums[tid] = local_sum;
    
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sums[0] / (2 * M);  // Convert to energy units
    }
}

// Chi-square reduction kernel for histogram validation
__global__ void chi2_reduction_kernel(int *h, int *g, double *DxE, double *DpE, 
                                     double *chi2_results, int BINS, int is_initial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ double shared_chi2[BLOCK_SIZE];
    shared_chi2[tid] = 0.0;
    
    // Different chi2 calculations based on simulation phase
    if (is_initial) {
        // Initial phase chi2 calculation for position histogram
        if (idx >= BINS && idx < 2 * BINS) {
            double expected = 2 * DxE[idx];
            if (expected > 0) {
                double diff = h[idx] - expected;
                shared_chi2[tid] = (diff * diff) / expected;
            }
        }
    } else {
        // Regular phase chi2 calculation
        if (idx >= 4 && idx < 2 * BINS) {
            double expected = DxE[idx];
            if (expected > 0) {
                double diff = h[idx] - expected;
                shared_chi2[tid] = (diff * diff) / expected;
            }
        }
    }
    
    __syncthreads();
    
    // Parallel reduction for chi-square sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_chi2[tid] += shared_chi2[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        chi2_results[blockIdx.x] = shared_chi2[0];
    }
}

// Advanced reduction kernel with warp shuffle optimization
__global__ void energy_sum_kernel_optimized(float *p, double *partial_sums, int N_PART, double M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ double shared_sums[BLOCK_SIZE / 32]; // Only need one value per warp
    
    // Each thread computes local sum
    double local_sum = 0.0;
    for (int i = idx; i < N_PART; i += blockDim.x * gridDim.x) {
        double p_double = (double)p[i];
        local_sum += p_double * p_double;
    }
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (tid % 32 == 0) {
        shared_sums[tid / 32] = local_sum;
    }
    
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < 32 && tid < blockDim.x / 32) {
        local_sum = shared_sums[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        
        if (tid == 0) {
            partial_sums[blockIdx.x] = local_sum / (2 * M);
        }
    }
}

// Generic reduction kernel for various statistics
template<typename T, typename Op>
__global__ void generic_reduction_kernel(T *input, T *output, int N, T identity, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ T shared_data[BLOCK_SIZE];
    
    // Load data with bounds checking
    T local_value = identity;
    if (idx < N) {
        local_value = input[idx];
    }
    shared_data[tid] = local_value;
    
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = op(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
} 