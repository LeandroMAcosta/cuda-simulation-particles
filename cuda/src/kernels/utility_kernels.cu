#include "../include/kernels.h"
#include "../include/types.h"
#include "../config.h"

// ============================================================================
// UTILITY KERNELS
// ============================================================================

// Device random number generation using cuRAND - single precision
__device__ float d_curand_float(curandState *state) {
    return curand_uniform(state);
}

// Device random number generation using cuRAND - double precision  
__device__ double d_curand_double(curandState *state) {
    return curand_uniform_double(state);
}

// Advanced Box-Muller transform for Gaussian random numbers
__device__ float2 d_box_muller_float(curandState *state) {
    float u1 = curand_uniform(state) + 1E-35f;  // Avoid log(0)
    float u2 = curand_uniform(state);
    
    float xi1 = sqrtf(-2.0f * logf(u1));
    float xi2 = 2.0f * PI * u2;
    
    return make_float2(xi1 * cosf(xi2), xi1 * sinf(xi2));
}

// Device function for fast trigonometric calculation
__device__ void d_fast_sincos(float x, float *sin_val, float *cos_val) {
    __sincosf(x, sin_val, cos_val);
}

// Device function for safe square root with bounds checking
__device__ double d_safe_sqrt(double value) {
    return sqrt(fmax(0.0, value));
}

// Device function for clamping values to a range
__device__ double d_clamp(double value, double min_val, double max_val) {
    return fmax(min_val, fmin(max_val, value));
}

// Device function for sign-preserving operations
__device__ double d_copysign_double(double value, double sign) {
    return copysign(value, sign);
}

// Device function for efficient modular arithmetic
__device__ int d_mod_fast(int value, int divisor) {
    // Optimized modulo for small powers of 2
    if (divisor == 2) {
        return value & 1;
    }
    return value % divisor;
}

// Memory validation kernel for debugging
__global__ void validate_memory_kernel(float *data, int N, float *error_flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float value = data[idx];
    
    // Check for NaN, infinity, or extreme values
    if (isnan(value) || isinf(value) || fabsf(value) > 1e10f) {
        error_flags[idx] = 1.0f;  // Mark as error
    } else {
        error_flags[idx] = 0.0f;  // Valid value
    }
}

// Particle boundary validation kernel
__global__ void validate_particle_bounds(float *x, float *p, int N_PART, int *error_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_PART) return;
    
    int local_errors = 0;
    
    // Check position bounds [-0.502, 0.502]
    if (fabsf(x[idx]) > 0.502f) {
        local_errors++;
    }
    
    // Check momentum bounds (reasonable physical range)
    if (fabsf(p[idx]) > 1e-20f) {
        local_errors++;
    }
    
    if (local_errors > 0) {
        atomicAdd(error_count, local_errors);
    }
}

// Simple checksum kernel for data validation
__global__ void compute_checksum(float *data, int N, unsigned long long *checksum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ unsigned long long shared_sum[BLOCK_SIZE];
    
    unsigned long long local_sum = 0;
    if (idx < N) {
        // Simple hash of the float bits
        unsigned int bits = __float_as_uint(data[idx]);
        local_sum = bits;
    }
    shared_sum[tid] = local_sum;
    
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(checksum, shared_sum[0]);
    }
} 