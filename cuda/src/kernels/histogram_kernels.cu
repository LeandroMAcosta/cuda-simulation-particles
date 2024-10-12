#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>

#include "../include/constants.h"
#include "../include/histogram_kernels.h"

using namespace std;

__global__ void init_DpE_kernel(double *DpE, int N_PART, int BINS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 2 * BINS) return;

    double numerator = 6.0E-26 * N_PART;
    double denominator = 5.24684E-24 * sqrt(2.0 * M_PI);
    double exponent = -pow(3.0e-23 * (1.0 * i / BINS - 0.999) / 5.24684E-24, 2) / 2;
    DpE[i] = (numerator / denominator) * exp(exponent);
}

__global__ void init_DxE_kernel(double *DxE, int N_PART, int BINS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (BINS + 1) << 1) return;
    if (i < 2) {
        // Maybe just i == 0?
        DxE[0] = 0.0;
        DxE[1] = 0.0;
        DxE[2 * BINS + 2] = 0.0;
        DxE[2 * BINS + 3] = 0.0;
    }
    DxE[i] = 1.0E-3 * N_PART;
}

__device__ uint32_t generate_random(uint32_t base_seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = base_seed + idx + blockIdx.x + threadIdx.x;

    curandState state;
    curand_init(seed, idx, 5, &state);

    int random = curand(&state); 
    return random;
}

__device__ uint32_t xorshift32(uint32_t* seed) {
    uint32_t x = *seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *seed = x;
    return x;
}

__device__ double d_xorshift(uint32_t *seed) {
    uint32_t x = xorshift32(seed);
    return (double)x / (double)UINT32_MAX;
}

__device__ float f_xorshift(uint32_t *seed) {
    uint32_t x = xorshift32(seed);
    return (float)x / (float)UINT32_MAX;
}

__global__ void init_x_kernel(float* x, uint32_t base_seed, int N_PART) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_PART) return;

    uint32_t seed = generate_random(base_seed);
    x[idx] = f_xorshift(&seed) * 0.5f;
}

__global__ void init_p_kernel(double* d_p,  uint32_t base_seed, int N_PART) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N_PART >> 1)) return;
    
    uint32_t seed = generate_random(base_seed);

    // Generate two random values using XORShift
    double randomValue1 = (double)(xorshift32(&seed)) / UINT32_MAX;
    double randomValue2 = (double)(xorshift32(&seed)) / UINT32_MAX;

    // Box-Muller transform to generate two normally distributed random numbers
    double xi1 = sqrt(-2.0 * log(randomValue1 + 1E-35));
    double xi2 = 2.0 * M_PI * randomValue2;

    // Store the generated values in the p array
    d_p[2 * idx] = xi1 * cos(xi2) * 5.24684E-24;
    d_p[2 * idx + 1] = xi1 * sin(xi2) * 5.24684E-24;
    
}

__global__ void update_histograms_kernel(float *x, double *d_p, int *h, int *g, int *hg, int N_PART, int BINS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_PART) return;

    // Calculate histogram indices based on particle data
    int h_idx = static_cast<int>(floorf((x[idx] + 0.5f) * (1.99999999999999f * BINS) + 2.0f));
    int g_idx = static_cast<int>(floor((d_p[idx] / 3.0e-23 + 1) * (0.999999999999994 * BINS)));

    int hg_idx = (2 * BINS) * h_idx + g_idx;

    // Use atomic operations to avoid race conditions when updating shared memory
    atomicAdd(&h[h_idx], 1);
    atomicAdd(&g[g_idx], 1);
    atomicAdd(&hg[hg_idx], 1);
}

// Kernel function to update positions and momenta
__global__ void simulate_particle_motion(int number_of_steps, float *x, double *d_p, int N_PART, float DT, double M, float sigmaL, float alfa, float pmin, float pmax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_PART) return;

    uint32_t seed = idx + blockIdx.x + threadIdx.x * 31;

    // Change to float for x_tmp since x is now float
    float x_tmp = x[idx];
    double p_tmp = d_p[idx];  // Keep p_tmp as double

    int signop, k;

    // Main particle loop
    for (int step = 0; step < number_of_steps; ++step) {
        // Operations with x_tmp now involve float
        x_tmp += p_tmp * (DT / M);  // Cast DT / M to float
        // x_tmp += (float) (p_tmp * (DT / M));  // Cast DT / M to float
        signop = copysign(1.0, p_tmp);  // Keep signop as double
        k = truncf(x_tmp + 0.5f * signop);  // Use truncf for float

        if (k == 0) continue;

        float randomValue = f_xorshift(&seed);
        float xi1 = sqrtf(-2.0f * logf(randomValue + 1E-35f));
        randomValue = f_xorshift(&seed);
        float xi2 = 2.0f * (float)M_PI * randomValue;
        float deltaX = sqrtf(fabsf(k)) * xi1 * cosf((float)xi2) * sigmaL;  // Use float functions

        deltaX = (fabsf(deltaX) > 1.0f ? 1.0f * copysignf(1.0f, deltaX) : deltaX);
        x_tmp = (k % 2 ? -1.0f : 1.0f) * (x_tmp - k) + deltaX;

        if (fabsf(x_tmp) > 0.502f) {
            x_tmp = 1.004f * copysignf(1.0f, x_tmp) - x_tmp;
        }
        p_tmp = fabs(p_tmp);  // Keep p_tmp operations in double

        for (int l = 1; l <= labs(k); ++l) {
            float DeltaE = alfa * (p_tmp - pmin) * (pmax - p_tmp);
            randomValue = f_xorshift(&seed);
            double value = p_tmp * p_tmp + DeltaE * (randomValue - 0.5);
            if (value < 0) {
                value = 0;
            }
            p_tmp = sqrt(value);
        }
        p_tmp *= (k % 2 ? -1.0 : 1.0) * signop;
    }
    // Update global memory
    x[idx] = x_tmp;  // Store the float result back to x
    d_p[idx] = p_tmp;  // Store the double result back to p
}

// CUDA kernel for energy sum calculation
__global__ void energy_sum_kernel(double *d_p, double *partialSum, int N_PART) {
    extern __shared__ double sharedData[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sharedData[tid] = (i < N_PART) ? d_p[i] * d_p[i] : 0.0;
    __syncthreads();
    
    // Perform reduction within each block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }
    
    // Write result of this block's partial sum to global memory
    if (tid == 0) {
        partialSum[blockIdx.x] = sharedData[0];
    }
}
