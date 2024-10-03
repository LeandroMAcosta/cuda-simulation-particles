#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>

#include "../include/constants.h"
#include "../include/histogram_kernels.h"

using namespace std;

__global__ void calculateDpE(double *DpE, int N_PART, int BINS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 2 * BINS) {
        double numerator = 6.0E-26 * N_PART;
        double denominator = 5.24684E-24 * sqrt(2.0 * M_PI);
        double exponent = -pow(3.0e-23 * (1.0 * i / BINS - 0.999) / 5.24684E-24, 2) / 2;
        DpE[i] = (numerator / denominator) * exp(exponent);
    }
}

__global__ void calculateDxE(double *DxE, int N_PART, int BINS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 && i < (BINS + 1) << 1) {
        DxE[i] = 1.0E-3 * N_PART;
    }
}

// __device__ uint32_t generate_random() {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t seed = idx + blockIdx.x + threadIdx.x * 31;

//     curandState state;
//     curand_init(seed, idx, 0, &state);  // Initialize random state

//     // Generate a random 32-bit unsigned integer
//     int random = curand(&state); 
//     // printf("[GENERATE RANDOM] random=%d\n", random);
//     return random;
// }

// XORShift function to generate pseudo-random numbers
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

__global__ void init_x_kernel(double* x, uint32_t global_seed, int N_PART) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N_PART) {
        // Generate a unique seed for each thread using global_seed, blockIdx, and threadIdx
        uint32_t seed = global_seed + idx;

        // Generate random value using XORShift and normalize to [0, 1]
        double randomValue = (double)(xorshift32(&seed)) / UINT32_MAX;
        x[idx] = randomValue * 0.5;
    }
}

__global__ void init_p_kernel(double* p, uint32_t global_seed, int N_PART) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < (N_PART >> 1)) {
        // Generate a unique seed for each thread using global_seed, blockIdx, and threadIdx
        uint32_t seed = global_seed + idx;

        // Generate two random values using XORShift
        double randomValue1 = (double)(xorshift32(&seed)) / UINT32_MAX;
        double randomValue2 = (double)(xorshift32(&seed)) / UINT32_MAX;

        // Box-Muller transform to generate two normally distributed random numbers
        double xi1 = sqrt(-2.0 * log(randomValue1 + 1E-35));
        double xi2 = 2.0 * M_PI * randomValue2;

        // Store the generated values in the p array
        p[2 * idx] = xi1 * cos(xi2) * 5.24684E-24;
        p[2 * idx + 1] = xi1 * sin(xi2) * 5.24684E-24;
    }
}

// Kernel function to update positions and momenta
__global__ void simulate_particle_motion(int j, double *x, double *p, double *DxE, double *DpE, int *h, int *g, int *hg, int N_PART, int *steps, double DT, double M, double sigmaL, double alfa, double pmin, double pmax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_PART) return;

    // Initialize seed for random number generator
    uint32_t seed = idx + blockIdx.x + threadIdx.x * 31;
    curandState state;
    curand_init(seed, idx, 0, &state);  // Initialize random state

    // Generate a random 32-bit unsigned integer
    unsigned int random_uint = curand(&state);  // Random 32-bit unsigned integer
    seed = random_uint;

    double x_tmp = x[idx];
    double p_tmp = p[idx];

    int signop, k;

    // Main particle loop
    for (int step = 0; step < steps[j]; ++step) {
        x_tmp += p_tmp * DT / M;
        signop = copysign(1.0, p_tmp);
        k = trunc(x_tmp + 0.5 * signop);

        if (k != 0) {
            // Generate two random values using CUDA's curand
            double randomValue = d_xorshift(&seed);
            double xi1 = sqrt(-2.0 * log(randomValue + 1E-35));
            randomValue = d_xorshift(&seed);
            double xi2 = 2.0 * M_PI * randomValue;
            // double deltaX = sqrt(labs(k)) * xi1 * cos(xi2) * sigmaL;
            double deltaX = sqrt(fabsf(k)) * xi1 * cos(xi2) * sigmaL;

            deltaX = (fabs(deltaX) > 1.0 ? 1.0 * copysign(1.0, deltaX) : deltaX);
            x_tmp = (k % 2 ? -1.0 : 1.0) * (x_tmp - k) + deltaX;

            if (fabs(x_tmp) > 0.502) {
                x_tmp = 1.004 * copysign(1.0, x_tmp) - x_tmp;
            }
            p_tmp = fabs(p_tmp);

            // printf("[SIMULATE PARTICLE MOTION] k=%d, labs(k)=%ld\n", k, labs(k));
            for (int l = 1; l <= labs(k); l++) {
                // printf("[SIMULATE PARTICLE MOTION] j=%d, step=%d, steps[j]=%d\n", j, step, steps[j]);
                double DeltaE = alfa * (p_tmp - pmin) * (pmax - p_tmp);
                randomValue = d_xorshift(&seed);
                p_tmp = sqrt(p_tmp * p_tmp + DeltaE * (randomValue - 0.5));
            }
            p_tmp *= (k % 2 ? -1.0 : 1.0) * signop;
        }
    }
    // Update global memory
    x[idx] = x_tmp;
    p[idx] = p_tmp;
}
