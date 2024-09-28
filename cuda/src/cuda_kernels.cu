#include <cuda_runtime.h>
#include <omp.h>
#include "../include/utils.h"

// CUDA kernel for initializing the momentum histogram
__global__ void initialize_momentum_histogram(double* DpE, int BINS, int N_PART) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (BINS << 1)) {
        // Calculate Gaussian distribution for momentum
        double numerator = 6.0E-26 * N_PART;
        double denominator = 5.24684E-24 * sqrt(2.0 * PI);
        double exponent = -pow(3.0e-23 * (1.0 * i / BINS - 0.999) / 5.24684E-24, 2) / 2;
        DpE[i] = (numerator / denominator) * exp(exponent);
    }
}

// CUDA kernel for initializing the position histogram
__global__ void initialize_position_histogram(double* DxE, int BINS, int N_PART) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 2 && i < (BINS + 1) << 1) {
        DxE[i] = 1.0E-3 * N_PART;
    }

    // Set boundary conditions in a single thread
    if (i == 0) {
        DxE[0] = 0.0;
        DxE[1] = 0.0;
        DxE[2 * BINS + 2] = 0.0;
        DxE[2 * BINS + 3] = 0.0;
    }
}