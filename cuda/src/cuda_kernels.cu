#include <cuda_runtime.h>
#include <omp.h>
#include "../include/utils.h"


void initialize_particles_and_histogram(int N_PART, double* x, double* p, int* h, int* g, int* hg, double* DxE, double* DpE, unsigned int evolution, double M, int BINS) {
    int X0 = 1;                                 // Control variable for resuming simulation
    double xi1 = 0.0, xi2 = 0.0;                // Temporary variables for random numbers

    while (X0 == 1) {
        // Initialize particles' positions and momenta
        #pragma omp parallel
        {
            uint32_t seed = (uint32_t)(time(NULL) + omp_get_thread_num());  // Seed for random number generation

            // Initialize particle positions
            #pragma omp for schedule(static)
            for (int i = 0; i < N_PART; i++) {
                double randomValue = d_xorshift(&seed);  // Generate random position
                x[i] = randomValue * 0.5;
            }

            // Initialize particle momenta
            #pragma omp for schedule(static)
            for (int i = 0; i < N_PART >> 1; i++) {
                double randomValue1 = d_xorshift(&seed);
                double randomValue2 = d_xorshift(&seed);

                // Box-Muller transform to generate random momentum values
                xi1 = sqrt(-2.0 * log(randomValue1 + 1E-35));
                xi2 = 2.0 * PI * randomValue2;

                p[2 * i] = xi1 * cos(xi2) * 5.24684E-24;
                p[2 * i + 1] = xi1 * sin(xi2) * 5.24684E-24;
            }
        }

        /* Update histograms based on particle positions and momenta */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N_PART; i++) {
            int h_idx = floor((x[i] + 0.5) * (1.99999999999999 * BINS) + 2.0);
            int g_idx = floor((p[i] / 3.0e-23 + 1) * (0.999999999999994 * BINS));
            int hg_idx = (2 * BINS) * h_idx + g_idx;
            h[h_idx]++;
            g[g_idx]++;
            hg[hg_idx]++;
        }

        // Calculate total energy and generate histogram file
        double Et = energy_sum(p, N_PART, evolution, M);
        X0 = make_hist(h, g, hg, DxE, DpE, "X0000000.dat", BINS, Et);
        if (X0 == 1) {
            printf("Error: Chi-square test failed: X0=%1d\n", X0);
        }
    }
}

// Function to initialize histograms on the device
void initialize_histograms(int BINS, int* h, int* g, int* hg, double* DxE, double* DpE, int N_PART) {
    // Define thread and block sizes for CUDA kernels
    int blockSize = 256;
    int gridSizeMomentum = (BINS << 1 + blockSize - 1) / blockSize;
    int gridSizePosition = ((BINS + 1) << 1 + blockSize - 1) / blockSize;

    // Allocate memory on the device
    int* d_h;
    int* d_g;
    int* d_hg;
    double* d_DxE;
    double* d_DpE;

    cudaMalloc(&d_h, (2 * BINS + 4) * sizeof(int));
    cudaMalloc(&d_g, (2 * BINS) * sizeof(int));
    cudaMalloc(&d_hg, (2 * BINS + 4) * (2 * BINS) * sizeof(int));
    cudaMalloc(&d_DxE, (2 * BINS + 4) * sizeof(double));
    cudaMalloc(&d_DpE, (2 * BINS) * sizeof(double));

    // Initialize momentum histogram
    initialize_momentum_histogram<<<gridSizeMomentum, blockSize>>>(d_DpE, BINS, N_PART);

    // Initialize position histogram
    initialize_position_histogram<<<gridSizePosition, blockSize>>>(d_DxE, BINS, N_PART);

    // Set memory for h, g, hg histograms to zero
    cudaMemset(d_h, 0, (2 * BINS + 4) * sizeof(int));
    cudaMemset(d_g, 0, (2 * BINS) * sizeof(int));
    cudaMemset(d_hg, 0, (2 * BINS + 4) * (2 * BINS) * sizeof(int));

    // Copy results from device back to host
    cudaMemcpy(h, d_h, (2 * BINS + 4) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(g, d_g, (2 * BINS) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hg, d_hg, (2 * BINS + 4) * (2 * BINS) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(DxE, d_DxE, (2 * BINS + 4) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(DpE, d_DpE, (2 * BINS) * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_h);
    cudaFree(d_g);
    cudaFree(d_hg);
    cudaFree(d_DxE);
    cudaFree(d_DpE);
}

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