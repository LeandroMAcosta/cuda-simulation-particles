#include "../include/kernels.h"
#include "../include/types.h"
#include "../config.h"

// ============================================================================
// INITIALIZATION KERNELS
// ============================================================================

// Initialize RNG states for each thread
__global__ void init_rng_states(curandState *states, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed + idx + RNG_SEED_OFFSET, idx, 0, &states[idx]);
    }
}

// Initialize particle positions and momenta
__global__ void initialize_particles(float *x, float *p, curandState *states, int N_PART) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PART) {
        curandState localState = states[idx];
        
        // Initialize position uniformly in [0, 0.5)
        x[idx] = curand_uniform(&localState) * 0.5f;
        
        // Initialize momentum using Box-Muller transform
        // Only generate for half the particles to avoid thread divergence
        if (idx < N_PART / 2) {
            float randomValue1 = curand_uniform(&localState) + 1E-35f;
            float randomValue2 = curand_uniform(&localState);
            
            // Box-Muller transform for Gaussian distribution
            float xi1 = sqrtf(-2.0f * logf(randomValue1));
            float xi2 = 2.0f * PI * randomValue2;
            
            // Generate two independent Gaussian values
            p[2 * idx] = xi1 * cosf(xi2) * 5.24684E-24f;
            if (2 * idx + 1 < N_PART) {
                p[2 * idx + 1] = xi1 * sinf(xi2) * 5.24684E-24f;
            }
        }
        
        // Update the state back to global memory
        states[idx] = localState;
    }
}

// Initialize expected momentum distribution (DpE) for validation
__global__ void initialize_momentum_distribution(double *DpE, int BINS, int N_PART) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 2 * BINS) {
        double numerator = 6.0E-26 * N_PART;
        double denominator = 5.24684E-24 * sqrt(2.0 * PI);
        double exponent = -pow(3.0e-23 * (1.0 * i / BINS - 0.999) / 5.24684E-24, 2) / 2;
        DpE[i] = (numerator / denominator) * exp(exponent);
    }
}

// Initialize expected position distribution (DxE) for validation
__global__ void initialize_position_distribution(double *DxE, int BINS, int N_PART) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bins = 2 * BINS + 4;
    
    if (i < total_bins) {
        if (i < 2 || i >= 2 * BINS + 2) {
            DxE[i] = 0.0;  // Edge bins
        } else {
            DxE[i] = 1.0E-3 * N_PART;  // Uniform distribution in center
        }
    }
} 