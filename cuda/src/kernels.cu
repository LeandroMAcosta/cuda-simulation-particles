#include "../include/kernels.h"
#include "../include/types.h"
#include "../config.h"

// Device random number generation using cuRAND
__device__ double d_curand(curandState *state) {
    return curand_uniform_double(state);
}

// Initialize RNG states for each thread
__global__ void init_rng_states(curandState *states, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed + idx + RNG_SEED_OFFSET, idx, 0, &states[idx]);
    }
}

// Initialize particle positions
__global__ void initialize_particles(double *x, double *p, curandState *states, int N_PART) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PART) {
        curandState localState = states[idx];
        
        // Initialize position
        x[idx] = curand_uniform_double(&localState) * 0.5;
        
        // Initialize momentum using Box-Muller transform
        if (idx < N_PART / 2) {
            double randomValue1 = curand_uniform_double(&localState) + 1E-35;
            double randomValue2 = curand_uniform_double(&localState);
            
            double xi1 = sqrt(-2.0 * log(randomValue1));
            double xi2 = 2.0 * PI * randomValue2;
            
            p[2 * idx] = xi1 * cos(xi2) * 5.24684E-24;
            if (2 * idx + 1 < N_PART) {
                p[2 * idx + 1] = xi1 * sin(xi2) * 5.24684E-24;
            }
        }
        
        states[idx] = localState;
    }
}

// Initialize momentum distribution (DpE)
__global__ void initialize_momentum_distribution(double *DpE, int BINS, int N_PART) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 2 * BINS) {
        double numerator = 6.0E-26 * N_PART;
        double denominator = 5.24684E-24 * sqrt(2.0 * PI);
        double exponent = -pow(3.0e-23 * (1.0 * i / BINS - 0.999) / 5.24684E-24, 2) / 2;
        DpE[i] = (numerator / denominator) * exp(exponent);
    }
}

// Initialize position distribution (DxE)
__global__ void initialize_position_distribution(double *DxE, int BINS, int N_PART) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bins = 2 * BINS + 4;
    
    if (i < total_bins) {
        if (i < 2 || i >= 2 * BINS + 2) {
            DxE[i] = 0.0;
        } else {
            DxE[i] = 1.0E-3 * N_PART;
        }
    }
}

// Main particle evolution kernel
__global__ void particle_evolution_kernel(double *x, double *p, curandState *states, 
                                         SimulationParams params, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.N_PART) return;
    
    curandState localState = states[idx];
    double x_tmp = x[idx];
    double p_tmp = p[idx];
    
    double pmin = 2.0E-026;
    double pmax = 3.0E-023;
    double alfa = 0.0; // As set in original code
    
    for (int step = 0; step < steps; step++) {
        // Update position
        x_tmp += p_tmp * params.DT / params.M;
        
        // Check for wall collision
        int signop = copysign(1.0, p_tmp);
        long int k = trunc(x_tmp + 0.5 * signop);
        
        if (k != 0) {
            // Wall collision occurred
            double randomValue = curand_uniform_double(&localState);
            double xi1 = sqrt(-2.0 * log(randomValue + 1E-35));
            randomValue = curand_uniform_double(&localState);
            double xi2 = 2.0 * PI * randomValue;
                         double deltaX = sqrt((double)labs(k)) * xi1 * cos(xi2) * params.sigmaL;
            deltaX = (fabs(deltaX) > 1.0 ? 1.0 * copysign(1.0, deltaX) : deltaX);
            
            x_tmp = (k % 2 ? -1.0 : 1.0) * (x_tmp - k) + deltaX;
            
            if (fabs(x_tmp) > 0.502) {
                x_tmp = 1.004 * copysign(1.0, x_tmp) - x_tmp;
            }
            
            p_tmp = fabs(p_tmp); // Remove sign
            
            // Energy redistribution for each collision
                         for (int l = 1; l <= labs(k); l++) {
                double DeltaE = alfa * (p_tmp - pmin) * (pmax - p_tmp);
                randomValue = curand_uniform_double(&localState);
                double value = p_tmp * p_tmp + DeltaE * (randomValue - 0.5);
                if (value < 0.0) {
                    value = 0.0;
                }
                p_tmp = sqrt(value);
            }
            
            p_tmp *= (k % 2 ? -1.0 : 1.0) * signop; // Restore sign
        }
    }
    
    x[idx] = x_tmp;
    p[idx] = p_tmp;
    states[idx] = localState;
}

// Clear histogram arrays
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

// Compute histograms using atomic operations
__global__ void compute_histograms(double *x, double *p, int *h, int *g, int *hg, 
                                  int N_PART, int BINS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_PART) return;
    
    // Calculate histogram indices
    int h_idx = floor((x[idx] + 0.5) * (1.99999999999999 * BINS) + 2.0);
    int g_idx = floor((p[idx] / 3.0e-23 + 1) * (0.999999999999994 * BINS));
    
    // Bounds checking
    h_idx = max(0, min(h_idx, 2 * BINS + 3));
    g_idx = max(0, min(g_idx, 2 * BINS - 1));
    
    int hg_idx = (2 * BINS) * h_idx + g_idx;
    
    // Atomic increments for thread safety
    atomicAdd(&h[h_idx], 1);
    atomicAdd(&g[g_idx], 1);
    atomicAdd(&hg[hg_idx], 1);
}

// Energy sum kernel with reduction
__global__ void energy_sum_kernel(double *p, double *partial_sums, int N_PART, double M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ double shared_sums[BLOCK_SIZE];
    
    // Each thread computes local sum
    double local_sum = 0.0;
    if (idx < N_PART) {
        local_sum = p[idx] * p[idx];
    }
    shared_sums[tid] = local_sum;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sums[0] / (2 * M);
    }
}

// Chi-square reduction kernel for histogram analysis
__global__ void chi2_reduction_kernel(int *h, int *g, double *DxE, double *DpE, 
                                     double *chi2_results, int BINS, int is_initial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ double shared_chi2[BLOCK_SIZE];
    shared_chi2[tid] = 0.0;
    
    // Different chi2 calculations based on phase
    if (is_initial) {
        // Initial phase chi2 calculation
        if (idx >= BINS && idx < 2 * BINS) {
            double expected = 2 * DxE[idx];
            if (expected > 0) {
                shared_chi2[tid] = pow(h[idx] - expected, 2) / expected;
            }
        }
    } else {
        // Regular phase chi2 calculation
        if (idx >= 4 && idx < 2 * BINS) {
            double expected = DxE[idx];
            if (expected > 0) {
                shared_chi2[tid] = pow(h[idx] - expected, 2) / expected;
            }
        }
    }
    
    __syncthreads();
    
    // Reduction
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