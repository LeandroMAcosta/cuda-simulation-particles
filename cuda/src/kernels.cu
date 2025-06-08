#include "../include/kernels.h"
#include "../include/types.h"
#include "../config.h"

// Definition of constant memory for simulation parameters
__device__ __constant__ SimulationParams d_params;

// Device random number generation using cuRAND
__device__ float d_curand_float(curandState *state) {
    return curand_uniform(state);
}

__device__ double d_curand_double(curandState *state) {
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
__global__ void initialize_particles(float *x, float *p, curandState *states, int N_PART) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PART) {
        curandState localState = states[idx];
        
        // Initialize position
        x[idx] = curand_uniform(&localState) * 0.5f;
        
        // Initialize momentum using Box-Muller transform
        if (idx < N_PART / 2) {
            float randomValue1 = curand_uniform(&localState) + 1E-35f;
            float randomValue2 = curand_uniform(&localState);
            
            float xi1 = sqrtf(-2.0f * logf(randomValue1));
            float xi2 = 2.0f * PI * randomValue2;
            
            p[2 * idx] = xi1 * cosf(xi2) * 5.24684E-24f;
            if (2 * idx + 1 < N_PART) {
                p[2 * idx + 1] = xi1 * sinf(xi2) * 5.24684E-24f;
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

// Optimized particle evolution kernel with mixed precision
__global__ void particle_evolution_kernel(float *x, float *p, curandState *states, 
                                         int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.N_PART) return;
    
    curandState localState = states[idx];
    
    // Use double precision for critical calculations
    double x_tmp = (double)x[idx];
    double p_tmp = (double)p[idx];
    
    // Constants in appropriate precision
    const double pmin = 2.0E-026;
    const double pmax = 3.0E-023;
    const double alfa = 0.0; // As set in original code
    const double dt_over_m = d_params.DT / d_params.M;
    const double sigmaL = d_params.sigmaL;
    
    // Pre-compute frequently used values
    const float two_pi = 2.0f * PI;
    const float log_min = 1E-35f;
    
    for (int step = 0; step < steps; step++) {
        // Critical position update in double precision
        x_tmp += p_tmp * dt_over_m;
        
        // Collision detection - can use float for comparison
        int signop = (p_tmp > 0.0) ? 1 : -1;
        long int k = (long int)trunc(x_tmp + 0.5 * signop);
        
        if (k != 0) {
            // Wall collision occurred - use float for random generation
            float rand1 = curand_uniform(&localState);
            float rand2 = curand_uniform(&localState);
            
            // Box-Muller in float (sufficient for noise)
            float xi1 = sqrtf(-2.0f * logf(rand1 + log_min));
            float xi2 = two_pi * rand2;
            
            // Critical calculations back to double
            double deltaX = sqrt((double)labs(k)) * (double)xi1 * cos((double)xi2) * sigmaL;
            deltaX = (fabs(deltaX) > 1.0 ? copysign(1.0, deltaX) : deltaX);
            
            x_tmp = (k % 2 ? -1.0 : 1.0) * (x_tmp - k) + deltaX;
            
            if (fabs(x_tmp) > 0.502) {
                x_tmp = 1.004 * copysign(1.0, x_tmp) - x_tmp;
            }
            
            p_tmp = fabs(p_tmp); // Remove sign
            
            // Energy redistribution - critical for energy conservation
            for (int l = 1; l <= labs(k); l++) {
                double DeltaE = alfa * (p_tmp - pmin) * (pmax - p_tmp);
                float rand3 = curand_uniform(&localState);
                double value = p_tmp * p_tmp + DeltaE * ((double)rand3 - 0.5);
                if (value < 0.0) {
                    value = 0.0;
                }
                p_tmp = sqrt(value);
            }
            
            p_tmp *= (k % 2 ? -1.0 : 1.0) * signop; // Restore sign
        }
    }
    
    // Store back as float
    x[idx] = (float)x_tmp;
    p[idx] = (float)p_tmp;
    states[idx] = localState;
}

// Alternative highly optimized kernel with reduced branching
__global__ void particle_evolution_kernel_v2(float *x, float *p, curandState *states, 
                                            int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.N_PART) return;
    
    curandState localState = states[idx];
    
    // Use double precision for critical calculations
    double x_tmp = (double)x[idx];
    double p_tmp = (double)p[idx];
    
    // Constants in constant memory or registers
    const double dt_over_m = d_params.DT / d_params.M;
    const double sigmaL = d_params.sigmaL;
    const double boundary = 0.502;
    const double reflection_factor = 1.004;
    
    // Precompute values to reduce function calls
    const float two_pi = 6.28318530718f;
    const float log_min = 1E-35f;
    
    for (int step = 0; step < steps; step++) {
        // Position update - most critical calculation
        x_tmp += p_tmp * dt_over_m;
        
        // Fast collision detection
        double signop_d = copysign(1.0, p_tmp);
        long int k = (long int)trunc(x_tmp + 0.5 * signop_d);
        
        // Use conditional assignment to reduce branching
        if (k != 0) {
            // Generate random numbers in batch
            float2 rand_pair = make_float2(curand_uniform(&localState), 
                                         curand_uniform(&localState));
            
            // Box-Muller transform with fast math
            float xi1 = __fsqrt_rn(-2.0f * __logf(rand_pair.x + log_min));
            float xi2 = two_pi * rand_pair.y;
            
            // Use fast trigonometric functions
            float cos_xi2, sin_xi2;
            __sincosf(xi2, &sin_xi2, &cos_xi2);
            
            // Critical position calculation in double
            double k_sqrt = sqrt((double)abs(k));
            double deltaX = k_sqrt * (double)xi1 * (double)cos_xi2 * sigmaL;
            
            // Clamp deltaX with branchless operation
            deltaX = fmax(-1.0, fmin(1.0, deltaX));
            
            // Position update
            double wall_factor = (k % 2) ? -1.0 : 1.0;
            x_tmp = wall_factor * (x_tmp - k) + deltaX;
            
            // Boundary reflection with branchless operation
            double abs_x = fabs(x_tmp);
            bool needs_reflection = abs_x > boundary;
            x_tmp = needs_reflection ? (reflection_factor * copysign(1.0, x_tmp) - x_tmp) : x_tmp;
            
            // Momentum handling
            p_tmp = fabs(p_tmp);
            
            // Energy redistribution loop - since alfa=0, this does nothing
            // but kept for correctness
            int k_abs = abs(k);
            for (int l = 1; l <= k_abs; l++) {
                // Since alfa = 0, DeltaE = 0, so this is just p_tmp = |p_tmp|
                // We can optimize this out when alfa=0
                float rand3 = curand_uniform(&localState);
                double value = p_tmp * p_tmp; // DeltaE term is 0
                p_tmp = sqrt(fmax(0.0, value));
            }
            
            // Restore momentum sign
            p_tmp *= wall_factor * signop_d;
        }
    }
    
    // Store results
    x[idx] = (float)x_tmp;
    p[idx] = (float)p_tmp;
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
__global__ void compute_histograms(float *x, float *p, int *h, int *g, int *hg, 
                                  int N_PART, int BINS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_PART) return;
    
    // Calculate histogram indices
    int h_idx = floorf((x[idx] + 0.5f) * (1.99999999999999f * BINS) + 2.0f);
    int g_idx = floorf((p[idx] / 3.0e-23f + 1.0f) * (0.999999999999994f * BINS));
    
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
__global__ void energy_sum_kernel(float *p, double *partial_sums, int N_PART, double M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ double shared_sums[BLOCK_SIZE];
    
    // Each thread computes local sum - convert to double for accuracy
    double local_sum = 0.0;
    if (idx < N_PART) {
        double p_double = (double)p[idx];
        local_sum = p_double * p_double;
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