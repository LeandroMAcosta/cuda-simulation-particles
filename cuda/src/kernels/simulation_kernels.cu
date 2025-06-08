#include "../include/kernels.h"
#include "../include/types.h"
#include "../config.h"

// ============================================================================
// SIMULATION KERNELS
// ============================================================================

// Definition of constant memory for simulation parameters
__device__ __constant__ SimulationParams d_params;

// Main particle evolution kernel with mixed precision optimization
__global__ void particle_evolution_kernel(float *x, float *p, curandState *states, 
                                         int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.N_PART) return;
    
    curandState localState = states[idx];
    
    // Use double precision for critical calculations to maintain energy conservation
    double x_tmp = (double)x[idx];
    double p_tmp = (double)p[idx];
    
    // Physical constants in appropriate precision
    const double pmin = 2.0E-026;
    const double pmax = 3.0E-023;
    const double alfa = 0.0; // Energy redistribution parameter (currently 0)
    const double dt_over_m = d_params.DT / d_params.M;
    const double sigmaL = d_params.sigmaL;
    
    // Pre-compute frequently used values to reduce function calls
    const float two_pi = 2.0f * PI;
    const float log_min = 1E-35f;
    
    for (int step = 0; step < steps; step++) {
        // Critical position update in double precision for accuracy
        x_tmp += p_tmp * dt_over_m;
        
        // Collision detection with wall boundaries
        int signop = (p_tmp > 0.0) ? 1 : -1;
        long int k = (long int)trunc(x_tmp + 0.5 * signop);
        
        if (k != 0) {
            // Wall collision occurred - generate random noise
            float rand1 = curand_uniform(&localState);
            float rand2 = curand_uniform(&localState);
            
            // Box-Muller transform in float (sufficient precision for noise)
            float xi1 = sqrtf(-2.0f * logf(rand1 + log_min));
            float xi2 = two_pi * rand2;
            
            // Critical position calculations back to double precision
            double deltaX = sqrt((double)labs(k)) * (double)xi1 * cos((double)xi2) * sigmaL;
            deltaX = (fabs(deltaX) > 1.0 ? copysign(1.0, deltaX) : deltaX);
            
            // Apply wall reflection with position correction
            x_tmp = (k % 2 ? -1.0 : 1.0) * (x_tmp - k) + deltaX;
            
            // Boundary check and correction
            if (fabs(x_tmp) > 0.502) {
                x_tmp = 1.004 * copysign(1.0, x_tmp) - x_tmp;
            }
            
            p_tmp = fabs(p_tmp); // Remove momentum sign for processing
            
            // Energy redistribution loop (currently disabled with alfa=0)
            for (int l = 1; l <= labs(k); l++) {
                double DeltaE = alfa * (p_tmp - pmin) * (pmax - p_tmp);
                float rand3 = curand_uniform(&localState);
                double value = p_tmp * p_tmp + DeltaE * ((double)rand3 - 0.5);
                if (value < 0.0) {
                    value = 0.0;
                }
                p_tmp = sqrt(value);
            }
            
            // Restore momentum sign based on wall reflection
            p_tmp *= (k % 2 ? -1.0 : 1.0) * signop;
        }
    }
    
    // Store results back as float
    x[idx] = (float)x_tmp;
    p[idx] = (float)p_tmp;
    states[idx] = localState;
}

// Alternative highly optimized kernel with reduced branching and fast math
__global__ void particle_evolution_kernel_v2(float *x, float *p, curandState *states, 
                                            int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.N_PART) return;
    
    curandState localState = states[idx];
    
    // Use double precision for critical calculations
    double x_tmp = (double)x[idx];
    double p_tmp = (double)p[idx];
    
    // Constants optimized for registers
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
            // Generate random numbers in batch for better performance
            float2 rand_pair = make_float2(curand_uniform(&localState), 
                                         curand_uniform(&localState));
            
            // Box-Muller transform with fast math intrinsics
            float xi1 = __fsqrt_rn(-2.0f * __logf(rand_pair.x + log_min));
            float xi2 = two_pi * rand_pair.y;
            
            // Use fast simultaneous trigonometric functions
            float cos_xi2, sin_xi2;
            __sincosf(xi2, &sin_xi2, &cos_xi2);
            
            // Critical position calculation in double precision
            double k_sqrt = sqrt((double)abs(k));
            double deltaX = k_sqrt * (double)xi1 * (double)cos_xi2 * sigmaL;
            
            // Clamp deltaX with branchless operation
            deltaX = fmax(-1.0, fmin(1.0, deltaX));
            
            // Position update with wall reflection
            double wall_factor = (k % 2) ? -1.0 : 1.0;
            x_tmp = wall_factor * (x_tmp - k) + deltaX;
            
            // Boundary reflection with branchless operation
            double abs_x = fabs(x_tmp);
            bool needs_reflection = abs_x > boundary;
            x_tmp = needs_reflection ? (reflection_factor * copysign(1.0, x_tmp) - x_tmp) : x_tmp;
            
            // Momentum handling
            p_tmp = fabs(p_tmp);
            
            // Energy redistribution loop - optimized for alfa=0 case
            int k_abs = abs(k);
            for (int l = 1; l <= k_abs; l++) {
                // Since alfa = 0, DeltaE = 0, this simplifies significantly
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