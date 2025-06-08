#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"

// ============================================================================
// CONSTANT MEMORY DECLARATION
// ============================================================================

// Declaration of constant memory for simulation parameters
extern __device__ __constant__ SimulationParams d_params;

// ============================================================================
// INITIALIZATION KERNELS (init_kernels.cu)
// ============================================================================

__global__ void init_rng_states(curandState *states, unsigned long seed, int N);
__global__ void initialize_particles(float *x, float *p, curandState *states, int N_PART);
__global__ void initialize_momentum_distribution(double *DpE, int BINS, int N_PART);
__global__ void initialize_position_distribution(double *DxE, int BINS, int N_PART);

// ============================================================================
// SIMULATION KERNELS (simulation_kernels.cu)
// ============================================================================

__global__ void particle_evolution_kernel(float *x, float *p, curandState *states, 
                                         int steps);
__global__ void particle_evolution_kernel_v2(float *x, float *p, curandState *states, 
                                            int steps);

// ============================================================================
// HISTOGRAM KERNELS (histogram_kernels.cu)
// ============================================================================

__global__ void clear_histograms(int *h, int *g, int *hg, int BINS);
__global__ void compute_histograms(float *x, float *p, int *h, int *g, int *hg, 
                                  int N_PART, int BINS);
__global__ void compute_histograms_shared(float *x, float *p, int *h, int *g, int *hg, 
                                         int N_PART, int BINS);

// ============================================================================
// REDUCTION KERNELS (reduction_kernels.cu)
// ============================================================================

__global__ void energy_sum_kernel(float *p, double *partial_sums, int N_PART, double M);
__global__ void chi2_reduction_kernel(int *h, int *g, double *DxE, double *DpE, 
                                     double *chi2_results, int BINS, int is_initial);
__global__ void energy_sum_kernel_optimized(float *p, double *partial_sums, int N_PART, double M);

// Generic reduction kernel template
template<typename T, typename Op>
__global__ void generic_reduction_kernel(T *input, T *output, int N, T identity, Op op);

// ============================================================================
// UTILITY KERNELS (utility_kernels.cu)
// ============================================================================

// Random number generation utilities
__device__ float d_curand_float(curandState *state);
__device__ double d_curand_double(curandState *state);
__device__ float2 d_box_muller_float(curandState *state);

// Mathematical utilities
__device__ void d_fast_sincos(float x, float *sin_val, float *cos_val);
__device__ double d_safe_sqrt(double value);
__device__ double d_clamp(double value, double min_val, double max_val);
__device__ double d_copysign_double(double value, double sign);
__device__ int d_mod_fast(int value, int divisor);

// Validation and debugging utilities
__global__ void validate_memory_kernel(float *data, int N, float *error_flags);
__global__ void validate_particle_bounds(float *x, float *p, int N_PART, int *error_count);
__global__ void compute_checksum(float *data, int N, unsigned long long *checksum);

#endif 