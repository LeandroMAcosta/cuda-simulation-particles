#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"

// Declaration of constant memory for simulation parameters
extern __device__ __constant__ SimulationParams d_params;

// Kernel function declarations

// Initialization kernels
__global__ void init_rng_states(curandState *states, unsigned long seed, int N);
__global__ void initialize_particles(double *x, double *p, curandState *states, int N_PART);
__global__ void initialize_momentum_distribution(double *DpE, int BINS, int N_PART);
__global__ void initialize_position_distribution(double *DxE, int BINS, int N_PART);

// Simulation kernels
__global__ void particle_evolution_kernel(double *x, double *p, curandState *states, 
                                         int steps);

// Histogram kernels  
__global__ void clear_histograms(int *h, int *g, int *hg, int BINS);
__global__ void compute_histograms(double *x, double *p, int *h, int *g, int *hg, 
                                  int N_PART, int BINS);

// Reduction kernels
__global__ void energy_sum_kernel(double *p, double *partial_sums, int N_PART, double M);
__global__ void chi2_reduction_kernel(int *h, int *g, double *DxE, double *DpE, 
                                     double *chi2_results, int BINS, int is_initial);

// Utility kernels
__device__ double d_curand(curandState *state);

#endif 