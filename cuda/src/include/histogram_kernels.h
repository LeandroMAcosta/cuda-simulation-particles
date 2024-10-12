#ifndef HISTOGRAM_KERNELS_H
#define HISTOGRAM_KERNELS_H

__global__ void init_DpE_kernel(double *DpE, int N_PART, int BINS);

__global__ void init_DxE_kernel(double *DxE, int N_PART, int BINS);

__global__ void simulate_particle_motion(int number_of_steps, float *x, double *p,
                                         int N_PART, double DT, double M, float sigmaL,
                                         float alfa, float pmin, float pmax);

__global__ void init_x_kernel(float *x, uint32_t base_seed, int N_PART);

__global__ void init_p_kernel(double *p, uint32_t base_seed, int N_PART);

__global__ void update_histograms_kernel(float *x, double *p, int *h, int *g, int *hg, int N_PART, int BINS);

__global__ void energy_sum_kernel(double *p, double *partial_sum, int N_PART);

#endif
