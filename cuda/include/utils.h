#ifndef UTILS_H
#define UTILS_H

#include "types.h"
#include "constants.h"

// Host utility functions
void load_parameters_from_file(char filename[], SimulationParams *params);
void read_data(char filename[], double *x, double *p, unsigned int *evolution, int N_PART);
void save_data(char filename[], double *x, double *p, unsigned int evolution, int N_PART);
double energy_sum_host(double *p, int N_PART, unsigned int evolution, double M);
int make_hist_host(int *h, int *g, int *hg, double *DxE, double *DpE, const char *filename, int BINS, double Et);

// Memory management functions
void allocate_simulation_data(SimulationData *data, SimulationParams *params);
void free_simulation_data(SimulationData *data);
void copy_data_to_device(SimulationData *h_data, SimulationData *d_data, SimulationParams *params);
void copy_data_to_host(SimulationData *h_data, SimulationData *d_data, SimulationParams *params);

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device random number generation
__device__ double d_curand(curandState *state);

#endif 