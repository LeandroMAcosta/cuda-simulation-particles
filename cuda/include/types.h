#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Simulation data structure for better memory management
typedef struct {
    double *x;          // Particle positions
    double *p;          // Particle momenta  
    double *DxE;        // Expected position distribution
    double *DpE;        // Expected momentum distribution
    int *h;             // Position histogram
    int *g;             // Momentum histogram
    int *hg;            // Combined histogram
    curandState *rng_states; // RNG states for each thread
} SimulationData;

// Simulation parameters structure
typedef struct {
    int N_PART;         // Number of particles
    int BINS;           // Number of histogram bins
    double DT;          // Time step
    double M;           // Particle mass
    int N_THREADS;      // Number of threads (for compatibility)
    unsigned int Ntandas; // Number of iterations
    int steps[500];     // Steps per iteration
    char inputFilename[255];
    char saveFilename[255];
    int resume;         // Resume from file flag
    int dump;           // Dump data flag
    double sigmaL;      // Uncertainty for L
    unsigned int evolution; // Current evolution step
} SimulationParams;

// Physical constants
#define BORDES 237
#define PI 3.14159265358979323846
#define epsmax2M 9.0E-46
#define DEmax2M 6.0e-50
#define epsmin2M 9.0E-52

// CUDA kernel configuration
#define BLOCK_SIZE 256
#define GRID_SIZE(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)

#endif 