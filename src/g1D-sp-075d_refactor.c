#include "../include/utils.h" // Include custom utility functions
#include <omp.h> // Include OpenMP for parallel processing
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

// Function to allocate memory for simulation arrays
bool allocate_memory(int N_PART, int BINS, double** x, double** p, double** DxE, double** DpE, int** h, int** g, int** hg) {
    *x = malloc(sizeof(double) * N_PART);
    *p = malloc(sizeof(double) * N_PART);
    *DxE = malloc(sizeof(double) * (2 * BINS + 5));
    *DpE = malloc(sizeof(double) * (2 * BINS + 1));
    *h = malloc(sizeof(int) * (2 * BINS + 5));
    *g = malloc(sizeof(int) * (2 * BINS + 1));
    *hg = malloc(sizeof(int) * (2 * BINS + 5) * (2 * BINS + 1));

    return (*x && *p && *DxE && *DpE && *h && *g && *hg);
}

// Function to initialize histograms
void initialize_histograms(int BINS, int* h, int* g, int* hg, double* DxE, double* DpE, int N_PART) {
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i <= (BINS + 2) << 1; i++) {
        DxE[i] = 1.0E-3 * N_PART;
    }
    DxE[0] = DxE[1] = DxE[2 * BINS + 3] = DxE[2 * BINS + 4] = 0.0;
    DxE[2] *= 0.5;
    DxE[2 * BINS + 2] *= 0.5;

    memset(h, 0, (2 * BINS + 5) * sizeof(int));
    memset(g, 0, (2 * BINS + 1) * sizeof(int));
    memset(hg, 0, (2 * BINS + 5) * (2 * BINS + 1) * sizeof(int));

    #pragma omp parallel for reduction(+ : DpE[ : 2 * BINS + 1]) schedule(static)
    for (int i = 0; i <= BINS << 1; i++) {
        double numerator = 6.0E-26 * N_PART;
        double denominator = 5.24684E-24 * sqrt(2.0 * PI);
        double exponent = -pow(3.0e-23 * (1.0 * i / BINS - 1) / 5.24684E-24, 2) / 2;
        DpE[i] = (numerator / denominator) * exp(exponent);
    }
}

// Function to initialize particles
void initialize_particles(int N_PART, double* x, double* p, int BINS) {
    #pragma omp parallel
    {
        uint32_t seed = (uint32_t)(time(NULL) + omp_get_thread_num());

        #pragma omp for schedule(static)
        for (int i = 0; i < N_PART; i++) {
            x[i] = d_xorshift(&seed) * 0.5;
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < N_PART >> 1; i++) {
            double randomValue1 = d_xorshift(&seed);
            double randomValue2 = d_xorshift(&seed);
            double xi1 = sqrt(-2.0 * log(randomValue1 + 1E-35));
            double xi2 = 2.0 * PI * randomValue2;

            p[2 * i] = xi1 * cos(xi2) * 5.24684E-24;
            p[2 * i + 1] = xi1 * sin(xi2) * 5.24684E-24;
        }
    }
}

// Function to update histograms with particle data
void update_histograms(int N_PART, int BINS, double* x, double* p, int* h, int* g, int* hg) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_PART; i++) {
        int h_idx = floor((2.0 * x[i] + 1) * BINS + 2.5);
        int g_idx = floor((p[i] / 3.0e-23 + 1) * BINS + 0.5);
        int hg_idx = (2 * BINS + 1) * h_idx + g_idx;

        if (hg_idx > (2 * BINS) * (2 * BINS + 4) || hg_idx < 0) {
            printf("Error en el índice: hg_idx=%d\n", hg_idx);
        }

        h[h_idx]++;
        g[g_idx]++;
        hg[hg_idx]++;
    }
}

// Function to run the main simulation steps
void run_simulation(int N_PART, double DT, double M, int* steps, unsigned int Ntandas, int BINS, double* x, double* p) {
    for (unsigned int j = 0; j < Ntandas; j++) {
        #pragma omp parallel shared(x, p)
        {
            uint32_t seed = (uint32_t)(time(NULL) + omp_get_thread_num());

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < N_PART; ++i) {
                double x_tmp = x[i];
                double p_tmp = p[i];

                for (int step = 0; step < steps[j]; step++) {
                    x_tmp += p_tmp * DT / M;
                    int signop = copysign(1.0, p_tmp);
                    long k = trunc(x_tmp + 0.5 * signop);

                    if (k != 0) {
                        double xi1 = d_xorshift(&seed);
                        double xi2 = d_xorshift(&seed);
                        p_tmp += 0.25 * (sqrt(-2.0 * log(xi1)) * sin(2.0 * PI * xi2) - p_tmp) / 1.0e-12;
                        x_tmp -= signop;
                    }
                }

                x[i] = x_tmp;
                p[i] = p_tmp;
            }
        }
    }
}

int main() {
    // Declare variables for configuration and simulation
    int N_THREADS = 0, N_PART = 0, BINS = 0, steps[50], retake = 0, dump = 0;
    unsigned int Ntandas = 0u;
    double DT = 0.0, M = 0.0, sigmaL = 0.0;
    char inputFilename[255], saveFilename[255];

    // Load simulation parameters from file
    load_parameters_from_file("datos.in", &N_PART, &BINS, &DT, &M, &N_THREADS, &Ntandas, steps, inputFilename, saveFilename, &retake, &dump, &sigmaL);

    // Allocate memory for simulation arrays
    double *x, *p, *DxE, *DpE;
    int *h, *g, *hg;
    if (!allocate_memory(N_PART, BINS, &x, &p, &DxE, &DpE, &h, &g, &hg)) {
        return 1; // Exit if memory allocation failed
    }

    // Initialize histograms and particle arrays
    initialize_histograms(BINS, h, g, hg, DxE, DpE, N_PART);
    
    if (retake != 0) {
        while (true) {
            initialize_particles(N_PART, x, p, BINS);
            update_histograms(N_PART, BINS, x, p, h, g, hg);
            int X0 = make_hist(h, g, hg, DxE, DpE, "X0000000.dat", BINS);
            if (X0 != 1) break;
            printf("Falló algún chi2: X0=%1d\n", X0);
        }
    } else {
        read_data(inputFilename, x, p, &evolution, N_PART);
    }

    // Run the main simulation
    run_simulation(N_PART, DT, M, steps, Ntandas, BINS, x, p);

    // Free allocated memory
    free(x); free(p); free(DxE); free(DpE); free(h); free(g); free(hg);

    return 0;
}
