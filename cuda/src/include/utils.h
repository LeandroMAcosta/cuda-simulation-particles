#ifndef UTILS_H
#define UTILS_H

#include "constants.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

inline double d_xorshift(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return (double)x / (double)UINT32_MAX;
}

void load_parameters_from_file(char filename[], int *N_PART, int *BINS, float *DT, float *M, int *N_THREADS,
                               unsigned int *Ntandas, int steps[], char inputFilename[], char saveFilename[],
                               bool *resume, bool *dump, float *sigmaL);

void read_data(char filename[], float *h_x, double *h_p, unsigned int *evolution, int N_PART);

void save_data(char filename[], float *h_x, double *h_p, unsigned int evolution, int N_PART);

float energy_sum(double *d_p, int N_PART, unsigned int evolution, float M);

int make_hist(int *h_h, int *h_g, int *h_hg, int *d_h, int *d_g, int *d_hg, float *d_DxE, double *DpE, const char *filename, int BINS, float Et);

#endif