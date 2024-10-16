#ifndef UTILS_H
#define UTILS_H

#include "constants.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

void load_parameters_from_file(char filename[], int *N_PART, int *BINS, double *DT, double *M, int *N_THREADS,
                               unsigned int *Ntandas, int steps[], char inputFilename[], char saveFilename[],
                               int *resume, int *dump, double *sigmaL);

void read_data(char filename[], double *x, double *p, unsigned int *evolution, int N_PART);

void save_data(char filename[], double *x, double *p, unsigned int evolution, int N_PART);

double energy_sum(double *p, int N_PART, unsigned int evolution, double M);

int make_hist(int *h, int *g, int *hg, float *d_DxE, double *DpE, const char *filename, int BINS, double Et);

#endif