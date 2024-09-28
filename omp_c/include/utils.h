#ifndef UTILS_H
#define UTILS_H

#include "constants.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

using namespace std;

// XORShift random number generator inline function
inline double d_xorshift(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return static_cast<double>(x) / static_cast<double>(UINT32_MAX);
}

// Function to load parameters from a file
void load_parameters_from_file(char filename[], int *N_PART, int *BINS, double *DT, double *M, int *N_THREADS,
                               unsigned int *Ntandas, int steps[], char inputFilename[], char saveFilename[],
                               int *resume, int *dump, double *sigmaL);

// Function to read data from a file into arrays x and p
void read_data(char filename[], double *x, double *p, unsigned int *evolution, int N_PART);

// Function to save data from arrays x and p to a file
void save_data(const char filename[], double *x, double *p, unsigned int evolution, int N_PART);

// Function to compute the sum of energies in the array p
double energy_sum(double *p, int N_PART, unsigned int evolution, double M);

// Function to generate histograms based on input data and save them to a file
int make_hist(int *h, int *g, int *hg, double *DxE, double *DpE, const char *filename, int BINS, double Et);

#endif
