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

#include "../include/types.h"

void load_parameters_from_file(char filename[], int *N_PART, int *BINS, RealTypeConstant *DT, RealTypeConstant *M, int *N_THREADS,
                               unsigned int *Ntandas, int steps[], char inputFilename[], char saveFilename[],
                               bool *resume, bool *dump, RealTypeConstant *sigmaL);

void read_data(char filename[], RealTypeX *h_x, RealTypeP *h_p, unsigned int *evolution, int N_PART);

void save_data(char filename[], RealTypeX *h_x, RealTypeP *h_p, unsigned int evolution, int N_PART);

double energy_sum(RealTypeP *d_p, int N_PART, unsigned int evolution, RealTypeConstant M);

int make_hist(int *h_h, int *h_g, int *h_hg, int *d_h, int *d_g, int *d_hg, RealTypeX *d_DxE, RealTypeP *DpE, const char *filename, int BINS, double Et);

#endif