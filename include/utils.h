#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BORDES 237 // para NPART = 2^21 (+ 100 cuentas)

double d_rand();

void load_parameters_from_file(char filename[],
                               int *N_PART,
                               int *BINS,
                               double *DT,
                               double *M,
                               int *N_THREADS,
                               unsigned int *Ntandas,
                               int steps[],
                               char indat[],
                               char saldat[],
                               int *retoma,
                               int *dump,
                               double *sigmaL);

void read_data(char filename[], double *x, double *p, int *evolution, int N_PART);

void save_data(char filename[], double *x, double *p, int evolution, int N_PART);

void energy_sum(double *p, int N_PART, int evolution, double M);

// avanza n pasos en el rango de partículas [s, e)
void iter_in_range(int n, int s, int e, double *x, double *p, double DT, double M, double alfa, double pmin075, double pmax075);

int make_hist(int *h, int *g, int *hg, double *DxE, double *DpE, const char *filename, int BINS);

#endif