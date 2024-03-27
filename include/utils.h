#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BORDES 237 // para NPART = 2^21 (+ 100 cuentas)
#define PI 3.14159265358979323846
#define epsmax2M 9.0E-46 // E maxima
#define DEmax2M                                                                                                        \
    6.0e-50              // mas grande para pmax (3.5964e-48: 1 canal de p menos que el max)
                         // // OJO: ahora DEmax2M nos da 1.2e-50 (8/2/24) 6.296208e-49 //
                         // 2*5.24684E-24*6.0e-26
#define epsmin2M 9.0E-52 // 2m * E minima = pmin^2

double d_rand();

void load_parameters_from_file(char filename[], int *N_PART, int *BINS, double *DT, double *M, int *N_THREADS,
                               unsigned int *Ntandas, int steps[], char inputFilename[], char saveFilename[],
                               int *retake, int *dump, double *sigmaL);

void read_data(char filename[], double *x, double *p, int *evolution, int N_PART);

void save_data(char filename[], double *x, double *p, int evolution, int N_PART);

void energy_sum(double *p, int N_PART, int evolution, double M);


int make_hist(int *h, int *g, int *hg, double *DxE, double *DpE, const char *filename, int BINS);

#endif