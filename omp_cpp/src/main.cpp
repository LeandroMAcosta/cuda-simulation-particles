#include "../include/utils.h"
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>

int main()
{
    int N_THREADS = 0, N_PART = 0, BINS = 0, steps[500], resume = 0, dump = 0;
    unsigned int Ntandas = 0u;
    char inputFilename[255], saveFilename[255];
    double DT = 0.0, M = 0.0, sigmaL = 0.0;

    double xi1 = 0.0, xi2 = 0.0;
    int X0 = 1;
    char filename[32];

    double d = 1.0e-72, alfa = 1.0e-4;
    unsigned int evolution = 0u;
    double pmin = 2.0E-026, pmax = 3.0E-023;

    char data_filename[] = "datos.in";

    load_parameters_from_file(data_filename, &N_PART, &BINS, &DT, &M, &N_THREADS, &Ntandas, steps, inputFilename,
                              saveFilename, &resume, &dump, &sigmaL);

    // Using malloc for arrays as requested
    double *x = static_cast<double *>(malloc(sizeof(double) * N_PART));
    double *p = static_cast<double *>(malloc(sizeof(double) * N_PART));
    double *DxE = static_cast<double *>(malloc(sizeof(double) * (2 * BINS + 4)));
    double *DpE = static_cast<double *>(malloc(sizeof(double) * (2 * BINS)));
    int *h = static_cast<int *>(malloc(sizeof(int) * (2 * BINS + 4)));
    int *g = static_cast<int *>(malloc(sizeof(int) * (2 * BINS)));
    int *hg = static_cast<int *>(malloc(sizeof(int) * (2 * BINS + 4) * (2 * BINS)));

    // Parallel loop to calculate DpE array using OpenMP
    #pragma omp parallel for reduction(+ : DpE[:2 * BINS]) schedule(static)
    for (int i = 0; i < BINS << 1; i++) {
        double numerator = 6.0E-26 * N_PART;
        double denominator = 5.24684E-24 * sqrt(2.0 * M_PI);
        double exponent = -pow(3.0e-23 * (1.0 * i / BINS - 0.999) / 5.24684E-24, 2) / 2;
        DpE[i] = (numerator / denominator) * exp(exponent);
    }

    // Parallel loop to calculate DxE array using OpenMP SIMD
    #pragma omp parallel for simd schedule(static)
    for (int i = 2; i < (BINS + 1) << 1; i++) {
        DxE[i] = 1.0E-3 * N_PART;
    }

    DxE[0] = 0.0;
    DxE[1] = 0.0;
    DxE[2 * BINS + 2] = 0.0;
    DxE[2 * BINS + 3] = 0.0;

    // Initialize h, g, hg arrays using memset
    memset(h, 0, (2 * BINS + 4) * sizeof(int));
    memset(g, 0, (2 * BINS) * sizeof(int));
    memset(hg, 0, (2 * BINS + 4) * (2 * BINS) * sizeof(int));

    // Check for resume condition
    if (resume != 0) {
        while (X0 == 1) {
            // Initialize particles
            #pragma omp parallel
            {
                uint32_t seed = static_cast<uint32_t>(time(NULL) + omp_get_thread_num());
                #pragma omp for schedule(static)
                for (int i = 0; i < N_PART; i++)
                {
                    double randomValue = d_xorshift(&seed);
                    x[i] = randomValue * 0.5;
                }
                #pragma omp for schedule(static)
                for (int i = 0; i < N_PART >> 1; i++)
                {
                    double randomValue1 = d_xorshift(&seed);
                    double randomValue2 = d_xorshift(&seed);

                    xi1 = sqrt(-2.0 * log(randomValue1 + 1E-35));
                    xi2 = 2.0 * M_PI * randomValue2;

                    p[2 * i] = xi1 * cos(xi2) * 5.24684E-24;
                    p[2 * i + 1] = xi1 * sin(xi2) * 5.24684E-24;
                }
            }

            // Parallel loop to update histograms
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N_PART; i++) {
                int h_idx = static_cast<int>(floor((x[i] + 0.5) * (1.99999999999999 * BINS) + 2.0));
                int g_idx = static_cast<int>(floor((p[i] / 3.0e-23 + 1) * (0.999999999999994 * BINS)));
                int hg_idx = (2 * BINS) * h_idx + g_idx;
                h[h_idx]++;
                g[g_idx]++;
                hg[hg_idx]++;
            }

            double Et = energy_sum(p, N_PART, evolution, M);
            X0 = make_hist(h, g, hg, DxE, DpE, "X0000000.dat", BINS, Et);
            if (X0 == 1) {
                cout << "Falló algún chi2: X0=" << X0 << endl;
            }
        }
    } else {
        // If not resuming, read data
        read_data(inputFilename, x, p, &evolution, N_PART);
    }

    double Et = energy_sum(p, N_PART, evolution, M);
    cout << "pmin=" << scientific << pmin << " d=" << d << " alfa=" << alfa << " Et=" << Et << endl;

    // Main loop to iterate through Ntandas
    for (unsigned int j = 0; j < Ntandas; j++) {
        long int k;
        int signop;

        #pragma omp parallel shared(x, p)
        {
            uint32_t seed = static_cast<uint32_t>(time(NULL) + omp_get_thread_num());

            #pragma omp for private(k, signop) schedule(dynamic)
            for (int i = 0; i < N_PART; ++i) {
                double x_tmp = x[i];
                double p_tmp = p[i];

                for (int step = 0; step < steps[j]; step++) {
                    x_tmp += p_tmp * DT / M;
                    signop = copysign(1.0, p_tmp);
                    k = trunc(x_tmp + 0.5 * signop);

                    if (k != 0) {
                        double randomValue = d_xorshift(&seed);
                        xi1 = sqrt(-2.0 * log(randomValue + 1E-35));
                        randomValue = d_xorshift(&seed);
                        xi2 = 2.0 * M_PI * randomValue;
                        double deltaX = sqrt(labs(k)) * xi1 * cos(xi2) * sigmaL;
                        deltaX = (fabs(deltaX) > 1.0 ? 1.0 * copysign(1.0, deltaX) : deltaX);
                        x_tmp = (k % 2 ? -1.0 : 1.0) * (x_tmp - k) + deltaX;

                        if (fabs(x_tmp) > 0.502) {
                            x_tmp = 1.004 * copysign(1.0, x_tmp) - x_tmp;
                        }
                        p_tmp = fabs(p_tmp);

                        for (int l = 1; l <= labs(k); l++) {
                            double DeltaE = alfa * (p_tmp - pmin) * (pmax - p_tmp);
                            randomValue = d_xorshift(&seed);
                            p_tmp = sqrt(p_tmp * p_tmp + DeltaE * (randomValue - 0.5));
                        }
                        p_tmp *= (k % 2 ? -1.0 : 1.0) * signop;
                    }
                }

                x[i] = x_tmp;
                p[i] = p_tmp;
            }
        }

        // Parallel loop to update histograms
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N_PART; i++) {
            int h_idx = static_cast<int>(floor((x[i] + 0.5) * (1.99999999999999 * BINS) + 2.0));
            int g_idx = static_cast<int>(floor((p[i] / 3.0e-23 + 1) * (0.999999999999994 * BINS)));
            int hg_idx = (2 * BINS) * h_idx + g_idx;
            h[h_idx]++;
            g[g_idx]++;
            hg[hg_idx]++;
        }

        evolution += steps[j];
        if (evolution < 10000000) {
            sprintf(filename, "X%07d.dat", evolution);
        } else {
            sprintf(filename, "X%1.3e.dat", static_cast<double>(evolution));
            char *e = static_cast<char*>(memchr(filename, 'e', 32)); // Explicit cast to char*
            if (e) {
                strcpy(e + 1, e + 3); // Adjusting the position after 'e'
            }
        }

        if (dump == 0) {
            save_data(saveFilename, x, p, evolution, N_PART);
        }

        Et = energy_sum(p, N_PART, evolution, M);
        make_hist(h, g, hg, DxE, DpE, filename, BINS, Et);
    }

    cout << "Completo evolution = " << evolution << endl;

    // Free allocated memory
    free(x);
    free(p);
    free(DxE);
    free(DpE);
    free(h);
    free(g);
    free(hg);

    return 0;
}
