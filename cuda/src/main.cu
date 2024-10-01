#include <cmath>
#include <cstring>
#include <cstdio>
#include <omp.h>
#include <iostream>
#include <cuda_runtime.h>

#include "./include/utils.h"
#include "./include/histogram_kernels.h"

using namespace std;

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

    // Unified Memory Allocation for arrays using cudaMallocManaged
    double *x, *p, *DxE, *DpE;
    int *h, *g, *hg;

    cudaMallocManaged(&x, sizeof(double) * N_PART);
    cudaMallocManaged(&p, sizeof(double) * N_PART);
    cudaMallocManaged(&DxE, sizeof(double) * (2 * BINS + 4));
    cudaMallocManaged(&DpE, sizeof(double) * (2 * BINS));
    cudaMallocManaged(&h, sizeof(int) * (2 * BINS + 4));
    cudaMallocManaged(&g, sizeof(int) * (2 * BINS));
    cudaMallocManaged(&hg, sizeof(int) * (2 * BINS + 4) * (2 * BINS));

    // Launch CUDA kernel for parallel DpE computation
    int threadsPerBlock = 256;

    int blocksPerGridForDpE = (2 * BINS + threadsPerBlock - 1) / threadsPerBlock;
    calculateDpE<<<blocksPerGridForDpE, threadsPerBlock>>>(DpE, N_PART, BINS);

    // Launch CUDA kernel for DxE calculation
    int blocksPerGridForDxE = (2 * BINS + threadsPerBlock - 1) / threadsPerBlock;
    calculateDxE<<<blocksPerGridForDxE, threadsPerBlock>>>(DxE, N_PART, BINS);

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

    // Free memory
    cudaFree(x);
    cudaFree(p);
    cudaFree(DxE);
    cudaFree(DpE);
    cudaFree(h);
    cudaFree(g);
    cudaFree(hg);

    return 0;
}



/* para graficar en el gnuplot: (sacando un archivo "hists.eps")
set terminal postscript enhanced color eps 20
set output "hists.eps"
# histograma de x  (las dos líneas siguientes alcanzan para graficar las x
dentro del gnuplot) set style fill solid 1.0 # o medio transparente: set style
fill transparent solid 0.5 noborder set key left ; set xrange[-0.5:0.5] p
'X1000000.dat' u 1:2 w boxes lc rgb "#dddddd" t 'X1000000.dat' , 'X2000000.dat'
u 1:2 w boxes lc rgb "#77ff77" t 'X2000000.dat' , 'X2000001.dat' u 1:2 w boxes
lc "#ffaaaa" t 'X2000001.dat' , 'X2000002.dat' u 1:2 w boxes lc "#dddd55" t
'X2000002.dat' , 'X2000003.dat' u 1:2 w boxes lc rgb "#ffdddd" t 'X2000003.dat'
, 'X2000008.dat' u 1:2 w boxes lc rgb "#cc44ff" t 'X2000008.dat' ,
'X2000018.dat' u 1:2 w boxes lc rgb "#888888" t 'X2000018.dat' , 'X2000028.dat'
u 1:2 w boxes lc rgb "#bbddbb" t 'X2000028.dat' , 'X2000038.dat' u 1:2 w boxes
lc rgb "#ffee00" t 'X2000038.dat' , 'X2000048.dat' u 1:2 w boxes lc rgb
"#8844ff" t 'X2000048.dat' , 'X2000058.dat' u 1:2 w boxes lc rgb "#cceeff" t
'X2000058.dat' , 'X2000068.dat' u 1:2 w boxes lc rgb "#44bb44" t 'X2000068.dat'
, 'X2000078.dat' u 1:2 w boxes lc rgb "#99ee77" t 'X2000078.dat' ,
'X2000088.dat' u 1:2 w boxes lc rgb "#ffdd66" t 'X2000088.dat' , 'X2000098.dat'
u 1:2 w boxes lc rgb "#4444ff" t 'X2000098.dat' # histograma de p  (las dos
líneas siguientes alcanzan para graficar las p dentro del gnuplot) set key left
; set xrange[-3e-23:3e-23] p 'X0000500.dat' u 3:4 w boxes lc rgb "#dddddd" t
'X0000500.dat' , 'X0001000.dat' u 3:4 w boxes lc rgb "#77ff77" t 'X0001000.dat'
, 'X0002000.dat' u 3:4 w boxes lc "#ffaaaa" t 'X0002000.dat' , 'X0005000.dat' u
3:4 w boxes lc "#dddd55" t 'X0005000.dat' , 'X0010000.dat' u 3:4 w boxes lc rgb
"#ffdddd" t 'X0010000.dat' , 'X0020000.dat' u 3:4 w boxes lc rgb "#cc44ff" t
'X0020000.dat' , 'X0050000.dat' u 3:4 w boxes lc rgb "#888888" t 'X0050000.dat'
, 'X0100000.dat' u 3:4 w boxes lc rgb "#bbddbb" t 'X0100000.dat' ,
'X0200000.dat' u 3:4 w boxes lc rgb "#ffee00" t 'X0200000.dat' , 'X0500000.dat'
u 3:4 w boxes lc rgb "#8844ff" t 'X0500000.dat' , 'X0995000.dat' u 3:4 w boxes
lc rgb "#cceeff" t 'X0995000.dat' , 'X0999000.dat' u 3:4 w boxes lc rgb
"#44bb44" t 'X0999000.dat' , 'X0999500.dat' u 3:4 w boxes lc rgb "#99ee77" t
'X0999500.dat' , 'X1000000.dat' u 3:4 w boxes lc rgb "#ffdd66" t 'X1000000.dat'
, 'X2000000.dat' u 3:4 w boxes lc rgb "#4444ff" t 'X2000000.dat' set terminal qt


p 'X0000001.dat' u 1:2 w boxes lc rgb "#dddddd" t 'X0000001.dat' ,
'X0000100.dat' u 1:2 w boxes lc rgb "#77ff77" t 'X0000100.dat' , 'X0001000.dat'
u 1:2 w boxes lc "#ffaaaa" t 'X0001000.dat' , 'X0001200.dat' u 1:2 w boxes lc
"#dddd55" t 'X0001200.dat' , 'X0001400.dat' u 1:2 w boxes lc rgb "#ffdddd" t
'X0001400.dat' , 'X0001500.dat' u 1:2 w boxes lc rgb "#cc44ff" t 'X0001500.dat'
, 'X0001600.dat' u 1:2 w boxes lc rgb "#888888" t 'X0001600.dat' ,
'X0001700.dat' u 1:2 w boxes lc rgb "#bbddbb" t 'X0001700.dat' , 'X0001800.dat'
u 1:2 w boxes lc rgb "#ffee00" t 'X0001800.dat' , 'X0001900.dat' u 1:2 w boxes
lc rgb "#8844ff" t 'X0001900.dat' , 'X0002000.dat' u 1:2 w boxes lc rgb
"#cceeff" t 'X0002000.dat'

*/