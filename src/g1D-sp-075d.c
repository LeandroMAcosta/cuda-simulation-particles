#include "../include/utils.h"
#include <omp.h>

/* Compilar usando el Makefile */

int main()
{
    omp_set_num_threads(omp_get_num_procs());
    
    int N_THREADS = 0, N_PART = 0, BINS = 0, steps[50], retake = 0, dump = 0;
    unsigned int Ntandas = 0u;
    char inputFilename[255], saveFilename[255];
    double DT = 0.0, M = 0.0, sigmaL = 0.0;

    double xi1 = 0.0, xi2 = 0.0;
    int X0 = 1;
    char filename[32];

    double d = 1.0e-72, alfa = 4.0E-88;
    int evolution = 0;
    double pmin075 = 7.20843424240426E-020, pmax075 = 1.2818610191887E-017;

    char data_filename[] = "datos.in";

    load_parameters_from_file(data_filename, &N_PART, &BINS, &DT, &M, &N_THREADS, &Ntandas, steps, inputFilename,
                              saveFilename, &retake, &dump, &sigmaL);

    double *x = malloc(sizeof(double) * N_PART);
    double *p = malloc(sizeof(double) * N_PART);
    double *DxE = malloc(sizeof(double) * (2 * BINS + 5));
    double *DpE = malloc(sizeof(double) * (2 * BINS + 1));
    int *h = malloc(sizeof(int) * (2 * BINS + 5));
    int *g = malloc(sizeof(int) * (2 * BINS + 1));
    int *hg = malloc(sizeof(int) * (2 * BINS + 5) * (2 * BINS + 1));

#pragma omp parallel for reduction(+ : DpE[ : 2 * BINS + 1])
    for (int i = 0; i <= BINS << 1; i++)
    {
        double numerator = 6.0E-26 * N_PART;
        double denominator = 5.24684E-24 * sqrt(2.0 * PI);
        double exponent = -pow(3.0e-23 * (1.0 * i / BINS - 1) / 5.24684E-24, 2) / 2;
        DpE[i] = numerator / denominator * exp(exponent);
    }

#pragma omp parallel for simd
    for (int i = 0; i <= (BINS + 2) << 1; i++)
    {
        DxE[i] = 1.0E-3 * N_PART;
    }

    DxE[0] = 0.0;
    DxE[1] = 0.0;
    DxE[2] = DxE[2] * 0.5;
    DxE[2 * BINS + 2] = DxE[2 * BINS + 2] * 0.5;
    DxE[2 * BINS + 3] = 0.0;
    DxE[2 * BINS + 4] = 0.0;

    memset(h, 0, (2 * BINS + 5) * sizeof(int));
    memset(g, 0, (2 * BINS + 1) * sizeof(int));
    memset(hg, 0, (2 * BINS + 5) * (2 * BINS + 1) * sizeof(int));

    if (retake != 0)
    {
        while (X0 == 1)
        {
// initialize particles
#pragma omp parallel
            {
                unsigned int seed = (unsigned int)(time(NULL) + omp_get_thread_num());
#pragma omp for
                for (int i = 0; i < N_PART; i++)
                {
                    double randomValue = (double)rand_r(&seed) / ((double)RAND_MAX + 1);
                    x[i] = randomValue * 0.5;
                }
#pragma omp for
                for (int i = 0; i < N_PART >> 1; i++)
                {
                    double randomValue1 = (double)rand_r(&seed) / ((double)RAND_MAX + 1);
                    double randomValue2 = (double)rand_r(&seed) / ((double)RAND_MAX + 1);

                    xi1 = sqrt(-2.0 * log(randomValue1 + 1E-35));
                    xi2 = 2.0 * PI * randomValue2;

                    p[2 * i] = xi1 * cos(xi2) * 5.24684E-24;
                    p[2 * i + 1] = xi1 * sin(xi2) * 5.24684E-24;
                }
            }

#pragma omp parallel for
            for (int i = 0; i < N_PART; i++)
            {
                h[(int)((2.0 * x[i] + 1) * BINS + 2.5)]++;
                g[(int)((p[i] / 3.0e-23 + 1) * BINS + 0.5)]++;
                hg[(2 * BINS + 1) * (int)((2.0 * x[i] + 1) * BINS + 2.5) + (int)((p[i] / 3e-23 + 1) * BINS + 0.5)]++;
            }

            X0 = make_hist(h, g, hg, DxE, DpE, "X0000000.dat", BINS);
            if (X0 == 1)
            {
                printf("Falló algún chi2: X0=%1d\n", X0);
            }
        }
    }
    else
    {
        read_data(inputFilename, x, p, &evolution, N_PART);
    }

    energy_sum(p, N_PART, evolution, M);
    printf("d=%12.9E  alfa=%12.9E\n", d, alfa);

    // Work code here:
    for (unsigned int j = 0; j < Ntandas; j++)
    {
        // iter_in_range code here:
        long int k;
        int signop;
#pragma omp parallel shared(x, p)
        {
            unsigned int seed = (unsigned int)(time(NULL) + omp_get_thread_num());
#pragma omp for private(k, signop) schedule(static)
            for (int i = 0; i < N_PART; ++i)
            {
                double x_tmp = x[i], p_tmp = p[i];
                for (int step = 0; step < steps[j]; step++)
                {
                    x_tmp += p_tmp * DT / M;
                    signop = copysign(1.0, p_tmp);
                    k = trunc(x_tmp + 0.5 * signop);
                    if (k != 0)
                    {
                        x_tmp = (k % 2 ? -1.0 : 1.0) * (x_tmp - k);
                        if (fabs(x_tmp) > 0.502)
                        {
                            x_tmp = 1.004 * copysign(1.0, x_tmp) - x_tmp;
                        }
                        for (int l = 1; l <= labs(k); l++)
                        {
                            double ptmp075 = pow(fabs(p_tmp), 0.75);
                            double DeltaE = alfa * pow((ptmp075 - pmin075) * (pmax075 - ptmp075), 4);
                            double randomValue = (double)rand_r(&seed) / ((double)RAND_MAX + 1);
                            p_tmp = sqrt(p_tmp * p_tmp + DeltaE * (randomValue - 0.5));
                        }
                        p_tmp *= (k % 2 ? -1.0 : 1.0) * signop;
                    }
                }
                x[i] = x_tmp;
                p[i] = p_tmp;
            }
        }
        // End of iter_in_range code.

        for (int i = 0; i < N_PART; i++)
        {
            int h_idx = (int)((2.0 * x[i] + 1) * BINS + 2.5);
            int g_idx = (int)((p[i] / 3.0e-23 + 1) * BINS + 0.5);
            h[h_idx]++;
            g[g_idx]++;
            hg[(2 * BINS + 1) * h_idx + g_idx]++;
        }
    }
    // End of Work code.

    for (unsigned int i = 0; i < Ntandas; i++)
    {
        evolution += steps[i];
        if (evolution < 10000000)
        {
            sprintf(filename, "X%07d.dat", evolution);
        }
        else
        {
            sprintf(filename, "X%1.3e.dat", (double)evolution);
            char *e = memchr(filename, 'e', 32);
            strcpy(e + 1, e + 3);
        }
        if (dump == 0)
        {
            save_data(saveFilename, x, p, evolution, N_PART);
        }
        make_hist(h, g, hg, DxE, DpE, filename, BINS);
        energy_sum(p, N_PART, evolution, M);
    }
    printf("Completo evolution = %d\n", evolution);

    free(x);
    free(p);
    free(DxE);
    free(DpE);
    free(h);
    free(g);
    free(hg);

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