#include "../include/utils.h"
#include <omp.h>


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
void initialize_particles(int N_PART, double* x, double* p) {
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


void run_simulation(int N_PART, double DT, double M, int* steps, unsigned int Ntandas, double* x, double* p, 
                    double* DxE, double* DpE, int* h, int* g, int* hg, int BINS, char* saveFilename, int dump, double sigmaL, int evolution) {

    // Initialize the necessary variables
    double d = 1.0e-72, alfa = 1.5E+42;
    double pmin = 3.0E-026, pmax = 3.0E-023;

    // Sum energy before starting the simulation
    energy_sum(p, N_PART, evolution, M);
    printf("d=%12.9E  alfa=%12.9E\n", d, alfa);

    for (unsigned int j = 0; j < Ntandas; j++) {
        long int k;
        int signop;
        
        #pragma omp parallel shared(x, p)
        {
            uint32_t seed = (uint32_t)(time(NULL) + omp_get_thread_num());
            
            #pragma omp for private(k, signop) schedule(dynamic)
            for (int i = 0; i < N_PART; ++i){
                double x_tmp = x[i];
                double p_tmp = p[i];
                
                for (int step = 0; step < steps[j]; step++) {
                    x_tmp += p_tmp * DT / M;    // Update particle position
                    
                    signop = copysign(1.0, p_tmp);
                    k = trunc(x_tmp + 0.5 * signop);
                    
                    if (k != 0) {
                        double randomValue = d_xorshift(&seed);
                        double xi1 = sqrt(-2.0 * log(randomValue + 1E-35));
                        randomValue = d_xorshift(&seed);
                        double xi2 = 2.0 * PI * randomValue;

                        double deltaX = sqrt(labs(k)) * xi1 * cos(xi2) * sigmaL;
                        deltaX = (fabs(deltaX) > 1.0 ? 1.0 * copysign(1.0, deltaX) : deltaX);
                        x_tmp = (k % 2 ? -1.0 : 1.0) * (x_tmp - k) + deltaX;

                        if (fabs(x_tmp) > 0.502) {
                            x_tmp = 1.004 * copysign(1.0, x_tmp) - x_tmp;
                        }

                        p_tmp = fabs(p_tmp);
                        for (int l = 1; l <= labs(k); l++) {
                            double DeltaE = alfa * pow((p_tmp - pmin) * (pmax - p_tmp), 2);
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

        // Update histograms
        #pragma omp for schedule(static)
        for (int i = 0; i < N_PART; i++) {
            int h_idx = floor((2.0 * x[i] + 1) * BINS + 2.5);
            int g_idx = floor((p[i] / 3.0e-23 + 1) * BINS + 0.5);
            int hg_idx = (2 * BINS + 1) * h_idx + g_idx;

            h[h_idx]++;
            g[g_idx]++;
            hg[hg_idx]++;
        }

        // Increment evolution
        evolution += steps[j];

        // Create filename based on evolution
        char filename[32];
        if (evolution < 10000000) {
            sprintf(filename, "X%07d.dat", evolution);
        } else {
            sprintf(filename, "X%1.3e.dat", (double)evolution);
            char *e = memchr(filename, 'e', 32);
            strcpy(e + 1, e + 3);
        }

        // Save data if needed
        if (dump == 0) {
            save_data(saveFilename, x, p, evolution, N_PART);
        }

        // Generate histograms
        make_hist(h, g, hg, DxE, DpE, filename, BINS);
        energy_sum(p, N_PART, evolution, M);
    }
}


int main() {
    // ============= Initialize variables =============
    int N_THREADS = 0, N_PART = 0, BINS = 0, steps[50], retake = 0, dump = 0;
    unsigned int Ntandas = 0u;
    char inputFilename[255], saveFilename[255];
    double DT = 0.0, M = 0.0, sigmaL = 0.0;

    int evolution = 0;

    load_parameters_from_file("datos.in", &N_PART, &BINS, &DT, &M, &N_THREADS, &Ntandas, steps, inputFilename,
                              saveFilename, &retake, &dump, &sigmaL);

    // Allocate memory for simulation arrays
    double *x, *p, *DxE, *DpE;
    int *h, *g, *hg;
    if (!allocate_memory(N_PART, BINS, &x, &p, &DxE, &DpE, &h, &g, &hg)) {
        return 1; // Exit if memory allocation failed
    }

    // Initialize histograms and particle arrays
    initialize_histograms(BINS, h, g, hg, DxE, DpE, N_PART);

    // ============= Initialize particles arrays =============
    if (retake != 0) {
        while (true) {
            initialize_particles(N_PART, x, p);
            update_histograms(N_PART, BINS, x, p, h, g, hg);
            int X0 = make_hist(h, g, hg, DxE, DpE, "X0000000.dat", BINS);
            if (X0 != 1) break;
            printf("Falló algún chi2: X0=%1d\n", X0);
        }
    } else {
        read_data(inputFilename, x, p, &evolution, N_PART);
    }

    // Run the main simulation
    run_simulation(N_PART, DT, M, steps, Ntandas, x, p, DxE, DpE,  h, g, hg, BINS, saveFilename, dump, sigmaL, evolution);

    printf("Completo evolution = %d\n", evolution);

    // Free allocated memory
    free(x); free(p); free(DxE); free(DpE); free(h); free(g); free(hg);

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
