#include "../include/utils.h"
#include <omp.h>


// Function to allocate memory for simulation arrays
bool allocate_memory(int N_PART, int BINS, double** x, double** p, double** DxE, double** DpE, int** h, int** g, int** hg) {

    /* Allocate memory for simulation arrays */
    *x = (double*) malloc(sizeof(double) * N_PART);           // Positions of particles
    *p = (double*) malloc(sizeof(double) * N_PART);           // Momenta of particles
    *DxE = (double*) malloc(sizeof(double) * (2 * BINS + 4)); // Histogram array for positions
    *DpE = (double*) malloc(sizeof(double) * (2 * BINS));     // Histogram array for momenta
    *h = (int*) malloc(sizeof(int) * (2 * BINS + 4));         // Position histogram counts
    *g = (int*) malloc(sizeof(int) * (2 * BINS));             // Momentum histogram counts
    *hg = (int*) malloc(sizeof(int) * (2 * BINS + 4) * (2 * BINS)); // 2D histogram for positions and momenta

    return (*x && *p && *DxE && *DpE && *h && *g && *hg);
}


void initialize_histograms(int BINS, int* h, int* g, int* hg, double* DxE, double* DpE, int N_PART) {
    /* Initialize histograms for momentum distribution */
    #pragma omp parallel for reduction(+ : DpE[ : 2 * BINS]) schedule(static)
    for (int i = 0; i < BINS << 1; i++) {
        // Calculate Gaussian distribution for momentum
        double numerator = 6.0E-26 * N_PART;
        double denominator = 5.24684E-24 * sqrt(2.0 * PI);
        double exponent = -pow(3.0e-23 * (1.0 * i / BINS - 0.999) / 5.24684E-24, 2) / 2;
        DpE[i] = (numerator / denominator) * exp(exponent);
    }

    /* Initialize histograms for position distribution */
    #pragma omp parallel for simd schedule(static)
    for (int i = 2; i < (BINS + 1) << 1; i++) {
        DxE[i] = 1.0E-3 * N_PART;  // Set position histogram with initial values
    }

    // Set boundary conditions for position histograms
    DxE[0] = 0.0;
    DxE[1] = 0.0;
    DxE[2 * BINS + 2] = 0.0;
    DxE[2 * BINS + 3] = 0.0;

    /* Initialize the histograms (h, g, hg) to zero */
    memset(h, 0, (2 * BINS + 4) * sizeof(int));
    memset(g, 0, (2 * BINS) * sizeof(int));
    memset(hg, 0, (2 * BINS + 4) * (2 * BINS) * sizeof(int));
}


void initialize_particles_and_histogram(int N_PART, double* x, double* p, int* h, int* g, int* hg, double* DxE, double* DpE, unsigned int evolution, double M, int BINS) {
    int X0 = 1;                                 // Control variable for resuming simulation
    double xi1 = 0.0, xi2 = 0.0;                // Temporary variables for random numbers

    while (X0 == 1) {
        // Initialize particles' positions and momenta
        #pragma omp parallel
        {
            uint32_t seed = (uint32_t)(time(NULL) + omp_get_thread_num());  // Seed for random number generation

            // Initialize particle positions
            #pragma omp for schedule(static)
            for (int i = 0; i < N_PART; i++) {
                double randomValue = d_xorshift(&seed);  // Generate random position
                x[i] = randomValue * 0.5;
            }

            // Initialize particle momenta
            #pragma omp for schedule(static)
            for (int i = 0; i < N_PART >> 1; i++) {
                double randomValue1 = d_xorshift(&seed);
                double randomValue2 = d_xorshift(&seed);

                // Box-Muller transform to generate random momentum values
                xi1 = sqrt(-2.0 * log(randomValue1 + 1E-35));
                xi2 = 2.0 * PI * randomValue2;

                p[2 * i] = xi1 * cos(xi2) * 5.24684E-24;
                p[2 * i + 1] = xi1 * sin(xi2) * 5.24684E-24;
            }
        }

        /* Update histograms based on particle positions and momenta */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N_PART; i++) {
            int h_idx = floor((x[i] + 0.5) * (1.99999999999999 * BINS) + 2.0);
            int g_idx = floor((p[i] / 3.0e-23 + 1) * (0.999999999999994 * BINS));
            int hg_idx = (2 * BINS) * h_idx + g_idx;
            h[h_idx]++;
            g[g_idx]++;
            hg[hg_idx]++;
        }

        // Calculate total energy and generate histogram file
        double Et = energy_sum(p, N_PART, evolution, M);
        X0 = make_hist(h, g, hg, DxE, DpE, "X0000000.dat", BINS, Et);
        if (X0 == 1) {
            printf("Error: Chi-square test failed: X0=%1d\n", X0);
        }
    }
}

int main() {
    // Initialize basic simulation variables
    int N_THREADS = 0, N_PART = 0, BINS = 0;               // Number of threads, number of particles, number of bins
    int steps[500], resume = 0, dump = 0;                  // Steps array, resume flag, dump flag
    unsigned int Ntandas = 0u;                             // Number of simulation rounds
    char inputFilename[255], saveFilename[255];            // Input and save filenames
    double DT = 0.0, M = 0.0, sigmaL = 0.0;                // Time step, mass, and sigmaL parameter for calculations

    // Random number generator and filename parameters
    char filename[32];                                     // Filename for saving data

    // Simulation constants
    double d = 1.0e-72, alfa = 1.0e-4;                     // Constants d and alfa for energy updates
    unsigned int evolution = 0u;                           // Tracks the evolution of the system
    double pmin = 2.0E-026, pmax = 3.0E-023;               // Minimum and maximum momentum values

    // Define the filename for the input data file
    char data_filename[] = "datos.in";                     // Input file containing simulation parameters

    // Load simulation parameters from input file
    load_parameters_from_file(data_filename, &N_PART, &BINS, &DT, &M, &N_THREADS, &Ntandas, steps, inputFilename,
                              saveFilename, &resume, &dump, &sigmaL);

    // Allocate memory for simulation arrays
    double *x, *p, *DxE, *DpE;
    int *h, *g, *hg;
    if (!allocate_memory(N_PART, BINS, &x, &p, &DxE, &DpE, &h, &g, &hg)) {
        return 1; // Exit if memory allocation failed
    }

    // Initialize histograms and particle arrays
    initialize_histograms(BINS, h, g, hg, DxE, DpE, N_PART);

    // Resume simulation if required
    if (resume != 0) {
        // Initialize particles and histograms
        initialize_particles_and_histogram(N_PART, x, p, h, g, hg, DxE, DpE, evolution, M, BINS);
    } else {
        // Load particle data from input file if
        read_data(inputFilename, x, p, &evolution, N_PART);
    }

    /* Calculate the initial total energy */
    double Et = energy_sum(p, N_PART, evolution, M);
    printf("pmin=%12.9E      d=%9.6E     alfa=%12.9E   Et=%12.9E\n", pmin, d, alfa, Et);

    /* ========= Start the main simulation loop ========= */
    printf("Starting simulation...\n");
    double xi1 = 0.0, xi2 = 0.0;                // Temporary variables for random numbers
    for (unsigned int j = 0; j < Ntandas; j++) {
        // Iterate through the simulation steps for this round
        long int k;
        int signop;
        #pragma omp parallel shared(x, p)
        {
            uint32_t seed = (uint32_t)(time(NULL) + omp_get_thread_num());  // Seed for random numbers

            // Update particle positions and momenta for each step
            #pragma omp for private(k, signop) schedule(dynamic)
            for (int i = 0; i < N_PART; ++i) {
                double x_tmp = x[i];
                double p_tmp = p[i];

                for (int step = 0; step < steps[j]; step++) {
                    x_tmp += p_tmp * DT / M;  // Update position based on momentum
                    signop = copysign(1.0, p_tmp);  // Determine sign of momentum
                    k = trunc(x_tmp + 0.5 * signop);

                    if (k != 0) {
                        // Apply random fluctuations to particle motion
                        double randomValue = d_xorshift(&seed);
                        xi1 = sqrt(-2.0 * log(randomValue + 1E-35));
                        randomValue = d_xorshift(&seed);
                        xi2 = 2.0 * PI * randomValue;
                        double deltaX = sqrt(labs(k)) * xi1 * cos(xi2) * sigmaL;
                        deltaX = (fabs(deltaX) > 1.0 ? 1.0 * copysign(1.0, deltaX) : deltaX);
                        x_tmp = (k % 2 ? -1.0 : 1.0) * (x_tmp - k) + deltaX;
                        if (fabs(x_tmp) > 0.502) {
                            x_tmp = 1.004 * copysign(1.0, x_tmp) - x_tmp;
                        }

                        p_tmp = fabs(p_tmp);  // Remove sign from momentum

                        // Update momentum with random energy changes
                        for (int l = 1; l <= labs(k); l++) {
                            double DeltaE = alfa * (p_tmp - pmin) * (pmax - p_tmp);
                            randomValue = d_xorshift(&seed);
                            p_tmp = sqrt(p_tmp * p_tmp + DeltaE * (randomValue - 0.5));
                        }
                        p_tmp *= (k % 2 ? -1.0 : 1.0) * signop;  // Restore sign to momentum
                    }
                }

                // Update the particle arrays with new positions and momenta
                x[i] = x_tmp;
                p[i] = p_tmp;
            }
        }

        /* Update histograms after each round */
        #pragma omp for schedule(static)
        for (int i = 0; i < N_PART; i++) {
            int h_idx = floor((x[i] + 0.5) * (1.99999999999999 * BINS) + 2.0);
            int g_idx = floor((p[i] / 3.0e-23 + 1) * (0.999999999999994 * BINS));
            int hg_idx = (2 * BINS) * h_idx + g_idx;
            h[h_idx]++;
            g[g_idx]++;
            hg[hg_idx]++;
        }

        // Update evolution count
        evolution += steps[j];

        // Format the filename for saving the histogram data
        if (evolution < 10000000) {
            sprintf(filename, "X%07d.dat", evolution);
        } else {
            sprintf(filename, "X%1.3e.dat", (double)evolution);
            char *e = (char*) memchr(filename, 'e', 32);
            strcpy(e + 1, e + 3);  // Adjust scientific notation in filename
        }

        // Save simulation data
        if (dump == 0) {
            save_data(saveFilename, x, p, evolution, N_PART);
        }

        // Calculate and save histograms
        Et = energy_sum(p, N_PART, evolution, M);
        make_hist(h, g, hg, DxE, DpE, filename, BINS, Et);
    }

    // Final message indicating completion
    printf("Simulation complete: evolution = %d\n", evolution);

    /* Free allocated memory */
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