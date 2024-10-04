#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <omp.h>
#include <cuda_runtime.h>
#include "../include/utils.h"
#include "../include/histogram_kernels.h"



using namespace std;

static double d_rand() {
    srand(time(NULL));
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

void load_parameters_from_file(char filename[], int *N_PART, int *BINS, double *DT, double *M, int *N_THREADS,
                               unsigned int *Ntandas, int steps[], char inputFilename[], char saveFilename[],
                               int *resume, int *dump, double *sigmaL)
{
    char du[4];
    FILE *inputFile = fopen(filename, "r");
    if (inputFile == NULL) {
        cout << "Error al abrir el archivo " << filename << endl;
        exit(1);
    }
    fscanf(inputFile, " %*[^\n]");
    fscanf(inputFile, " %*[^:]: %d", N_PART);
    fscanf(inputFile, " %*[^:]: %d", BINS);
    fscanf(inputFile, " %*[^:]: %le", DT);
    fscanf(inputFile, " %*[^:]: %le", M);
    fscanf(inputFile, " %*[^:]: %d", N_THREADS);
    fscanf(inputFile, " %*[^\n]");
    *Ntandas = 0;
    while (fscanf(inputFile, " %d", &steps[*Ntandas]) == 1) {
        (*Ntandas)++;
    }
    fscanf(inputFile, " %*[^:]: %s %s", du, inputFilename);
    *resume = strcmp(du, "sí");
    cout << du << " lee " << inputFilename << "\t";
    fscanf(inputFile, " %*[^:]: %s %s", du, saveFilename);
    cout << du << " escribe " << saveFilename << "\t";
    *dump = strcmp(du, "sí");
    fscanf(inputFile, " %*[^:]: %le", sigmaL);
    cout << "sigma(L) = " << *sigmaL << endl;
    fclose(inputFile);
}

void read_data(char filename[], double *x, double *p, unsigned int *evolution, int N_PART)
{
    FILE *readFile = fopen(filename, "r");
    if (readFile == NULL)
    {
        cout << "Error al abrir el archivo " << filename << endl;
        exit(1);
    }
    fread(evolution, sizeof(*evolution), 1, readFile); // lee bien evolution como unsigned int
    fread(x, sizeof(x[0]) * N_PART, 1, readFile);
    fread(p, sizeof(p[0]) * N_PART, 1, readFile);
    fclose(readFile);
}

// double energy_sum(double *p, int N_PART, unsigned int evolution, double M) {
//     double sumEnergy = 0;
//     #pragma omp parallel for reduction(+ : sumEnergy) schedule(static)
//     for (int i = 0; i < N_PART; i++) {
//         sumEnergy += p[i] * p[i];
//     }
//     cout << "N° de pasos " << evolution << "\tEnergía total = " << sumEnergy / (2 * M) << endl;
//     return sumEnergy / (2 * M);
// }

double energy_sum(double *p, int N_PART, unsigned int evolution, double M) {
    double *d_p, *d_partialSum;
    double *partialSum;
    int threadsPerBlock = 256; // Define the number of threads per block
    int blocksPerGrid = (N_PART + threadsPerBlock - 1) / threadsPerBlock; // Calculate grid size

    // Allocate host memory for partial sums
    partialSum = (double*)malloc(blocksPerGrid * sizeof(double));

    // Allocate device memory
    cudaMalloc(&d_p, N_PART * sizeof(double));
    cudaMalloc(&d_partialSum, blocksPerGrid * sizeof(double));

    // Copy input array to device
    cudaMemcpy(d_p, p, N_PART * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    size_t shared_memory_size = threadsPerBlock * sizeof(double);
    energy_sum_kernel<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>>(d_p, d_partialSum, N_PART);

    // Copy partial sums back to host
    cudaMemcpy(partialSum, d_partialSum, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    double sumEnergy = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        sumEnergy += partialSum[i];
    }

    // Clean up memory
    free(partialSum);
    cudaFree(d_p);
    cudaFree(d_partialSum);

    // Output the result
    std::cout << "N° de pasos " << evolution << "\tEnergía total = " << sumEnergy / (2 * M) << std::endl;
    return sumEnergy / (2 * M);
}

void save_data(char filename[], double *x, double *p, unsigned int evolution, int N_PART) {
    ofstream saveFile(filename, ios::binary);
    if (!saveFile) {
        cerr << "Error al abrir el archivo " << filename << endl;
        exit(1);
    }
    saveFile.write(reinterpret_cast<const char *>(&evolution), sizeof(evolution));
    saveFile.write(reinterpret_cast<const char *>(x), sizeof(x[0]) * N_PART);

    int Npmod = (0 * N_PART) / (1 << 21);
    if (evolution % 1000000 == 0 && Npmod > 0) {
        double f = 0.7071; // fraccion de p+ que queda en p+'
        double *sqrtp2 = static_cast<double *>(malloc(sizeof(double) * Npmod));
        int np = 0;
        int i0 = d_rand() * N_PART;
        int i = i0;
        while ((np < Npmod) && (i < N_PART)) {
            if (fabs(p[i]) > (2.43 + 0.3 * np / Npmod) * 5.24684E-24) {
                sqrtp2[np] = sqrt(1.0 - f * f) * p[i];
                np++;
                p[i] *= f;
            }
            i++;
        }
        i = 0;
        while ((np < Npmod) && (i < i0)) {
            if (fabs(p[i]) > (2.43 + 0.3 * np / Npmod) * 5.24684E-24)
            {
                sqrtp2[np] = sqrt(1.0 - f * f) * p[i];
                np++;
                p[i] *= f;
            }
            i++;
        }
        Npmod = np;
        cout << "np=" << np << "   (2.43-2.73)sigma" << endl;
        np = 0;
        while ((np < Npmod) && (i < N_PART)) {
            int signopr = copysign(1.0, sqrtp2[np]);
            if ((signopr * p[i] > 0) && (fabs(p[i]) > 0.15 * 5.24684E-24) && (fabs(p[i]) < 0.9 * 5.24684E-24)) {
                p[i] = sqrt(p[i] * p[i] + sqrtp2[np] * sqrtp2[np] / 2.0);
                np++;
            }
            i++;
        }
        i = 0;
        while (np < Npmod) {
            int signopr = copysign(1.0, sqrtp2[np]);
            if ((signopr * p[i] > 0) && (fabs(p[i]) > 0.15 * 5.24684E-24) && (fabs(p[i]) < 0.9 * 5.24684E-24)) {
                p[i] = sqrt(p[i] * p[i] + sqrtp2[np] * sqrtp2[np] / 2.0);
                np++;
            }
            i++;
            
        }
        free(sqrtp2);
    }
    saveFile.write(reinterpret_cast<const char *>(p), sizeof(p[0]) * N_PART);
}

int make_hist(int *h, int *g, int *hg, double *DxE, double *DpE, const char *filename, int BINS, double Et) {
    double chi2x = 0.0, chi2xr = 0.0, chi2p = 0.0, chiIp = 0.0, chiPp = 0.0, chiIx = 0.0, chiPx = 0.0;

    if (strcmp(filename, "X0000000.dat") == 0) {
        #pragma omp parallel for reduction(+ : chi2x) schedule(static)
        for (int i = BINS ; i < 2 * BINS; i++) {
            chi2x += pow(h[i] - 2 * DxE[i], 2) / (2 * DxE[i]);
        }
        chi2x /= BINS;
    } else {
        #pragma omp parallel for reduction(+ : chi2x) schedule(static)
        for (int i = 4; i < 2 * BINS; i++) {
            chi2x += pow(h[i] - DxE[i], 2) / DxE[i];
        }
        chi2x /= (2.0 * BINS - 4);
        chi2xr = chi2x; // chi2xr = chi2x reducido
    }
    
    #pragma omp parallel for reduction(+ : chi2p) schedule(static)
    for (int i = 0; i < 2 * (BINS - BORDES); i++) {
        chi2p += pow(g[i + BORDES] - DpE[i + BORDES], 2) / DpE[i + BORDES];
    }
    
    #pragma omp parallel for reduction(+ : chiIp, chiPp) schedule(static)
    for (int i = 0; i < (BINS - BORDES); i++) {
        chiIp += pow(g[i + BORDES] - g[2 * BINS - 1 - BORDES - i], 2) / DpE[i + BORDES];
        chiPp += pow(g[i + BORDES] + g[2 * BINS - 1 - BORDES - i] - 2.0 * DpE[i + BORDES], 2) / DpE[i + BORDES];
    }

    #pragma omp parallel for reduction(+ : chiIx, chiPx) schedule(static)
    for (int i = 4; i <= BINS + 1; i++) {
        chiIx += pow(h[i] - h[2 * BINS + 3 - i], 2) / DxE[i];
        chiPx += pow(h[i] + h[2 * BINS + 3 - i] - 2.0 * DxE[i], 2) / DxE[i];
    }

    chiIx = chiIx / (2.0 * (BINS - 2));
    chiPx = chiPx / (2.0 * (BINS - 2));
    chi2p = chi2p / (2.0 * (BINS - BORDES));
    chiIp = chiIp / (2.0 * (BINS - BORDES));
    chiPp = chiPp / (2.0 * (BINS - BORDES));

    FILE *hist = fopen(filename, "w");
    fprintf(hist,
            "#   x    poblacion       p      poblacion    chi2x =%9.6f  chi2xr "
            "=%9.6f  chiIx =%9.6f  chiPx =%9.6f  chi2p =%9.6f  chiIp =%9.6f  "
            "chiPp =%9.6f  Et=%12.9E\n",
            chi2x, chi2xr, chiIx, chiPx, chi2p, chiIp, chiPp, Et);
    cout << "#   x    poblacion       p      poblacion    chi2x =" << chi2x << "  chi2xr =" << chi2xr << "  chiIx ="
         << chiIx << "  chiPx =" << chiPx << "  chi2p =" << chi2p << "  chiIp =" << chiIp << "  chiPp =" << chiPp
         << "  Et=" << Et << endl;
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", -0.5015, h[0], -2.997e-23, g[0]);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", -0.5005, h[1], -2.997e-23, g[0]);
    for (int i = 0; i < BINS << 1; i++) {
        fprintf(hist, "%8.5f %6d %24.12E %6d\n", (0.5 * i / BINS - 0.4995), h[i + 2], (3.0e-23 * i / BINS - 2.997e-23),
                g[i]);
    }
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", 0.5005, h[2 * BINS + 2], 2.997e-23, g[2 * BINS - 1]);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", 0.5015, h[2 * BINS + 3], 2.997e-23, g[2 * BINS - 1]);

    fclose(hist);

    memset(h, 0, (2 * BINS + 4) * sizeof(int));
    memset(g, 0, (2 * BINS) * sizeof(int));
    memset(hg, 0, (2 * BINS + 4) * (2 * BINS) * sizeof(int));

    return 0; // avisa que se cumplió la condición sobre los chi2
}