#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>
#include "../include/utils.h"
#include "../include/histogram_kernels.h"



using namespace std;

static double d_rand() {
    srand(time(NULL));
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

void load_parameters_from_file(char filename[], int *N_PART, int *BINS, float *DT, double *M, int *N_THREADS,
                               unsigned int *Ntandas, int steps[], char inputFilename[], char saveFilename[],
                               bool *resume, bool *dump, float *sigmaL)
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
    fscanf(inputFile, " %*[^:]: %f", DT);
    fscanf(inputFile, " %*[^:]: %le", M);
    fscanf(inputFile, " %*[^:]: %d", N_THREADS);
    fscanf(inputFile, " %*[^\n]");
    *Ntandas = 0;
    while (fscanf(inputFile, " %d", &steps[*Ntandas]) == 1) {
        (*Ntandas)++;
    }
    fscanf(inputFile, " %*[^:]: %s %s", du, inputFilename);
    // *resume = strcmp(du, "sí");
    *resume = (strcmp(du, "sí") == 0);
    cout << du << " lee " << inputFilename << "\t";
    fscanf(inputFile, " %*[^:]: %s %s", du, saveFilename);
    cout << du << " escribe " << saveFilename << "\t";
    *dump = (strcmp(du, "sí") == 0);
    fscanf(inputFile, " %*[^:]: %f", sigmaL);
    cout << "sigma(L) = " << *sigmaL << endl;
    fclose(inputFile);
}

void read_data(char filename[], float *h_x, double *h_p, unsigned int *evolution, int N_PART)
{
    FILE *readFile = fopen(filename, "r");
    if (readFile == NULL)
    {
        cout << "Error al abrir el archivo " << filename << endl;
        exit(1);
    }
    fread(evolution, sizeof(*evolution), 1, readFile); // lee bien evolution como unsigned int
    fread(h_x, sizeof(h_x[0]) * N_PART, 1, readFile);
    fread(h_p, sizeof(h_p[0]) * N_PART, 1, readFile);
    fclose(readFile);
}

float energy_sum(double *d_p, int N_PART, unsigned int evolution, double M) {
    double *d_partialSum;
    double *partialSum;
    int threadsPerBlock = 256; // Define the number of threads per block
    int blocksPerGrid = (N_PART + threadsPerBlock - 1) / threadsPerBlock; // Calculate grid size

    // Allocate host memory for partial sums
    partialSum = (double*)malloc(blocksPerGrid * sizeof(double));

    // Allocate device memory
    cudaMalloc(&d_partialSum, blocksPerGrid * sizeof(double));

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
    cudaFree(d_partialSum);

    // Output the result
    std::cout << "N° de pasos " << evolution << "\tEnergía total = " << sumEnergy / (2 * M) << std::endl;
    return (float)(sumEnergy / (2 * M));
}

void save_data(char filename[], float *h_x, double *h_p, unsigned int evolution, int N_PART) {
    ofstream saveFile(filename, ios::binary);
    if (!saveFile) {
        cerr << "Error al abrir el archivo " << filename << endl;
        exit(1);
    }
    saveFile.write(reinterpret_cast<const char *>(&evolution), sizeof(evolution));
    saveFile.write(reinterpret_cast<const char *>(h_x), sizeof(h_x[0]) * N_PART);

    int Npmod = (0 * N_PART) / (1 << 21);
    if (evolution % 1000000 == 0 && Npmod > 0) {
        double f = 0.7071; // fraccion de p+ que queda en p+'
        double *sqrtp2 = static_cast<double *>(malloc(sizeof(double) * Npmod));
        int np = 0;
        int i0 = d_rand() * N_PART;
        int i = i0;
        while ((np < Npmod) && (i < N_PART)) {
            if (fabs(h_p[i]) > (2.43 + 0.3 * np / Npmod) * 5.24684E-24) {
                sqrtp2[np] = sqrt(1.0 - f * f) * h_p[i];
                np++;
                h_p[i] *= f;
            }
            i++;
        }
        i = 0;
        while ((np < Npmod) && (i < i0)) {
            if (fabs(h_p[i]) > (2.43 + 0.3 * np / Npmod) * 5.24684E-24)
            {
                sqrtp2[np] = sqrt(1.0 - f * f) * h_p[i];
                np++;
                h_p[i] *= f;
            }
            i++;
        }
        Npmod = np;
        cout << "np=" << np << "   (2.43-2.73)sigma" << endl;
        np = 0;
        while ((np < Npmod) && (i < N_PART)) {
            int signopr = copysign(1.0, sqrtp2[np]);
            if ((signopr * h_p[i] > 0) && (fabs(h_p[i]) > 0.15 * 5.24684E-24) && (fabs(h_p[i]) < 0.9 * 5.24684E-24)) {
                h_p[i] = sqrt(h_p[i] * h_p[i] + sqrtp2[np] * sqrtp2[np] / 2.0);
                np++;
            }
            i++;
        }
        i = 0;
        while (np < Npmod) {
            int signopr = copysign(1.0, sqrtp2[np]);
            if ((signopr * h_p[i] > 0) && (fabs(h_p[i]) > 0.15 * 5.24684E-24) && (fabs(h_p[i]) < 0.9 * 5.24684E-24)) {
                h_p[i] = sqrt(h_p[i] * h_p[i] + sqrtp2[np] * sqrtp2[np] / 2.0);
                np++;
            }
            i++;
            
        }
        free(sqrtp2);
    }
    saveFile.write(reinterpret_cast<const char *>(h_p), sizeof(h_p[0]) * N_PART);
}

__global__ void chi2x_kernel(int *h, double *DxE, double *chi2x, int BINS, bool isX0000000) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local_chi2x = 0.0;

    if (isX0000000) {
        if (i >= BINS && i < 2 * BINS) {
            local_chi2x = pow(h[i] - 2 * DxE[i], 2) / (2 * DxE[i]);
        }
    } else {
        if (i >= 4 && i < 2 * BINS) {
            local_chi2x = pow(h[i] - DxE[i], 2) / DxE[i];
        }
    }

    // Atomic addition to avoid race conditions
    atomicAdd(chi2x, local_chi2x);
}

__global__ void chi2p_kernel(int *g, double *DpE, double *chi2p, int BINS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 2 * (BINS - BORDES)) {
        double local_chi2p = pow(g[i + BORDES] - DpE[i + BORDES], 2) / DpE[i + BORDES];
        atomicAdd(chi2p, local_chi2p);
    }
}

__global__ void chiIp_Pp_kernel(int *g, double *DpE, double *chiIp, double *chiPp, int BINS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (BINS - BORDES)) {
        double chiIp_local = pow(g[i + BORDES] - g[2 * BINS - 1 - BORDES - i], 2) / DpE[i + BORDES];
        double chiPp_local = pow(g[i + BORDES] + g[2 * BINS - 1 - BORDES - i] - 2.0 * DpE[i + BORDES], 2) / DpE[i + BORDES];

        atomicAdd(chiIp, chiIp_local);
        atomicAdd(chiPp, chiPp_local);
    }
}

__global__ void chiIx_Px_kernel(int *h, double *DxE, double *chiIx, double *chiPx, int BINS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4 && i <= BINS + 1) {
        double chiIx_local = pow(h[i] - h[2 * BINS + 3 - i], 2) / DxE[i];
        double chiPx_local = pow(h[i] + h[2 * BINS + 3 - i] - 2.0 * DxE[i], 2) / DxE[i];

        atomicAdd(chiIx, chiIx_local);
        atomicAdd(chiPx, chiPx_local);
    }
}


int make_hist(int *h_h, int *h_g, int *h_hg, int *d_h, int *d_g, int *d_hg, double *DxE, double *DpE, const char *filename, int BINS, float Et) {
    double *d_chi2x, *d_chi2p, *d_chiIp, *d_chiPp, *d_chiIx, *d_chiPx;

    // Allocate memory for reduction variables on GPU
    cudaMallocManaged(&d_chi2x, sizeof(double));
    cudaMallocManaged(&d_chi2p, sizeof(double));
    cudaMallocManaged(&d_chiIp, sizeof(double));
    cudaMallocManaged(&d_chiPp, sizeof(double));
    cudaMallocManaged(&d_chiIx, sizeof(double));
    cudaMallocManaged(&d_chiPx, sizeof(double));

    *d_chi2x = 0.0;
    *d_chi2p = 0.0;
    *d_chiIp = 0.0;
    *d_chiPp = 0.0;
    *d_chiIx = 0.0;
    *d_chiPx = 0.0;

    int blockSize = 256; // Can be tuned
    int numBlocksX = (2 * BINS + blockSize - 1) / blockSize;
    int numBlocksP = (2 * (BINS - BORDES) + blockSize - 1) / blockSize;

    bool isX0000000 = (strcmp(filename, "X0000000.dat") == 0);

    // Launch kernels for chi2x calculation
    chi2x_kernel<<<numBlocksX, blockSize>>>(d_h, DxE, d_chi2x, BINS, isX0000000);
    cudaDeviceSynchronize();

    // Calculate chi2xr if needed
    double chi2xr = *d_chi2x;
    if (!isX0000000) {
        *d_chi2x /= (2.0 * BINS - 4);
    }

    // Launch kernel for chi2p calculation
    chi2p_kernel<<<numBlocksP, blockSize>>>(d_g, DpE, d_chi2p, BINS);
    cudaDeviceSynchronize();

    // Launch kernel for chiIp and chiPp
    chiIp_Pp_kernel<<<numBlocksP, blockSize>>>(d_g, DpE, d_chiIp, d_chiPp, BINS);
    cudaDeviceSynchronize();

    // Launch kernel for chiIx and chiPx
    chiIx_Px_kernel<<<numBlocksX, blockSize>>>(d_h, DxE, d_chiIx, d_chiPx, BINS);
    cudaDeviceSynchronize();

    // Scale results for chi2p, chiIp, chiPp
    *d_chi2p /= (2.0 * (BINS - BORDES));
    *d_chiIp /= (2.0 * (BINS - BORDES));
    *d_chiPp /= (2.0 * (BINS - BORDES));
    *d_chiIx /= (2.0 * (BINS - 2));
    *d_chiPx /= (2.0 * (BINS - 2));

    // Copy the result back to the host
    cudaMemcpy(h_h, d_h, (2 * BINS + 4) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g, d_g, (2 * BINS) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hg, d_hg, (2 * BINS + 4) * (2 * BINS) * sizeof(int), cudaMemcpyDeviceToHost);

    // // Write results to file
    FILE *hist = fopen(filename, "w");

    fprintf(hist, "test\n");

    fprintf(hist,
            "#   x    poblacion       p      poblacion    chi2x =%9.6f  chi2xr "
            "=%9.6f  chiIx =%9.6f  chiPx =%9.6f  chi2p =%9.6f  chiIp =%9.6f  "
            "chiPp =%9.6f  Et=%12.9E\n",
            *d_chi2x, chi2xr, *d_chiIx, *d_chiPx, *d_chi2p, *d_chiIp, *d_chiPp, Et);

    fprintf(hist, "%8.5f %6d %24.12E %6d\n", -0.5015, h_h[0], -2.997e-23, h_g[0]);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", -0.5005, h_h[1], -2.997e-23, h_g[0]);

    for (int i = 0; i < BINS << 1; i++) {
        fprintf(hist, "%8.5f %6d %24.12E %6d\n", (0.5 * i / BINS - 0.4995), h_h[i + 2], (3.0e-23 * i / BINS - 2.997e-23),
                h_g[i]);
    }
    
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", 0.5005, h_h[2 * BINS + 2], 2.997e-23, h_g[2 * BINS - 1]);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", 0.5015, h_h[2 * BINS + 3], 2.997e-23, h_g[2 * BINS - 1]);
    fclose(hist);

    // Other print statements remain similar...
    // Cleanup GPU memory
    cudaFree(d_chi2x);
    cudaFree(d_chi2p);
    cudaFree(d_chiIp);
    cudaFree(d_chiPp);
    cudaFree(d_chiIx);
    cudaFree(d_chiPx);

    // Reset memory for h, g, hg
    cudaMemset(d_h, 0, (2 * BINS + 4) * sizeof(int));
    cudaMemset(d_g, 0, (2 * BINS) * sizeof(int));
    cudaMemset(d_hg, 0, (2 * BINS + 4) * (2 * BINS) * sizeof(int));

    memset(h_h, 0, (2 * BINS + 4) * sizeof(int));
    memset(h_g, 0, (2 * BINS) * sizeof(int));
    memset(h_hg, 0, (2 * BINS + 4) * (2 * BINS) * sizeof(int));

    return 0;
}