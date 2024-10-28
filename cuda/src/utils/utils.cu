#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>

#include <type_traits>

#include "../include/constants.h"
#include "../include/utils.h"
#include "../include/histogram_kernels.h"
#include "../include/types.h"


using namespace std;

static double d_rand() {
    srand(time(NULL));
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

void load_parameters_from_file(char filename[], int *N_PART, int *BINS, RealTypeConstant *DT, RealTypeConstant *M, int *N_THREADS,
                               unsigned int *Ntandas, int steps[], char inputFilename[], char saveFilename[],
                               bool *resume, bool *dump, RealTypeConstant *sigmaL) {
    char du[4];
    FILE *inputFile = fopen(filename, "r");
    if (inputFile == NULL) {
        cout << "Error al abrir el archivo " << filename << endl;
        exit(1);
    }
    fscanf(inputFile, " %*[^\n]");
    fscanf(inputFile, " %*[^:]: %d", N_PART);
    fscanf(inputFile, " %*[^:]: %d", BINS);

    fscanf(inputFile, " %*[^:]: %lf", DT);
    fscanf(inputFile, " %*[^:]: %lf", M);
    // fscanf(inputFile, " %*[^:]: %f", DT);
    // fscanf(inputFile, " %*[^:]: %f", M);

    fscanf(inputFile, " %*[^:]: %d", N_THREADS);
    fscanf(inputFile, " %*[^\n]");
    *Ntandas = 0;
    while (fscanf(inputFile, " %d", &steps[*Ntandas]) == 1) {
        (*Ntandas)++;
    }
    fscanf(inputFile, " %*[^:]: %s %s", du, inputFilename);
    *resume = (strcmp(du, "sí") == 0);
    cout << du << " lee " << inputFilename << "\t";
    fscanf(inputFile, " %*[^:]: %s %s", du, saveFilename);
    cout << du << " escribe " << saveFilename << "\t";
    *dump = (strcmp(du, "sí") == 0);

    fscanf(inputFile, " %*[^:]: %lf", sigmaL);
    // fscanf(inputFile, " %*[^:]: %f", sigmaL);

    cout << "sigma(L) = " << *sigmaL << endl;
    fclose(inputFile);
}

void read_data(char filename[], RealTypeX *h_x, RealTypeP *h_p, unsigned int *evolution, int N_PART) {
    FILE *readFile = fopen(filename, "r");
    if (readFile == NULL)
    {
        cout << "Error al abrir el archivo " << filename << endl;
        exit(1);
    }
    fread(evolution, sizeof(*evolution), 1, readFile); // lee bien evolution como unsigned int

    if constexpr (std::is_same<RealTypeX, double>::value) {
        fread(h_x, sizeof(h_x[0]) * N_PART, 1, readFile);
    } else if (std::is_same<RealTypeX, float>::value) {
        double *h_x_float = static_cast<double *>(malloc(sizeof(h_x_float[0]) * N_PART));
        fread(h_x_float, sizeof(h_x_float[0]) * N_PART, 1, readFile);
        for (int i = 0; i < N_PART; i++) {
            h_x[i] = static_cast<RealTypeX>(h_x_float[i]);
        }
        free(h_x_float);
    }

    if constexpr (std::is_same<RealTypeP, double>::value) {
        fread(h_p, sizeof(h_p[0]) * N_PART, 1, readFile);
    } else if (std::is_same<RealTypeP, float>::value) {
        double *h_p_float = static_cast<double *>(malloc(sizeof(h_p_float[0]) * N_PART));
        fread(h_p_float, sizeof(h_p_float[0]) * N_PART, 1, readFile);
        for (int i = 0; i < N_PART; i++) {
            h_p[i] = static_cast<RealTypeP>(h_p_float[i]);
        }
        free(h_p_float);
    }

    fclose(readFile);
}

RealTypePartialSum energy_sum(RealTypeP *d_p, int N_PART, unsigned int evolution, RealTypeConstant M) {
    RealTypePartialSum *d_partialSum;
    RealTypePartialSum *partialSum;
    int threadsPerBlock = 256; // Define the number of threads per block
    int blocksPerGrid = (N_PART + threadsPerBlock - 1) / threadsPerBlock; // Calculate grid size

    // Allocate host memory for partial sums
    partialSum = (RealTypePartialSum*)malloc(blocksPerGrid * sizeof(partialSum[0]));

    // Allocate device memory
    cudaMalloc(&d_partialSum, blocksPerGrid * sizeof(d_partialSum[0]));

    // Launch kernel
    size_t shared_memory_size = threadsPerBlock * sizeof(RealTypePartialSum);
    energy_sum_kernel<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>>(d_p, d_partialSum, N_PART);

    // Copy partial sums back to host
    cudaMemcpy(partialSum, d_partialSum, blocksPerGrid * sizeof(partialSum[0]), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    RealTypePartialSum sumEnergy = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        sumEnergy += partialSum[i];
    }

    // Clean up memory
    free(partialSum);
    cudaFree(d_partialSum);

    // Output the result
    cout << "Sum energy: " << sumEnergy << endl;
    std::cout << "N° de pasos " << evolution << "\tEnergía total = " << scientific << sumEnergy / (2 * M) << std::endl;
    return (sumEnergy / (2 * M));
}

void save_data(char filename[], RealTypeX *h_x, RealTypeP *h_p, unsigned int evolution, int N_PART) {
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
        double *sqrtp2 = static_cast<double *>(malloc(sizeof(sqrtp2[0]) * Npmod));
        int np = 0;
        int i0 = d_rand() * N_PART;
        int i = i0;
        while ((np < Npmod) && (i < N_PART)) {
            if (fabs(h_p[i]) > (2.43 + 0.3 * np / Npmod) * SIGMA_VELOCITY) {
                sqrtp2[np] = sqrt(1.0 - f * f) * h_p[i];
                np++;
                h_p[i] *= f;
            }
            i++;
        }
        i = 0;
        while ((np < Npmod) && (i < i0)) {
            if (fabs(h_p[i]) > (2.43 + 0.3 * np / Npmod) * SIGMA_VELOCITY)
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
            if ((signopr * h_p[i] > 0) && (fabs(h_p[i]) > 0.15 * SIGMA_VELOCITY) && (fabs(h_p[i]) < 0.9 * SIGMA_VELOCITY)) {
                h_p[i] = sqrt(h_p[i] * h_p[i] + sqrtp2[np] * sqrtp2[np] / 2.0);
                np++;
            }
            i++;
        }
        i = 0;
        while (np < Npmod) {
            int signopr = copysign(1.0, sqrtp2[np]);
            if ((signopr * h_p[i] > 0) && (fabs(h_p[i]) > 0.15 * SIGMA_VELOCITY) && (fabs(h_p[i]) < 0.9 * SIGMA_VELOCITY)) {
                h_p[i] = sqrt(h_p[i] * h_p[i] + sqrtp2[np] * sqrtp2[np] / 2.0);
                np++;
            }
            i++;
            
        }
        free(sqrtp2);
    }
    saveFile.write(reinterpret_cast<const char *>(h_p), sizeof(h_p[0]) * N_PART);
}

__global__ void chi2x_kernel(int *d_h, RealTypeX *d_DxE, double *chi2x, int BINS, bool isX0000000) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local_chi2x = 0.0;

    if (isX0000000) {
        if (i >= BINS && i < 2 * BINS) {
            local_chi2x = pow(d_h[i] - 2 * d_DxE[i], 2) / (2 * d_DxE[i]);
        }
    } else {
        if (i >= 4 && i < 2 * BINS) {
            local_chi2x = pow(d_h[i] - d_DxE[i], 2) / d_DxE[i];
        }
    }

    // Atomic addition to avoid race conditions
    atomicAdd(chi2x, local_chi2x);
}

__global__ void chi2p_kernel(int *g, RealTypeP *d_DpE, double *chi2p, int BINS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 2 * (BINS - BORDES)) {
        double local_chi2p = pow(g[i + BORDES] - d_DpE[i + BORDES], 2) / d_DpE[i + BORDES];
        atomicAdd(chi2p, local_chi2p);
    }
}

__global__ void chiIp_Pp_kernel(int *g, RealTypeP *d_DpE, double *chiIp, double *chiPp, int BINS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (BINS - BORDES)) {
        double chiIp_local = pow(g[i + BORDES] - g[2 * BINS - 1 - BORDES - i], 2) / d_DpE[i + BORDES];
        double chiPp_local = pow(g[i + BORDES] + g[2 * BINS - 1 - BORDES - i] - 2.0 * d_DpE[i + BORDES], 2) / d_DpE[i + BORDES];

        atomicAdd(chiIp, chiIp_local);
        atomicAdd(chiPp, chiPp_local);
    }
}

__global__ void chiIx_Px_kernel(int *h, RealTypeX *d_DxE, double *chiIx, double *chiPx, int BINS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4 && i <= BINS + 1) {
        double chiIx_local = pow(h[i] - h[2 * BINS + 3 - i], 2) / d_DxE[i];
        double chiPx_local = pow(h[i] + h[2 * BINS + 3 - i] - 2.0 * d_DxE[i], 2) / d_DxE[i];

        atomicAdd(chiIx, chiIx_local);
        atomicAdd(chiPx, chiPx_local);
    }
}


int make_hist(int *h_h, int *h_g, int *h_hg, int *d_h, int *d_g, int *d_hg, RealTypeX *d_DxE, RealTypeP *d_DpE, const char *filename, int BINS, double Et) {
    double *d_chi2x, *d_chi2p, *d_chiIp, *d_chiPp, *d_chiIx, *d_chiPx;

    // Allocate memory for reduction variables on GPU
    cudaMallocManaged(&d_chi2x, sizeof(d_chi2x[0]));
    cudaMallocManaged(&d_chi2p, sizeof(d_chi2p[0]));
    cudaMallocManaged(&d_chiIp, sizeof(d_chiIp[0]));
    cudaMallocManaged(&d_chiPp, sizeof(d_chiPp[0]));
    cudaMallocManaged(&d_chiIx, sizeof(d_chiIx[0]));
    cudaMallocManaged(&d_chiPx, sizeof(d_chiPx[0]));

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
    chi2x_kernel<<<numBlocksX, blockSize>>>(d_h, d_DxE, d_chi2x, BINS, isX0000000);
    cudaDeviceSynchronize();

    // Calculate chi2xr if needed
    double chi2xr = *d_chi2x;
    if (!isX0000000) {
        *d_chi2x /= (2.0 * BINS - 4);
    }

    // Launch kernel for chi2p calculation
    chi2p_kernel<<<numBlocksP, blockSize>>>(d_g, d_DpE, d_chi2p, BINS);
    cudaDeviceSynchronize();

    // Launch kernel for chiIp and chiPp
    chiIp_Pp_kernel<<<numBlocksP, blockSize>>>(d_g, d_DpE, d_chiIp, d_chiPp, BINS);
    cudaDeviceSynchronize();

    // Launch kernel for chiIx and chiPx
    chiIx_Px_kernel<<<numBlocksX, blockSize>>>(d_h, d_DxE, d_chiIx, d_chiPx, BINS);
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