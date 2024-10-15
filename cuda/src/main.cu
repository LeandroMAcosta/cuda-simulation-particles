#include <cmath>
#include <cstring>
#include <cstdio>
// #include <omp.h_h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "./include/utils.h"
#include "./include/histogram_kernels.h"

using namespace std;

int main()
{
    int N_THREADS = 0, N_PART = 0, BINS = 0;
    bool resume, dump;

    unsigned int Ntandas = 0u;
    char inputFilename[255], saveFilename[255];
    
    float M;
    float DT; 
    float sigmaL = 0.0f;

    int X0 = 1;
    char filename[32];

    unsigned int evolution = 0u;
    // float alfa = 1.0e-4f;
    // float pmin = 2.0E-026f, pmax = 3.0E-023f;

    int steps[500];
    char data_filename[] = "datos.in";
    load_parameters_from_file(data_filename, &N_PART, &BINS, &DT, &M, &N_THREADS, &Ntandas, steps, inputFilename,
                              saveFilename, &resume, &dump, &sigmaL);

    printf("Parameters loaded from file:\n");
    printf("N_PART=%d\n", N_PART);
    printf("BINS=%d\n", BINS);
    printf("DT=%.60f\n", DT);
    printf("M=%.60f\n", M);
    printf("N_THREADS=%d\n", N_THREADS);
    printf("Ntandas=%d\n", Ntandas);
    printf("inputFilename=%s\n", inputFilename);
    printf("saveFilename=%s\n", saveFilename);
    printf("resume=%d\n", resume);
    printf("dump=%d\n", dump);
    printf("sigmaL=%f\n", sigmaL);

    // Unified Memory Allocation for arrays using cudaMallocManaged
    double *h_p;
    float *h_x;
    h_x = (float *)malloc(sizeof(float) * N_PART);
    h_p = (double *)malloc(sizeof(double) * N_PART);

    double *d_p, *d_DpE;
    float *d_x, *d_DxE;
    cudaMalloc(&d_x, sizeof(float) * N_PART);
    cudaMalloc(&d_p, sizeof(double) * N_PART);
    cudaMalloc(&d_DxE, sizeof(float) * (2 * BINS + 4));
    cudaMalloc(&d_DpE, sizeof(double) * (2 * BINS));

    // Launch CUDA kernel for parallel d_DpE computation
    int threadsPerBlock = 512;

    int blocksPerGridForDpE = (2 * BINS + threadsPerBlock - 1) / threadsPerBlock;
    init_DpE_kernel<<<blocksPerGridForDpE, threadsPerBlock>>>(d_DpE, N_PART, BINS);

    int blocksPerGridForDxE = (2 * BINS + threadsPerBlock - 1) / threadsPerBlock;
    init_DxE_kernel<<<blocksPerGridForDxE, threadsPerBlock>>>(d_DxE, N_PART, BINS);
  
    // Host arrays (used in CPU)
    int *h_h, *h_g, *h_hg;

    // Initialize host arrays with zeros
    h_h = (int *)calloc(2 * BINS + 4, sizeof(int));
    h_g = (int *)calloc(2 * BINS, sizeof(int));
    h_hg = (int *)calloc((2 * BINS + 4) * (2 * BINS), sizeof(int));

    // Device arrays (used in GPU)
    int *d_h, *d_g, *d_hg;
    cudaMalloc(&d_h, sizeof(int) * (2 * BINS + 4));
    cudaMalloc(&d_g, sizeof(int) * (2 * BINS));
    cudaMalloc(&d_hg, sizeof(int) * (2 * BINS + 4) * (2 * BINS));

    cudaMemset(d_h, 0, (2 * BINS + 4) * sizeof(int));
    cudaMemset(d_g, 0, (2 * BINS) * sizeof(int));
    cudaMemset(d_hg, 0, (2 * BINS + 4) * (2 * BINS) * sizeof(int));

    // Check for resume condition
    if (!resume) {
        while (X0 == 1) {
            uint32_t base_seed_1 = static_cast<uint32_t>(time(NULL));
            uint32_t base_seed_2 = static_cast<uint32_t>(time(NULL) + 1);

            int numBlocksInitX = (N_PART + threadsPerBlock - 1) / threadsPerBlock;
            init_x_kernel<<<numBlocksInitX, threadsPerBlock>>>(d_x, base_seed_1, N_PART);

            int numBlocksInitP = ((N_PART >> 1) + threadsPerBlock - 1) / threadsPerBlock;
            init_p_kernel<<<numBlocksInitP, threadsPerBlock>>>(d_p, base_seed_2, N_PART);

            // The kernel  update_histograms_kernel uses d_x and p arrays to update h_h, h_g, h_hg arrays, so we need to synchronize.
            cudaDeviceSynchronize();

            int numBlocksUpdateHist = (N_PART + threadsPerBlock - 1) / threadsPerBlock;
            update_histograms_kernel<<<numBlocksUpdateHist, threadsPerBlock>>>(d_x, d_p, d_h, d_g, d_hg, N_PART, BINS);
            cudaDeviceSynchronize();

            float Et = energy_sum(d_p, N_PART, evolution, M);
            X0 = make_hist(h_h, h_g, h_hg, d_h, d_g, d_hg, d_DxE, d_DpE, "X0000000.dat", BINS, Et);
            if (X0 == 1) {
                cout << "Falló algún chi2: X0=" << X0 << endl;
            }
        }
    } else {
        read_data(inputFilename, h_x, h_p, &evolution, N_PART);
        cudaMemcpy(d_x, h_x, sizeof(float) * N_PART, cudaMemcpyHostToDevice);
        cudaMemcpy(d_p, h_p, sizeof(double) * N_PART, cudaMemcpyHostToDevice);
    }

    float Et = energy_sum(d_p, N_PART, evolution, M);
    cout << "PMIN=" << scientific << PMIN << " ALFA=" << ALFA << " Et=" << Et << endl;

    // Main loop to iterate through Ntandas
    for (unsigned int j = 0; j < Ntandas; j++) {
        // Kernel launch parameters
        int numBlocks = (N_PART + threadsPerBlock - 1) / threadsPerBlock;

        simulate_particle_motion<<<numBlocks, threadsPerBlock>>>(steps[j], d_x, d_p, N_PART, DT, M, sigmaL);
        cudaDeviceSynchronize();

        int numBlocksUpdateHist = (N_PART + threadsPerBlock - 1) / threadsPerBlock;
        update_histograms_kernel<<<numBlocksUpdateHist, threadsPerBlock>>>(d_x, d_p, d_h, d_g, d_hg, N_PART, BINS);

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

        if (dump) {
            cudaMemcpy(h_x, d_x, sizeof(float) * N_PART, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_p, d_p, sizeof(double) * N_PART, cudaMemcpyDeviceToHost);
            save_data(saveFilename, h_x, d_p, evolution, N_PART);
        }

        Et = energy_sum(d_p, N_PART, evolution, M);
        make_hist(h_h, h_g, h_hg, d_h, d_g, d_hg, d_DxE, d_DpE, filename, BINS, Et);
    }

    cout << "Completo evolution = " << evolution << endl;

    // Free memory
    cudaFree(d_x);
    cudaFree(d_p);
    cudaFree(d_DxE);
    cudaFree(d_DpE);
    cudaFree(d_h);
    cudaFree(d_g);
    cudaFree(d_hg);
    
    // Check for any device errors (after synchronization)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    return 0;
}
