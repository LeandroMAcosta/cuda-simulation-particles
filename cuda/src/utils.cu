#include "../include/utils.h"
#include "../include/types.h"
#include "../config.h"

// Host utility function implementations

void load_parameters_from_file(char filename[], SimulationParams *params) {
    char du[4];
    FILE *inputFile = fopen(filename, "r");
    if (inputFile == NULL) {
        printf("Error al abrir el archivo %s\n", filename);
        exit(1);
    }
    
    fscanf(inputFile, " %*[^\n]");
    fscanf(inputFile, " %*[^:]: %d", &params->N_PART);
    fscanf(inputFile, " %*[^:]: %d", &params->BINS);
    fscanf(inputFile, " %*[^:]: %le", &params->DT);
    fscanf(inputFile, " %*[^:]: %le", &params->M);
    fscanf(inputFile, " %*[^:]: %d", &params->N_THREADS);
    fscanf(inputFile, " %*[^\n]");
    
    params->Ntandas = 0;
    while (fscanf(inputFile, " %d", &params->steps[params->Ntandas]) == 1) {
        (params->Ntandas)++;
    }
    
    fscanf(inputFile, " %*[^:]: %s %s", du, params->inputFilename);
    params->resume = strcmp(du, "sí");
    printf("%s lee %s\t", du, params->inputFilename);
    
    fscanf(inputFile, " %*[^:]: %s %s", du, params->saveFilename);
    printf("%s escribe %s\t", du, params->saveFilename);
    params->dump = strcmp(du, "sí");
    
    fscanf(inputFile, " %*[^:]: %le", &params->sigmaL);
    printf("sigma(L) = %le\n", params->sigmaL);
    
    params->evolution = 0;
    fclose(inputFile);
}

void read_data(char filename[], float *x, float *p, unsigned int *evolution, int N_PART) {
    FILE *readFile = fopen(filename, "r");
    if (readFile == NULL) {
        printf("Error al abrir el archivo %s\n", filename);
        exit(1);
    }
    fread(evolution, sizeof(*evolution), 1, readFile);
    fread(x, sizeof(x[0]) * N_PART, 1, readFile);
    fread(p, sizeof(p[0]) * N_PART, 1, readFile);
    fclose(readFile);
}

double energy_sum_host(float *p, int N_PART, unsigned int evolution, double M) {
    double sumEnergy = 0;
    for (int i = 0; i < N_PART; i++) {
        // Convert to double for energy calculation accuracy
        double p_double = (double)p[i];
        sumEnergy += p_double * p_double;
    }
    double total_energy = sumEnergy / (2 * M);
    printf("N° de pasos %6d\tEnergía total = %12.9E\n", evolution, total_energy);
    return total_energy;
}

void save_data(char filename[], float *x, float *p, unsigned int evolution, int N_PART) {
    FILE *saveFile = fopen(filename, "w");
    if (saveFile == NULL) {
        printf("Error al abrir el archivo %s\n", filename);
        exit(1);
    }
    fwrite(&evolution, sizeof(evolution), 1, saveFile);
    fwrite(x, sizeof(x[0]) * N_PART, 1, saveFile);
    
    // Complex energy redistribution logic from original
    int Npmod = (0 * N_PART) / (1 << 21);
    if (evolution % 1000000 == 0 && Npmod > 0) {
        float f = 0.7071f;
        float *sqrtp2 = (float*)malloc(sizeof(float) * Npmod);
        int np = 0;
        int i0 = rand() * N_PART / RAND_MAX;
        int i = i0;
        
        while ((np < Npmod) && (i < N_PART)) {
            if (fabsf(p[i]) > (2.43f + 0.3f * np / Npmod) * 5.24684E-24f) {
                sqrtp2[np] = sqrtf(1.0f - f * f) * p[i];
                np++;
                p[i] *= f;
            }
            i++;
        }
        
        i = 0;
        while ((np < Npmod) && (i < i0)) {
            if (fabsf(p[i]) > (2.43f + 0.3f * np / Npmod) * 5.24684E-24f) {
                sqrtp2[np] = sqrtf(1.0f - f * f) * p[i];
                np++;
                p[i] *= f;
            }
            i++;
        }
        
        Npmod = np;
        printf("np=%d   (2.43-2.73)sigma\n", np);
        
        // Redistribute energy
        np = 0;
        while ((np < Npmod) && (i < N_PART)) {
            int signopr = copysignf(1.0f, sqrtp2[np]);
            if ((signopr * p[i] > 0) && (fabsf(p[i]) > 0.15f * 5.24684E-24f) && (fabsf(p[i]) < 0.9f * 5.24684E-24f)) {
                p[i] = sqrtf(p[i] * p[i] + sqrtp2[np] * sqrtp2[np] / 2.0f);
                np++;
            }
            i++;
        }
        
        // Continue redistribution loops...
        free(sqrtp2);
    }
    
    fwrite(p, sizeof(p[0]) * N_PART, 1, saveFile);
    fclose(saveFile);
}

int make_hist_host(int *h, int *g, int *hg, double *DxE, double *DpE, const char *filename, int BINS, double Et) {
    double chi2x = 0.0, chi2xr = 0.0, chi2p = 0.0, chiIp = 0.0, chiPp = 0.0, chiIx = 0.0, chiPx = 0.0;

    if (strcmp(filename, "X0000000.dat") == 0) {
        for (int i = BINS; i < 2 * BINS; i++) {
            chi2x += pow(h[i] - 2 * DxE[i], 2) / (2 * DxE[i]);
        }
        chi2x /= BINS;
    } else {
        for (int i = 4; i < 2 * BINS; i++) {
            chi2x += pow(h[i] - DxE[i], 2) / DxE[i];
        }
        chi2x /= (2.0 * BINS - 4);
        chi2xr = chi2x;
    }
    
    for (int i = 0; i < 2 * (BINS - BORDES); i++) {
        chi2p += pow(g[i + BORDES] - DpE[i + BORDES], 2) / DpE[i + BORDES];
    }
    
    for (int i = 0; i < (BINS - BORDES); i++) {
        chiIp += pow(g[i + BORDES] - g[2 * BINS - 1 - BORDES - i], 2) / DpE[i + BORDES];
        chiPp += pow(g[i + BORDES] + g[2 * BINS - 1 - BORDES - i] - 2.0 * DpE[i + BORDES], 2) / DpE[i + BORDES];
    }
    
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
    
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", -0.5015, h[0], -2.997e-23, g[0]);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", -0.5005, h[1], -2.997e-23, g[0]);
    
    for (int i = 0; i < BINS << 1; i++) {
        fprintf(hist, "%8.5f %6d %24.12E %6d\n", (0.5 * i / BINS - 0.4995), h[i + 2], 
                (3.0e-23 * i / BINS - 2.997e-23), g[i]);
    }
    
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", 0.5005, h[2 * BINS + 2], 2.997e-23, g[2 * BINS - 1]);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", 0.5015, h[2 * BINS + 3], 2.997e-23, g[2 * BINS - 1]);

    fclose(hist);

    // Clear histograms
    memset(h, 0, (2 * BINS + 4) * sizeof(int));
    memset(g, 0, (2 * BINS) * sizeof(int));
    memset(hg, 0, (2 * BINS + 4) * (2 * BINS) * sizeof(int));

    return 0;
}

// Memory management functions
void allocate_simulation_data(SimulationData *data, SimulationParams *params) {
    // Allocate host memory
    data->x = (float*)malloc(sizeof(float) * params->N_PART);
    data->p = (float*)malloc(sizeof(float) * params->N_PART);
    data->DxE = (double*)malloc(sizeof(double) * (2 * params->BINS + 4));
    data->DpE = (double*)malloc(sizeof(double) * (2 * params->BINS));
    data->h = (int*)malloc(sizeof(int) * (2 * params->BINS + 4));
    data->g = (int*)malloc(sizeof(int) * (2 * params->BINS));
    data->hg = (int*)malloc(sizeof(int) * (2 * params->BINS + 4) * (2 * params->BINS));
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&data->rng_states, sizeof(curandState) * params->N_PART));
    
    if (!data->x || !data->p || !data->DxE || !data->DpE || !data->h || !data->g || !data->hg) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
}

void free_simulation_data(SimulationData *data) {
    if (data->x) free(data->x);
    if (data->p) free(data->p);
    if (data->DxE) free(data->DxE);
    if (data->DpE) free(data->DpE);
    if (data->h) free(data->h);
    if (data->g) free(data->g);
    if (data->hg) free(data->hg);
    if (data->rng_states) cudaFree(data->rng_states);
}

void copy_data_to_device(SimulationData *h_data, SimulationData *d_data, SimulationParams *params) {
    CUDA_CHECK(cudaMemcpy(d_data->x, h_data->x, sizeof(float) * params->N_PART, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data->p, h_data->p, sizeof(float) * params->N_PART, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data->DxE, h_data->DxE, sizeof(double) * (2 * params->BINS + 4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data->DpE, h_data->DpE, sizeof(double) * (2 * params->BINS), cudaMemcpyHostToDevice));
}

void copy_data_to_host(SimulationData *h_data, SimulationData *d_data, SimulationParams *params) {
    CUDA_CHECK(cudaMemcpy(h_data->x, d_data->x, sizeof(float) * params->N_PART, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_data->p, d_data->p, sizeof(float) * params->N_PART, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_data->h, d_data->h, sizeof(int) * (2 * params->BINS + 4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_data->g, d_data->g, sizeof(int) * (2 * params->BINS), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_data->hg, d_data->hg, sizeof(int) * (2 * params->BINS + 4) * (2 * params->BINS), cudaMemcpyDeviceToHost));
} 