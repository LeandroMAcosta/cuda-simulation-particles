#include "../include/types.h"
#include "../include/utils.h"
#include "../include/kernels.h"
#include "../config.h"

// Function to allocate device memory for simulation data
void allocate_device_memory(SimulationData *d_data, SimulationParams *params) {
    CUDA_CHECK(cudaMalloc(&d_data->x, sizeof(double) * params->N_PART));
    CUDA_CHECK(cudaMalloc(&d_data->p, sizeof(double) * params->N_PART));
    CUDA_CHECK(cudaMalloc(&d_data->DxE, sizeof(double) * (2 * params->BINS + 4)));
    CUDA_CHECK(cudaMalloc(&d_data->DpE, sizeof(double) * (2 * params->BINS)));
    CUDA_CHECK(cudaMalloc(&d_data->h, sizeof(int) * (2 * params->BINS + 4)));
    CUDA_CHECK(cudaMalloc(&d_data->g, sizeof(int) * (2 * params->BINS)));
    CUDA_CHECK(cudaMalloc(&d_data->hg, sizeof(int) * (2 * params->BINS + 4) * (2 * params->BINS)));
    CUDA_CHECK(cudaMalloc(&d_data->rng_states, sizeof(curandState) * params->N_PART));
}

void free_device_memory(SimulationData *d_data) {
    if (d_data->x) cudaFree(d_data->x);
    if (d_data->p) cudaFree(d_data->p);
    if (d_data->DxE) cudaFree(d_data->DxE);
    if (d_data->DpE) cudaFree(d_data->DpE);
    if (d_data->h) cudaFree(d_data->h);
    if (d_data->g) cudaFree(d_data->g);
    if (d_data->hg) cudaFree(d_data->hg);
    if (d_data->rng_states) cudaFree(d_data->rng_states);
}

// Energy sum on device with reduction
double compute_energy_on_device(double *d_p, int N_PART, double M) {
    int num_blocks = GRID_SIZE(N_PART);
    double *d_partial_sums;
    double *h_partial_sums = (double*)malloc(sizeof(double) * num_blocks);
    
    CUDA_CHECK(cudaMalloc(&d_partial_sums, sizeof(double) * num_blocks));
    
    energy_sum_kernel<<<num_blocks, BLOCK_SIZE>>>(d_p, d_partial_sums, N_PART, M);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, sizeof(double) * num_blocks, cudaMemcpyDeviceToHost));
    
    double total_energy = 0.0;
    for (int i = 0; i < num_blocks; i++) {
        total_energy += h_partial_sums[i];
    }
    
    free(h_partial_sums);
    cudaFree(d_partial_sums);
    
    return total_energy;
}

int main() {
    SimulationParams params;
    SimulationData h_data = {0}; // Host data
    SimulationData d_data = {0}; // Device data
    
    // Initialize CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    
    printf("CUDA Particle Simulation - Found %d CUDA device(s)\n", device_count);
    
    // Load parameters from file
    char data_filename[] = "datos.in";
    load_parameters_from_file(data_filename, &params);
    
    printf("Simulation parameters:\n");
    printf("  N_PART: %d\n", params.N_PART);
    printf("  BINS: %d\n", params.BINS);
    printf("  DT: %e\n", params.DT);
    printf("  M: %e\n", params.M);
    printf("  sigmaL: %e\n", params.sigmaL);
    printf("  Iterations: %d\n", params.Ntandas);
    
    // Allocate memory
    allocate_simulation_data(&h_data, &params);
    allocate_device_memory(&d_data, &params);
    
    // Set up grid and block dimensions
    dim3 gridSize = GRID_SIZE(params.N_PART);
    dim3 blockSize = BLOCK_SIZE;
    
    printf("CUDA configuration: Grid=%d, Block=%d\n", gridSize.x, blockSize.x);
    
    // Initialize RNG states on device
    init_rng_states<<<gridSize, blockSize>>>(d_data.rng_states, time(NULL), params.N_PART);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initialize distributions on device
    dim3 hist_grid = GRID_SIZE(2 * params.BINS + 4);
    initialize_momentum_distribution<<<GRID_SIZE(2 * params.BINS), blockSize>>>(d_data.DpE, params.BINS, params.N_PART);
    initialize_position_distribution<<<hist_grid, blockSize>>>(d_data.DxE, params.BINS, params.N_PART);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initialize or load particle data
         if (params.resume != 0) {
        // Initialize particles from scratch
        int X0 = 1;
        
        while (X0 == 1) {
            // Clear histograms
            clear_histograms<<<hist_grid, blockSize>>>(d_data.h, d_data.g, d_data.hg, params.BINS);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Initialize particles on device
            initialize_particles<<<gridSize, blockSize>>>(d_data.x, d_data.p, d_data.rng_states, params.N_PART);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Compute initial histograms
            compute_histograms<<<gridSize, blockSize>>>(d_data.x, d_data.p, d_data.h, d_data.g, d_data.hg, 
                                                       params.N_PART, params.BINS);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Copy histograms back to host for analysis
            copy_data_to_host(&h_data, &d_data, &params);
            
            // Compute energy
            double Et = compute_energy_on_device(d_data.p, params.N_PART, params.M);
            
            // Generate initial histogram file
            X0 = make_hist_host(h_data.h, h_data.g, h_data.hg, h_data.DxE, h_data.DpE, "X0000000.dat", params.BINS, Et);
            
            if (X0 == 1) {
                printf("Falló algún chi2: X0=%d\n", X0);
            }
        }
    } else {
        // Load from file
        read_data(params.inputFilename, h_data.x, h_data.p, &params.evolution, params.N_PART);
        copy_data_to_device(&h_data, &d_data, &params);
    }
    
    // Main simulation loop
    double Et = compute_energy_on_device(d_data.p, params.N_PART, params.M);
    printf("Initial energy = %12.9E\n", Et);
    
    for (unsigned int j = 0; j < params.Ntandas; j++) {
        printf("Processing iteration %d with %d steps...\n", j + 1, params.steps[j]);
        
        // Clear histograms for this iteration
        clear_histograms<<<hist_grid, blockSize>>>(d_data.h, d_data.g, d_data.hg, params.BINS);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Particle evolution kernel
        particle_evolution_kernel<<<gridSize, blockSize>>>(d_data.x, d_data.p, d_data.rng_states, 
                                                          params, params.steps[j]);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Compute histograms after evolution
        compute_histograms<<<gridSize, blockSize>>>(d_data.x, d_data.p, d_data.h, d_data.g, d_data.hg, 
                                                   params.N_PART, params.BINS);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Update evolution counter
        params.evolution += params.steps[j];
        
        // Generate output filename
        char filename[32];
        if (params.evolution < 10000000) {
            sprintf(filename, "X%07d.dat", params.evolution);
        } else {
            sprintf(filename, "X%1.3e.dat", (double)params.evolution);
            char *e = (char*)memchr(filename, 'e', 32);
            if (e) {
                strcpy(e + 1, e + 3);
            }
        }
        
        // Copy results back to host
        copy_data_to_host(&h_data, &d_data, &params);
        
        // Save data if required
        if (params.dump == 0) {
            save_data(params.saveFilename, h_data.x, h_data.p, params.evolution, params.N_PART);
        }
        
        // Compute final energy and create histogram
        Et = compute_energy_on_device(d_data.p, params.N_PART, params.M);
        make_hist_host(h_data.h, h_data.g, h_data.hg, h_data.DxE, h_data.DpE, filename, params.BINS, Et);
        
        printf("Iteration %d completed. Evolution = %d, Energy = %12.9E\n", j + 1, params.evolution, Et);
    }
    
    printf("Simulation completed. Final evolution = %d\n", params.evolution);
    
    // Cleanup
    free_simulation_data(&h_data);
    free_device_memory(&d_data);
    
    return 0;
} 