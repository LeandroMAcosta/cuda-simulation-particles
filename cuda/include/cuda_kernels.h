// cuda_functions.h
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

void initialize_histograms(int BINS, int* h, int* g, int* hg, double* DxE, double* DpE, int N_PART);

#endif  // CUDA_KERNELS_H
