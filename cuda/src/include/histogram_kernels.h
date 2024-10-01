#ifndef HISTOGRAM_KERNELS_H
#define HISTOGRAM_KERNELS_H

__global__ void calculateDpE(double *DpE, int N_PART, int BINS);

__global__ void calculateDxE(double *DxE, int N_PART, int BINS);

#endif
