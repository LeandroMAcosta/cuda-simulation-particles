#include <cmath>
#include "../include/constants.h"
#include "../include/histogram_kernels.h"

__global__ void calculateDpE(double *DpE, int N_PART, int BINS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 2 * BINS) {
        double numerator = 6.0E-26 * N_PART;
        double denominator = 5.24684E-24 * sqrt(2.0 * M_PI);
        double exponent = -pow(3.0e-23 * (1.0 * i / BINS - 0.999) / 5.24684E-24, 2) / 2;
        DpE[i] = (numerator / denominator) * exp(exponent);
    }
}
