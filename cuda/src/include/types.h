#ifndef TYPES_H
#define TYPES_H

// Define your precision types
// Intended to use RealTypeConstant for DT, M, Et, sigmaL 
using RealTypeConstant = float;

// Intended to use RealTypeConstant for d_x, h_x, d_DxE 
using RealTypeX = float;

// Use RealType2 for d_p, DpE, h_p, 
using RealTypeP = double;

// Use RealTypePartialSum for partial_sum, d_partial_sum
// and return type for energy_sum
using RealTypePartialSum = double;

#endif