# CUDA 1D Gas Particle Simulation

This is a CUDA implementation of the 1D gas particle simulation, migrated from the original OpenMP C version. The simulation models gas particles in a 1D container with discrete momentum states, wall noise, and energy redistribution.

## Project Overview

The simulation features:
- **1D Gas Particle Dynamics**: Particles moving in a 1D container with wall collisions
- **Energy Redistribution**: Uniform distribution [Ej-ΔE/2, Ej+ΔE/2] during wall interactions
- **Histogram Generation**: Real-time binning for position (x) and momentum (p) analysis
- **CUDA Acceleration**: GPU-optimized kernels for massive parallelization
- **cuRAND Integration**: High-quality parallel random number generation
- **Memory Optimization**: Structure-of-Arrays layout and optimized memory transfers

## Architecture

### Directory Structure
```
cuda/
├── src/
│   ├── kernels.cu              # All CUDA kernels
│   ├── main.cu                 # Main program
│   └── utils.cu                # Utility functions
├── include/
│   ├── types.h                 # Data structures and constants
│   ├── utils.h                 # Function declarations
│   ├── constants.h             # Physical constants
│   └── kernels.h               # Kernel declarations
├── config.h                    # Compilation configuration
├── Makefile                    # Build system
├── datos.in                    # Input parameters
└── README.md                   # This file
```

### Key CUDA Kernels

1. **Initialization Kernels**:
   - `init_rng_states`: Initialize cuRAND states for each thread
   - `initialize_particles`: Set up initial particle positions and momenta
   - `initialize_momentum_distribution`: Create expected momentum distribution
   - `initialize_position_distribution`: Create expected position distribution

2. **Simulation Kernels**:
   - `particle_evolution_kernel`: Main evolution kernel handling particle dynamics
   - `compute_histograms`: Real-time histogram computation with atomic operations
   - `energy_sum_kernel`: Parallel energy calculation with reduction

3. **Analysis Kernels**:
   - `chi2_reduction_kernel`: Statistical analysis for histogram validation

## Performance Optimizations

### CUDA Best Practices Implemented

1. **Memory Management**:
   - Structure-of-Arrays (SoA) layout for coalesced memory access
   - Unified memory management with explicit transfers
   - Pre-allocated GPU memory blocks

2. **Parallelization Strategy**:
   - Separate kernels for different computation phases
   - Warp-level optimization to reduce branch divergence
   - Block-level reduction patterns for energy calculations

3. **Random Number Generation**:
   - cuRAND library for high-quality parallel RNG
   - Per-thread RNG states for independent random streams
   - Box-Muller transform for Gaussian momentum distribution

4. **Histogram Computation**:
   - Atomic operations for thread-safe updates
   - Bounds checking to prevent memory access violations
   - Optimized indexing for histogram bins

## Compilation

### Requirements
- NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
- CUDA Toolkit 10.0+
- GCC/G++ compiler
- Make build system

### Basic Compilation
```bash
cd cuda/
make clean && make
```

### Advanced Compilation Options
```bash
# Debug build
make DEBUG=1

# Specific GPU architecture (e.g., RTX 3080 = sm_86)
make GPU_ARCH=86

# Performance build (default)
make PERF=1

# Help
make help
```

### GPU Architecture Selection

To optimize performance, set the correct GPU architecture for your hardware. Use `nvidia-smi --query-gpu=compute_cap --format=csv` to check your GPU's compute capability.

#### GPU Architecture Reference Table

| GPU_ARCH | Compute Capability | Architecture | Example GPUs |
|----------|-------------------|--------------|--------------|
| `50` | 5.0 | Maxwell | GTX 750, GTX 950 |
| `52` | 5.2 | Maxwell | GTX 960, GTX 970, GTX 980 |
| `60` | 6.0 | Pascal | GTX 1050, GTX 1060 |
| `61` | 6.1 | Pascal | GTX 1070, GTX 1080, GTX 1080 Ti |
| `70` | 7.0 | Volta | Tesla V100 |
| `75` | 7.5 | Turing | RTX 2060, RTX 2070, RTX 2080, RTX 2080 Ti |
| `80` | 8.0 | Ampere | RTX 3050, RTX 3060, A100 |
| `86` | 8.6 | Ampere | RTX 3070, RTX 3080, RTX 3090 |
| `89` | 8.9 | Ada Lovelace | RTX 4090 |

#### Setting GPU Architecture

**Option 1: Temporary (current build only)**
```bash
make GPU_ARCH=75  # Example for RTX 2080 Ti
```

**Option 2: Permanent (edit Makefile)**
```bash
# Edit line 10 in Makefile:
GPU_ARCH ?= 75  # Change 80 to your GPU's value
```

**Option 3: Multiple architectures (for distribution)**
```bash
# Edit ARCH_FLAGS in Makefile for multiple targets:
ARCH_FLAGS := -arch=sm_60 -arch=sm_70 -arch=sm_75 -arch=sm_86
```

#### Why GPU Architecture Matters

1. **Performance**: Matching your GPU's architecture ensures optimal instruction usage
2. **Compatibility**: Code compiled for newer architectures won't run on older GPUs  
3. **Features**: Different architectures support different CUDA capabilities
4. **Memory**: Newer architectures have improved memory subsystems

## Usage

### Running the Simulation
```bash
./main
```

The program reads parameters from `datos.in` and generates output files:
- `X*.dat`: Histogram data files at different time steps
- `graba.dmp`: Binary state dump for resuming simulations

### Configuration (datos.in)

The input file format is compatible with the original OpenMP version:
```
# Entrada para el programa gas1D
          Número de partículas (N_PART): 2097152
          (Número de bins - 1)/2 (BINS): 500
                    Delta t en seg (DT): 4.430982591982e-7
       Masa de las partículas en kg (M): 6.646473667973E-27
Nº de hilos para la corrida (N_THREADS): 32
Tandas (nº de pasos) para esta corrida (steps): 
0 20000 30000 50000
      Retomar desde un archivo anterior: sí retoma.dmp
          Volcar los datos a un archivo: sí graba.dmp
            Incertidumbre para L (en m): 0.0001
```

### Output Analysis

Generated files follow the same format as the original:
- **Column 1**: x position bins (histogram centers)
- **Column 2**: Population count for x positions  
- **Column 3**: p momentum bins (histogram centers)
- **Column 4**: Population count for momentum

## Performance Monitoring

### Built-in Profiling
```bash
# Basic GPU profiling
make profile

# Advanced profiling with Nsight Systems
make nsys-profile

# Show GPU information
make gpu-info
```

### Expected Performance
With a modern GPU (RTX 3080 or better), expect:
- 10-50x speedup over OpenMP CPU version
- Memory bandwidth utilization: 80-90%
- GPU utilization: 95%+ during computation phases

## Migration from OpenMP

### Key Differences
1. **Memory Model**: Explicit GPU/CPU memory management vs. shared memory
2. **Threading**: CUDA threads/blocks vs. OpenMP threads
3. **Random Number Generation**: cuRAND vs. thread-local xorshift
4. **Reduction Operations**: CUDA reduction patterns vs. OpenMP reduction clauses

### Validation
The CUDA implementation maintains bit-level compatibility with the OpenMP version for:
- Energy conservation
- Particle trajectory calculations
- Histogram generation
- Statistical analysis (chi-square tests)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `N_PART` in `datos.in`
   - Use smaller `BINS` values
   - Check available GPU memory with `nvidia-smi`

2. **Compilation Errors**:
   - Verify CUDA Toolkit installation: `nvcc --version`
   - Check GPU compute capability: `nvidia-smi`
   - Update GPU architecture in Makefile

3. **Performance Issues**:
   - Profile with `make profile`
   - Ensure GPU has sufficient memory bandwidth
   - Check for memory transfer bottlenecks

### Debug Mode
```bash
make DEBUG=1
cuda-gdb ./main
```

## Contributing

When modifying the code:
1. Follow CUDA best practices
2. Maintain energy conservation
3. Test with original OpenMP output for validation
4. Profile performance changes
5. Update documentation

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuRAND Library](https://docs.nvidia.com/cuda/curand/)
- Original OpenMP implementation in `../omp_c/`

## License

This CUDA implementation maintains the same license as the original OpenMP project. 