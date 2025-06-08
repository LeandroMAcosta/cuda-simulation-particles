# CUDA Particle Simulation Makefile

# Compiler settings
NVCC := nvcc
CFLAGS := -Wall -Wextra -O3 -std=c++14
CUDAFLAGS := -O3 -lineinfo --ptxas-options=-v -rdc=true
LIBS := -lm -lcurand

# GPU Architecture - can be overridden with make GPU_ARCH=XX
GPU_ARCH ?= 75
ARCH_FLAGS := -arch=sm_$(GPU_ARCH)

# Target executable
TARGET := main

# Source files
SRC_DIR := src
KERNELS_DIR := $(SRC_DIR)/kernels
INCLUDE_DIR := include
SOURCES := $(SRC_DIR)/main.cu $(SRC_DIR)/utils.cu \
           $(KERNELS_DIR)/init_kernels.cu $(KERNELS_DIR)/simulation_kernels.cu \
           $(KERNELS_DIR)/histogram_kernels.cu $(KERNELS_DIR)/reduction_kernels.cu \
           $(KERNELS_DIR)/utility_kernels.cu
OBJECTS := $(SOURCES:.cu=.o)

# Include directories
INCLUDES := -I$(INCLUDE_DIR) -I.

# Debug flags (can be enabled with make DEBUG=1)
ifdef DEBUG
    CUDAFLAGS += -g -G -DDEBUG
    CFLAGS += -g -DDEBUG
endif

# Performance flags (enabled by default, disable with make PERF=0)
PERF ?= 1
ifeq ($(PERF), 1)
    CUDAFLAGS += -use_fast_math --optimize 3
    CFLAGS += -ffast-math -march=native
endif

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	$(NVCC) $(ARCH_FLAGS) $(CUDAFLAGS) -rdc=true -o $@ $^ $(LIBS)

# Compile .cu files
%.o: %.cu
	$(NVCC) $(ARCH_FLAGS) $(CUDAFLAGS) $(INCLUDES) -c -o $@ $<

# Clean build artifacts
clean:
	rm -f $(TARGET) $(SRC_DIR)/*.o $(KERNELS_DIR)/*.o *.dat *.dmp *.png *.gp

# Deep clean including output files
distclean: clean
	rm -f X*.dat graba.dmp retoma.dmp histogram_*.png plot_histograms_*.gp

# Format source code (requires clang-format)
format:
	clang-format -style=Google -i $(SRC_DIR)/*.cu $(INCLUDE_DIR)/*.h

# Run basic tests
test: $(TARGET)
	@echo "Running basic CUDA device query..."
	./$(TARGET) --version 2>/dev/null || echo "Built successfully"

# Generate compilation database for IDE support
compile_commands: clean
	bear -- $(MAKE) all

# Show GPU information
gpu-info:
	nvidia-smi
	@echo ""
	nvcc --version

# Profile with nvprof (requires CUDA toolkit)
profile: $(TARGET)
	nvprof --print-gpu-summary ./$(TARGET)

# Profile with nsys (requires Nsight Systems)
nsys-profile: $(TARGET)
	nsys profile --output=cuda_simulation ./$(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing required packages..."
	sudo apt update
	sudo apt install -y nvidia-cuda-toolkit build-essential clang-format bear

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the simulation (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  distclean    - Remove all generated files"
	@echo "  format       - Format source code"
	@echo "  test         - Run basic tests"
	@echo "  gpu-info     - Show GPU information"
	@echo "  profile      - Profile with nvprof"
	@echo "  nsys-profile - Profile with Nsight Systems"
	@echo "  install-deps - Install dependencies (Ubuntu/Debian)"
	@echo ""
	@echo "Build options:"
	@echo "  DEBUG=1      - Enable debug symbols and checks"
	@echo "  PERF=0       - Disable performance optimizations"
	@echo "  GPU_ARCH=XX  - Target specific GPU architecture (default: 80)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build with default settings"
	@echo "  make DEBUG=1            # Build with debug info"
	@echo "  make GPU_ARCH=75        # Build for Turing architecture"
	@echo "  make clean all          # Clean rebuild"

.PHONY: all clean distclean format test compile_commands gpu-info profile nsys-profile install-deps help 