CC := nvcc
CFLAGS := -std=c++14 -Xcompiler="-Wall -Wextra -pedantic -O3 -march=native -ffast-math -g" -arch=sm_61

CLIBS := -lm -lgomp
OMP := -Xcompiler=-fopenmp

SRC_DIR := src
BUILD_DIR := build

KERNELS_DIR := $(SRC_DIR)/kernels
UTILS_DIR := $(SRC_DIR)/utils
INCLUDE_DIR := $(SRC_DIR)/include

KERNELS_OBJ_DIR := $(BUILD_DIR)/kernels
UTILS_OBJ_DIR := $(BUILD_DIR)/utils

OBJS := $(KERNELS_OBJ_DIR)/histogram_kernels.o $(UTILS_OBJ_DIR)/utils.o
DEPS := $(INCLUDE_DIR)/constants.h $(INCLUDE_DIR)/histogram_kernels.h $(INCLUDE_DIR)/utils.h

# Test script
TEST_SCRIPT = ./test_script.sh  # Replace with your actual script name


TARGET := main

all: $(TARGET)

$(TARGET): $(OBJS) $(SRC_DIR)/main.cu
	$(CC) $(CFLAGS) $(OMP) -I$(INCLUDE_DIR) -o $@ $(SRC_DIR)/main.cu $(OBJS) $(CLIBS)

$(KERNELS_OBJ_DIR)/histogram_kernels.o: $(KERNELS_DIR)/histogram_kernels.cu $(DEPS)
	@mkdir -p $(KERNELS_OBJ_DIR)
	$(CC) $(CFLAGS) $(OMP) -I$(INCLUDE_DIR) -c $(KERNELS_DIR)/histogram_kernels.cu -o $@

$(UTILS_OBJ_DIR)/utils.o: $(UTILS_DIR)/utils.cu $(DEPS)
	@mkdir -p $(UTILS_OBJ_DIR)
	$(CC) $(CFLAGS) $(OMP) -I$(INCLUDE_DIR) -c $(UTILS_DIR)/utils.cu -o $@

clean:
	@echo "Cleaning up..."
	rm -rf $(BUILD_DIR) $(TARGET)

test:
	@echo "Running test script..."
	bash $(TEST_SCRIPT)

run: $(TARGET)
	./$(TARGET)

run_output: $(TARGET)
	./$(TARGET) > output.txt

graph: run_output
	gnuplot graph.plt