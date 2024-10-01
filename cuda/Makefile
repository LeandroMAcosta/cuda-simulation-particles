CC := nvcc
CFLAGS := -std=c++14 -Xcompiler="-Wall -Wextra -pedantic -O3 -march=native -ffast-math -g"

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

TARGET := main

all: $(TARGET)

$(TARGET): $(OBJS) $(SRC_DIR)/main.cu
	$(CC) $(CFLAGS) $(OMP) -I$(INCLUDE_DIR) -o $@ $(SRC_DIR)/main.cu $(OBJS) $(CLIBS)

$(KERNELS_OBJ_DIR)/histogram_kernels.o: $(KERNELS_DIR)/histogram_kernels.cu $(DEPS)
	@mkdir -p $(KERNELS_OBJ_DIR)
	$(CC) $(CFLAGS) $(OMP) -I$(INCLUDE_DIR) -c $(KERNELS_DIR)/histogram_kernels.cu -o $@

$(UTILS_OBJ_DIR)/utils.o: $(UTILS_DIR)/utils.cpp $(DEPS)
	@mkdir -p $(UTILS_OBJ_DIR)
	$(CC) $(CFLAGS) $(OMP) -I$(INCLUDE_DIR) -c $(UTILS_DIR)/utils.cpp -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
