# CC := nvcc
# CFLAGS := -std=c++14 -Xcompiler="-Wall -Wextra -pedantic -O3 -march=native -ffast-math"
# CLIBS := -lm -lgomp
# OMP := -Xcompiler=-fopenmp

# # Target name
# TARGET := build/main

# # Automatically gather source files
# CPP_SRC := $(wildcard src/*.cpp src/utils/*.cpp)
# CU_SRC := $(wildcard src/main.cu src/kernels/*.cu)

# # Object files in ./build/ folder
# CPP_OBJ := $(CPP_SRC:src/%.cpp=build/%.o)
# CU_OBJ := $(CU_SRC:src/kernels/%.cu=build/kernels/%.o) $(CU_SRC:src/main.cu=build/main.o)

# # Ensure the build directory exists
# $(shell mkdir -p build build/kernels build/utils)

# all: $(TARGET)

# # Linking step
# $(TARGET): $(CPP_OBJ) $(CU_OBJ)
# 	$(CC) $(CFLAGS) $(OMP) -o $@ $^ $(CLIBS)

# # # Compilation step for C++ source files, place .o files in build/
# # build/%.o: src/%.cpp
# # 	$(CC) $(CFLAGS) -c -o $@ $<

# # # Compilation step for CUDA source files, place .o files in build/kernels/
# # build/kernels/%.o: src/kernels/%.cu
# # 	$(CC) $(CFLAGS) $(OMP) -c -o $@ $<


# # Compilation step for C++ source files, place .o files in build/
# build/%.o: src/%.cpp
# 	$(CC) $(CFLAGS) -c -o $@ $<

# # Compilation step for CUDA source files, place .o files in build/kernels/
# build/kernels/%.o: src/kernels/%.cu
#     $(CC) $(CFLAGS) $(OMP) -c -o $@ $<

# # Compilation step for main.cu, place .o files in build/
# build/%.o: src/%.cu
#     $(CC) $(CFLAGS) $(OMP) -c -o $@ $<


# format:
# 	clang-format -style=Microsoft -i src/*.cu include/*.h

# clean:
# 	rm -f build/main build/*.o build/kernels/*.o build/utils/*.o
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
