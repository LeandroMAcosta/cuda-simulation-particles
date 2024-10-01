CC := nvcc
CFLAGS := -std=c++14 -Xcompiler="-Wall -Wextra -pedantic -O3 -march=native -ffast-math"
CLIBS := -lm -lgomp
OMP := -Xcompiler=-fopenmp

# Target name
TARGET := build/main

# Automatically gather source files
CPP_SRC := $(wildcard src/*.cpp src/utils/*.cpp)
CU_SRC := $(wildcard src/kernels/*.cu)

# Object files in ./build/ folder
CPP_OBJ := $(CPP_SRC:src/%.cpp=build/%.o)
CU_OBJ := $(CU_SRC:src/kernels/%.cu=build/kernels/%.o)

# Ensure the build directory exists
$(shell mkdir -p build build/kernels)

all: $(TARGET)

# Linking step
$(TARGET): $(CPP_OBJ) $(CU_OBJ)
	$(CC) $(CFLAGS) $(OMP) -o $@ $^ $(CLIBS)

# Compilation step for C++ source files, place .o files in build/
build/%.o: src/%.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

# Compilation step for CUDA source files, place .o files in build/kernels/
build/kernels/%.o: src/kernels/%.cu
	$(CC) $(CFLAGS) $(OMP) -c -o $@ $<

format:
	clang-format -style=Microsoft -i src/*.cu include/*.h

clean:
	rm -f build/main build/*.o build/kernels/*.o
