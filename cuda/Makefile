CC := nvcc
CFLAGS := -std=c++14 -Xcompiler="-Wall -Wextra -pedantic -O3 -march=native -ffast-math" # Removed -Werror for CUDA
CLIBS := -lm -lgomp
OMP := -Xcompiler=-fopenmp

# Target name
TARGET := main

# Automatically gather source files
CPP_SRC := $(wildcard src/*.cpp src/utils/*.cpp)
CU_SRC := $(wildcard src/kernels/*.cu)

# Object files
CPP_OBJ := $(CPP_SRC:.cpp=.o)
CU_OBJ := $(CU_SRC:.cu=.o)

all: $(TARGET)

# Linking step
$(TARGET): $(CPP_OBJ) $(CU_OBJ)
	$(CC) $(CFLAGS) $(OMP) -o $@ $^ $(CLIBS)

# Compilation step for C++ source files
%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

# Compilation step for CUDA source files
%.o: %.cu
	$(CC) $(CFLAGS) $(OMP) -c -o $@ $<


format:
	clang-format -style=Microsoft -i src/*.cu include/*.h

clean:
	rm -f $(TARGET) $(OBJ)
