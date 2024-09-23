CC := nvcc
CFLAGS := -Xcompiler="-Wall -Wextra -pedantic -O3 -march=native -ffast-math" # Removed -Werror for CUDA
CLIBS := -lm -lgomp
OMP := -Xcompiler=-fopenmp

TARGET := main

SRC := src/g1D-1000.cu src/utils.cu
OBJ := $(SRC:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OMP) -o $@ $^ $(CLIBS)

%.o: %.cu
	$(CC) $(CFLAGS) $(OMP) -c -o $@ $<

format:
	clang-format -style=Microsoft -i src/*.cu include/*.h

clean:
	rm -f $(TARGET) $(OBJ)
