CC := gcc
CFLAGS := -Wall -Werror -Wextra -pedantic -std=c99 -O3 -march=native -std=c99
CLIBS := -lm -lpthread
OMP := -fopenmp

TARGET := main

SRC := src/g1D-sp-075d.c src/utils.c
OBJ := $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(CLIBS) $(OMP)

%.o: %.c
	$(CC) $(CFLAGS) $(OMP) -c -o $@ $<

format:
	clang-format -style=Microsoft -i src/*.c include/*.h

clean:
	rm -f $(TARGET) $(OBJ) graba.dmp hists.eps *.dat
