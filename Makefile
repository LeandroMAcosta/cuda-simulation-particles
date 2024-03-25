CC := gcc
CFLAGS := -Wall -Werror -Wextra -pedantic -std=c99 -O3 -march=native -std=c99
CLIBS := -lm -lpthread

TARGET := main

SRC := src/g1D-sp-075d.c
OBJ := $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(CLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) $(OBJ)