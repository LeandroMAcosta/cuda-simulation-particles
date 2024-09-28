# Compiler and flags
CXX = g++
CXXFLAGS = -fopenmp -std=c++14 -Wall -Iinclude

# Target executable
TARGET := main

# Source and object files
SRC := src/main.cpp src/utils.cpp
OBJ := $(SRC:.c=.o)

# Default rule to build the target
all: $(TARGET)

# Linking the object files to create the final executable
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile each source file into object files
%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
