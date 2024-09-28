#!/bin/bash

# Set the C++ source file and output binary
OUTPUT_BINARY="main"

# --------------------- GPROF PROFILING ---------------------
echo "Running gprof profiling..."
# Run the program to generate gprof profiling data
./$OUTPUT_BINARY

# Generate gprof report
gprof $OUTPUT_BINARY gmon.out > gprof_report.txt
echo "gprof report generated: gprof_report.txt"

# --------------------- PERF PROFILING ----------------------
echo "Running perf profiling..."
# Record performance data with perf
perf record ./$OUTPUT_BINARY

# Generate perf report
perf report > perf_report.txt
echo "perf report generated: perf_report.txt"

# --------------------- VALGRIND (CALLGRIND) ----------------
echo "Running valgrind callgrind profiling..."
# Run the program with valgrind's callgrind tool
valgrind --tool=callgrind ./$OUTPUT_BINARY

# Annotate callgrind output
CALLGRIND_OUTPUT=$(ls callgrind.out.*)
callgrind_annotate $CALLGRIND_OUTPUT > callgrind_report.txt
echo "valgrind callgrind report generated: callgrind_report.txt"

# Optionally, clean up profiling data files
echo "Cleaning up intermediate files..."
rm -f gmon.out callgrind.out.*

echo "Performance analysis complete!"
