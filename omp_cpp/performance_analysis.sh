#!/bin/bash

# Set the C++ source file and output binary

# --------------------- PERF PROFILING ----------------------
echo "Running perf profiling..."
# Record performance data with perf
perf record -g ./main

# Generate perf report
perf report > perf_report.txt

perf stat ./main > perf_report_stat.txt

echo "Performance analysis complete!"
