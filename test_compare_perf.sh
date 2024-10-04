#!/bin/bash

# Change to the cuda directory
cd cuda || { echo "Failed to change to cuda directory"; exit 1; }

# Execute the test script in cuda directory
if ./test_script.sh; then
    echo "Successfully executed test_script.sh in cuda"
else
    echo "Failed to execute test_script.sh in cuda"
    exit 1
fi

# Change back to the original directory
cd ../omp_c || { echo "Failed to change to omp_c directory"; exit 1; }

# Execute the test script in omp_c directory
if ./test_script.sh; then
    echo "Successfully executed test_script.sh in omp_c"
else
    echo "Failed to execute test_script.sh in omp_c"
    exit 1
fi
