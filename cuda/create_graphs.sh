#!/bin/bash

# Output GNU plot script filename
gnuplot_script="plot_histograms.gp"

# Write the header for the gnuplot script
cat <<EOF > $gnuplot_script
set terminal pngcairo size 1200,600
set output 'histogram.png'
set title "CUDA Simulation"
set xlabel "x (Position)"
set ylabel "Population"
set style fill solid 1.0
set boxwidth 0.001

# Move the legend to the outside of the plot
set key outside right top

p \\
EOF

# Loop through all .dat files and append the plot commands
for file in *.dat; do
    if [[ "$file" == *.dat ]]; then
        echo "  '$file' u 1:2 w l t '$file', \\" >> $gnuplot_script
    fi
done

# Remove the trailing comma and backslash from the last line
sed -i '$ s/, \\$//' $gnuplot_script

echo "Generated GNU plot script: $gnuplot_script"

echo "Running gnuplot script..."
gnuplot $gnuplot_script

code histogram.png