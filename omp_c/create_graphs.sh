#!/bin/bash

# Output GNU plot script filenames for both images
gnuplot_script1="plot_histograms_1_2.gp"
gnuplot_script2="plot_histograms_3_4.gp"

# Write the header for the first gnuplot script (1:2 columns)
cat <<EOF > $gnuplot_script1
set terminal pngcairo size 1200,600
set output 'histogram_1_2.png'
set title "OMP Simulation (Columns 1:2)"
set xlabel "x (Position)"
set ylabel "Population"
set style fill solid 1.0
set boxwidth 0.001
set key outside right top
# set yrange [0:5000]

p \\
EOF

# Write the header for the second gnuplot script (3:4 columns)
cat <<EOF > $gnuplot_script2
set terminal pngcairo size 1200,600
set output 'histogram_3_4.png'
set title "OMP Simulation (Columns 3:4)"
set xlabel "x (Position)"
set ylabel "Population"
set style fill solid 1.0
set boxwidth 0.001
set key outside right top

p \\
EOF

# Check if parameters are provided; if so, iterate over them, otherwise use *.dat files
if [ "$#" -gt 0 ]; then
    files=("$@")
else
    files=(*.dat)
fi

# Loop through all specified files and append the plot commands for both scripts
for file in "${files[@]}"; do
    if [[ "$file" == *.dat ]]; then
        echo "  '$file' u 1:2 w l t '$file', \\" >> $gnuplot_script1
        echo "  '$file' u 3:4 w l t '$file', \\" >> $gnuplot_script2
    fi
done

# Remove the trailing comma and backslash from the last line in both scripts
sed -i '$ s/, \\$//' $gnuplot_script1
sed -i '$ s/, \\$//' $gnuplot_script2

# Generate the first plot (1:2 columns)
echo "Running gnuplot script for columns 1:2..."
gnuplot $gnuplot_script1
echo "Generated histogram_1_2.png"

# Generate the second plot (3:4 columns)
echo "Running gnuplot script for columns 3:4..."
gnuplot $gnuplot_script2
echo "Generated histogram_3_4.png"

# Optionally, open both generated images
code --diff histogram_1_2.png histogram_3_4.png
