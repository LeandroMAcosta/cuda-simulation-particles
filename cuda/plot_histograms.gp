set terminal pngcairo size 1200,600
set output 'histogram.png'
set title "CUDA Simulation"
set xlabel "x (Position)"
set ylabel "Population"
set style fill solid 1.0
set boxwidth 0.001

# Move the legend to the outside of the plot
set key outside right top

p \
  'X0000000.dat' u 1:2 w l t 'X0000000.dat', \
  'X0200000.dat' u 1:2 w l t 'X0200000.dat', \
  'X0400000.dat' u 1:2 w l t 'X0400000.dat', \
  'X0600000.dat' u 1:2 w l t 'X0600000.dat', \
  'X0800000.dat' u 1:2 w l t 'X0800000.dat', \
  'X1000000.dat' u 1:2 w l t 'X1000000.dat', \
  'X1200000.dat' u 1:2 w l t 'X1200000.dat', \
  'X1400000.dat' u 1:2 w l t 'X1400000.dat'
