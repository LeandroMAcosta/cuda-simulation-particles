set terminal pngcairo size 800,600
set output 'histogram.png'
set title "OMP Simulation"
set xlabel "x (Position)"
set ylabel "Population"
set style fill solid 1.0
set boxwidth 0.001

p 'X0000000.dat' u 1:2 w l t 'X0000000.dat', \
  'X0010000.dat' u 1:2 w l t 'X0010000.dat'

