set terminal pngcairo size 800,600
set output 'histogram-distribucion-impulso.png'
set title "CUDA Simulation"
set xlabel "x (Position)"
set ylabel "Population"
set style fill solid 1.0
set boxwidth 0.001
set yrange [0:12000]  # Limit y-axis to 5000

p 'X0000000.dat' u 3:4 w l t 'X0000000.dat', \
  'X0000100.dat' u 3:4 w l t 'X0000100.dat', \
  'X0001100.dat' u 3:4 w l t 'X0001100.dat'