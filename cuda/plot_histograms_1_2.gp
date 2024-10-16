set terminal pngcairo size 1200,600
set output 'histogram_1_2.png'
set title "CUDA Simulation (Columns 1:2)"
set xlabel "x (Position)"
set ylabel "Population"
set style fill solid 1.0
set boxwidth 0.001
set key outside right top

p \
  'X0000000.dat' u 1:2 w l t 'X0000000.dat', \
  'X0000100.dat' u 1:2 w l t 'X0000100.dat', \
  'X0001100.dat' u 1:2 w l t 'X0001100.dat', \
  'X0003100.dat' u 1:2 w l t 'X0003100.dat', \
  'X0008100.dat' u 1:2 w l t 'X0008100.dat', \
  'X0018100.dat' u 1:2 w l t 'X0018100.dat'
