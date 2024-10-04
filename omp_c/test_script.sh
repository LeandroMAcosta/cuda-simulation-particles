#!/bin/bash

make clean

make

perf stat ./main

gnuplot graph.plt