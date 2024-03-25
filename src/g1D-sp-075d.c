#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include "../include/utils.h"

// gcc -Wall -Wextra -Werror -std=c99 -pedantic -g -O3 -o g1D-sp-075d g1D-sp-075d.c -lm -lpthread

/*
 gas 1D con p discretos, SIN ruido en L del recipiente + ruido en p al rebotar con la pared

 a cada E_j le corresponde una Ej' sorteada con una distribución uniforme en [Ej-DeltaE/2, Ej+DeltaE/2], DeltaE=alfa((p^0.75-pmin^0.75)(pmax^0.75-p^0.75))^4+d

 En la distribucion en x permitimos que los bordes no sean filosos: agregamos 2 canales en cada extremo, redefiniendo los chi2x

NO: Al cabo de un ciclo, antes de grabar el dmp, pasamos el 50% de la energia de (14) particulas con |p|>3sigma a 14 particulas con 0.15sigma<|p|<.9sigma
 */

#define PI 3.14159265358979323846
// #define N_PART (1 << 21)            // 2097152
// #define BINS 500                    // (2*BINS+1)
#define epsmax2M 9.0E-46 // E maxima
#define DEmax2M 6.0e-50  // mas grande para pmax (3.5964e-48: 1 canal de p menos que el max) // OJO: ahora DEmax2M nos da 1.2e-50 (8/2/24)
                         // 6.296208e-49 // 2*5.24684E-24*6.0e-26
#define epsmin2M 9.0E-52 // 2m * E minima = pmin^2

typedef struct
{
  int s, e;
  double *xx, *pp;
  int *hh, *gg, *hghg;
  unsigned int Ntandas;
  int *steps;
  int BINS;
  double DT, M, alfa, pmin075, pmax075;
} range_t;

sem_t iter_sem, hist_sem; // semaphore

static void atomic_increment(int *ptr)
{
  __sync_fetch_and_add(ptr, 1);
}

void *work(void *range)
{
  int s = ((range_t *)range)->s;
  int e = ((range_t *)range)->e;
  double *x = ((range_t *)range)->xx;
  double *p = ((range_t *)range)->pp;
  int *h = ((range_t *)range)->hh;
  int *g = ((range_t *)range)->gg;
  int *hg = ((range_t *)range)->hghg;
  unsigned int Ntandas = ((range_t *)range)->Ntandas;
  int *steps = ((range_t *)range)->steps;
  int BINS = ((range_t *)range)->BINS;
  double DT = ((range_t *)range)->DT;
  double M = ((range_t *)range)->M;
  double alfa = ((range_t *)range)->alfa;
  double pmin075 = ((range_t *)range)->pmin075;
  double pmax075 = ((range_t *)range)->pmax075;
  for (unsigned int j = 0; j < Ntandas; j++)
  {
    sem_wait(&iter_sem); // semaforo que señala que un hilo queda reservado para ejecución

    iter_in_range(steps[j], s, e, x, p, DT, M, alfa, pmin075, pmax075); // avanza steps[j] pasos en el rango de partículas [s, e)

    for (int i = s; i < e; i++)
    {
      int h_idx = (2.0 * x[i] + 1) * BINS + 2.5;
      int g_idx = (p[i] / 3.0e-23 + 1) * BINS + 0.5;
      atomic_increment(h + h_idx);                           // incrementa el casillero h_idx de h
      atomic_increment(g + g_idx);                           // incrementa el casillero g_idx de g
      atomic_increment(hg + (2 * BINS + 1) * h_idx + g_idx); // incrementa el casillero de hg
    }
    sem_post(&hist_sem); // semaforo que lo libera
  }
  return NULL;
}

int main(void)
{
  int N_THREADS, N_PART, BINS, steps[50], retoma, dump;
  unsigned int Ntandas;
  char indat[255], saldat[255];
  double DT, M, sigmaL;

  srand(time(NULL));

  double xi1, xi2;
  int X0 = 1;
  char filename[32];

  double d = 1.0e-72, alfa = 4.0E-88;
  int evolution = 0;
  double pmin075 = 7.20843424240426E-020, pmax075 = 1.2818610191887E-017;

  char data_filename[] = "datos.in";
  load_parameters_from_file(data_filename, &N_PART, &BINS, &DT, &M, &N_THREADS, &Ntandas, steps, indat, saldat, &retoma, &dump, &sigmaL);

  double *x = malloc(sizeof(double) * N_PART);
  double *p = malloc(sizeof(double) * N_PART);
  double *DxE = malloc(sizeof(double) * (2 * BINS + 5));
  double *DpE = malloc(sizeof(double) * (2 * BINS + 1));
  int *h = malloc(sizeof(int) * (2 * BINS + 5));
  int *g = malloc(sizeof(int) * (2 * BINS + 1));
  //    int *hg = calloc(2*BINS+1, sizeof(int) * (2*BINS+1)) ; // &&&&&&&&&&&&&&&&
  int *hg = malloc(sizeof(int) * (2 * BINS + 5) * (2 * BINS + 1)); // &&&&&&&&&&&&&&&&

  for (int i = 0; i <= BINS << 1; i++)
  {
    DpE[i] = 6.0E-26 * N_PART / (5.24684E-24 * sqrt(2.0 * PI)) * exp(-pow(3.0e-23 * (1.0 * i / BINS - 1) / 5.24684E-24, 2) / 2);
  }
  for (int i = 0; i <= (BINS + 2) << 1; i++)
  {
    DxE[i] = 1.0E-3 * N_PART;
  }
  DxE[0] = 0.0;                                // DxE[0]*0.02;
  DxE[1] = 0.0;                                // DxE[1]*0.08;
  DxE[2] = DxE[2] * 0.5;                       // DxE[2]*0.4;
  DxE[2 * BINS + 2] = DxE[2 * BINS + 2] * 0.5; // DxE[2*BINS+2]*0.4;
  DxE[2 * BINS + 3] = 0.0;                     // DxE[2*BINS+3]*0.08;
  DxE[2 * BINS + 4] = 0.0;                     // DxE[2*BINS+4]*0.02;

  memset(h, 0, (2 * BINS + 5) * sizeof(int));
  memset(g, 0, (2 * BINS + 1) * sizeof(int));
  memset(hg, 0, (2 * BINS + 5) * (2 * BINS + 1) * sizeof(int)); // &&&&&&&&&&&&&&&&
  if (retoma != 0)
  { // No retoma
    while (X0 == 1)
    {
      // initialize particles
      for (int i = 0; i < N_PART; i++)
        x[i] = d_rand() * 0.5;
      for (int i = 0; i < N_PART >> 1; i++)
      {
        xi1 = sqrt(-2.0 * log(d_rand() + 1E-35));
        xi2 = 2.0 * PI * d_rand();
        p[2 * i] = xi1 * cos(xi2) * 5.24684E-24;
        p[2 * i + 1] = xi1 * sin(xi2) * 5.24684E-24;
      }
      for (int i = 0; i < N_PART; i++)
      {
        h[(int)((2.0 * x[i] + 1) * BINS + 2.5)]++;
        g[(int)((p[i] / 3.0e-23 + 1) * BINS + 0.5)]++;
      }
      //        printf("fabricó h y g     a calcular hg...\n");
      for (int i = 0; i < N_PART; i++)
      { // &&&&&&&&&&&&&&&&
        hg[(2 * BINS + 1) * (int)((2.0 * x[i] + 1) * BINS + 2.5) + (int)((p[i] / 3e-23 + 1) * BINS + 0.5)]++;
      } // &&&&&&&&&&&&&&&&
        //        printf("fabricó hg\n");
      X0 = make_hist(h, g, hg, DxE, DpE, "X0000000.dat", BINS);
      if (X0 == 1)
        printf("falló algún chi2:   X0 =%1d\n", X0);
    }
  }
  else
  {
    read_data(indat, x, p, &evolution, N_PART);
  } // Sí retoma
  energy_sum(p, N_PART, evolution, M);
  printf("d=%12.9E  alfa=%12.9E\n", d, alfa);

  sem_init(&iter_sem, 0, 0);
  sem_init(&hist_sem, 0, 0);

  // create threads
  pthread_t threads[N_THREADS]; // indentificadores de c/hilo (tipo específico pthread_t)
  range_t args[N_THREADS];      // c/hilo se encarga de un grupo de partículas
  for (int i = 0; i < N_THREADS; i++)
  {
    args[i] = (range_t){
        .s = (N_PART / N_THREADS) * i,
        .e = (N_PART / N_THREADS) * (i + 1),
        .xx = x,
        .pp = p,
        .hh = h,
        .gg = g,
        .hghg = hg,
        .Ntandas = Ntandas,
        .steps = steps,
        .BINS = BINS,
        .DT = DT,
        .M = M,
        .alfa = alfa,
        .pmin075 = pmin075,
        .pmax075 = pmax075};
    pthread_create(&threads[i], NULL, work, &args[i]);
  }

  for (unsigned int i = 0; i < Ntandas; i++)
  {
    for (int i = 0; i < N_THREADS; i++)
      sem_post(&iter_sem);
    for (int i = 0; i < N_THREADS; i++)
      sem_wait(&hist_sem);

    evolution += steps[i];
    if (evolution < 10000000)
    {
      sprintf(filename, "X%07d.dat", evolution);
    }
    else
    {
      sprintf(filename, "X%1.3e.dat", (double)evolution);
      char *e = memchr(filename, 'e', 32);
      strcpy(e + 1, e + 3);
    }

    if (dump == 0)
      save_data(saldat, x, p, evolution, N_PART);
    make_hist(h, g, hg, DxE, DpE, filename, BINS);
    energy_sum(p, N_PART, evolution, M);
  }
  printf("Completo evolution = %d\n", evolution);

  for (int i = 0; i < N_THREADS; i++)
    pthread_join(threads[i], NULL);
  free(x);
  free(p);
  free(DxE);
  free(DpE);
  free(h);
  free(g);
  free(hg);

  return 0;
}

/* para graficar en el gnuplot: (sacando un archivo "hists.eps")
set terminal postscript enhanced color eps 20
set output "hists.eps"
# histograma de x  (las dos líneas siguientes alcanzan para graficar las x dentro del gnuplot)
set style fill solid 1.0
# o medio transparente:
set style fill transparent solid 0.5 noborder
set key left ; set xrange[-0.5:0.5]
p 'X1000000.dat' u 1:2 w boxes lc rgb "#dddddd" t 'X1000000.dat' , 'X2000000.dat' u 1:2 w boxes lc rgb "#77ff77" t 'X2000000.dat' , 'X2000001.dat' u 1:2 w boxes lc "#ffaaaa" t 'X2000001.dat' , 'X2000002.dat' u 1:2 w boxes lc "#dddd55" t 'X2000002.dat' , 'X2000003.dat' u 1:2 w boxes lc rgb "#ffdddd" t 'X2000003.dat' , 'X2000008.dat' u 1:2 w boxes lc rgb "#cc44ff" t 'X2000008.dat' , 'X2000018.dat' u 1:2 w boxes lc rgb "#888888" t 'X2000018.dat' , 'X2000028.dat' u 1:2 w boxes lc rgb "#bbddbb" t 'X2000028.dat' , 'X2000038.dat' u 1:2 w boxes lc rgb "#ffee00" t 'X2000038.dat' , 'X2000048.dat' u 1:2 w boxes lc rgb "#8844ff" t 'X2000048.dat' , 'X2000058.dat' u 1:2 w boxes lc rgb "#cceeff" t 'X2000058.dat' , 'X2000068.dat' u 1:2 w boxes lc rgb "#44bb44" t 'X2000068.dat' , 'X2000078.dat' u 1:2 w boxes lc rgb "#99ee77" t 'X2000078.dat' , 'X2000088.dat' u 1:2 w boxes lc rgb "#ffdd66" t 'X2000088.dat' , 'X2000098.dat' u 1:2 w boxes lc rgb "#4444ff" t 'X2000098.dat'
# histograma de p  (las dos líneas siguientes alcanzan para graficar las p dentro del gnuplot)
set key left ; set xrange[-3e-23:3e-23]
p 'X0000500.dat' u 3:4 w boxes lc rgb "#dddddd" t 'X0000500.dat' , 'X0001000.dat' u 3:4 w boxes lc rgb "#77ff77" t 'X0001000.dat' , 'X0002000.dat' u 3:4 w boxes lc "#ffaaaa" t 'X0002000.dat' , 'X0005000.dat' u 3:4 w boxes lc "#dddd55" t 'X0005000.dat' , 'X0010000.dat' u 3:4 w boxes lc rgb "#ffdddd" t 'X0010000.dat' , 'X0020000.dat' u 3:4 w boxes lc rgb "#cc44ff" t 'X0020000.dat' , 'X0050000.dat' u 3:4 w boxes lc rgb "#888888" t 'X0050000.dat' , 'X0100000.dat' u 3:4 w boxes lc rgb "#bbddbb" t 'X0100000.dat' , 'X0200000.dat' u 3:4 w boxes lc rgb "#ffee00" t 'X0200000.dat' , 'X0500000.dat' u 3:4 w boxes lc rgb "#8844ff" t 'X0500000.dat' , 'X0995000.dat' u 3:4 w boxes lc rgb "#cceeff" t 'X0995000.dat' , 'X0999000.dat' u 3:4 w boxes lc rgb "#44bb44" t 'X0999000.dat' , 'X0999500.dat' u 3:4 w boxes lc rgb "#99ee77" t 'X0999500.dat' , 'X1000000.dat' u 3:4 w boxes lc rgb "#ffdd66" t 'X1000000.dat' , 'X2000000.dat' u 3:4 w boxes lc rgb "#4444ff" t 'X2000000.dat'
set terminal qt


p 'X0000001.dat' u 1:2 w boxes lc rgb "#dddddd" t 'X0000001.dat' , 'X0000100.dat' u 1:2 w boxes lc rgb "#77ff77" t 'X0000100.dat' , 'X0001000.dat' u 1:2 w boxes lc "#ffaaaa" t 'X0001000.dat' , 'X0001200.dat' u 1:2 w boxes lc "#dddd55" t 'X0001200.dat' , 'X0001400.dat' u 1:2 w boxes lc rgb "#ffdddd" t 'X0001400.dat' , 'X0001500.dat' u 1:2 w boxes lc rgb "#cc44ff" t 'X0001500.dat' , 'X0001600.dat' u 1:2 w boxes lc rgb "#888888" t 'X0001600.dat' , 'X0001700.dat' u 1:2 w boxes lc rgb "#bbddbb" t 'X0001700.dat' , 'X0001800.dat' u 1:2 w boxes lc rgb "#ffee00" t 'X0001800.dat' , 'X0001900.dat' u 1:2 w boxes lc rgb "#8844ff" t 'X0001900.dat' , 'X0002000.dat' u 1:2 w boxes lc rgb "#cceeff" t 'X0002000.dat'

*/