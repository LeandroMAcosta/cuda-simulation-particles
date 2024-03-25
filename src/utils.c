#include "../include/utils.h"

double d_rand()
{
    return (rand() / (double)RAND_MAX);
}

void load_parameters_from_file(char filename[], int *N_PART, int *BINS, double *DT, double *M, int *N_THREADS,
                               unsigned int *Ntandas, int steps[], char inputFilename[], char saveFilename[],
                               int *retake, int *dump, double *sigmaL)
{
    char du[4];
    FILE *inputFile = fopen(filename, "r");
    if (inputFile == NULL)
    {
        printf("Error al abrir el archivo %s\n", filename);
        exit(1);
    }
    fscanf(inputFile, " %*[^\n]");
    fscanf(inputFile, " %*[^:]: %d", N_PART);
    fscanf(inputFile, " %*[^:]: %d", BINS);
    fscanf(inputFile, " %*[^:]: %le", DT);
    fscanf(inputFile, " %*[^:]: %le", M);
    fscanf(inputFile, " %*[^:]: %d", N_THREADS);
    fscanf(inputFile, " %*[^\n]");
    *Ntandas = 0;
    while (fscanf(inputFile, " %d", &steps[*Ntandas]) == 1)
    {
        (*Ntandas)++;
    }
    fscanf(inputFile, " %*[^:]: %s %s", du, inputFilename);
    *retake = strcmp(du, "sí");
    printf("%s lee %s\t", du, inputFilename);
    fscanf(inputFile, " %*[^:]: %s %s", du, saveFilename);
    printf("%s escribe %s\t", du, saveFilename);
    *dump = strcmp(du, "sí");
    fscanf(inputFile, " %*[^:]: %le", sigmaL);
    printf("sigma(L) = %le\n", *sigmaL);
    fclose(inputFile);
}

void read_data(char filename[], double *x, double *p, int *evolution, int N_PART)
{
    FILE *readFile = fopen(filename, "r");
    if (readFile == NULL)
    {
        printf("Error al abrir el archivo %s\n", filename);
        exit(1);
    }
    fread(evolution, sizeof(*evolution), 1, readFile);
    fread(x, sizeof(x[0]) * N_PART, 1, readFile);
    fread(p, sizeof(p[0]) * N_PART, 1, readFile);
    fclose(readFile);
}

void energy_sum(double *p, int N_PART, int evolution, double M)
{
    double sumEnergy = 0;
    for (int i = 0; i < N_PART; i++)
    {
        sumEnergy += p[i] * p[i];
    }
    printf("N° de pasos %6d\tEnergía total = %12.9E\n", evolution, sumEnergy / (2 * M));
}

void save_data(char filename[], double *x, double *p, int evolution, int N_PART)
{
    FILE *saveFile = fopen(filename, "w");
    if (saveFile == NULL)
    {
        printf("Error al abrir el archivo %s\n", filename);
        exit(1);
    }
    fwrite(&evolution, sizeof(evolution), 1, saveFile);
    fwrite(x, sizeof(x[0]) * N_PART, 1, saveFile);
    int Npmod = (0 * N_PART) / (1 << 21);
    if (evolution % 1000000 == 0 && Npmod > 0)
    {
        double f = 0.7071; // fraccion de p+ que queda en p+'
        double *sqrtp2 = malloc(sizeof(double) * Npmod);
        int np = 0;
        int i0 = d_rand() * N_PART;
        int i = i0;
        while ((np < Npmod) && (i < N_PART))
        {
            if (fabs(p[i]) > (2.43 + 0.3 * np / Npmod) * 5.24684E-24)
            {
                sqrtp2[np] = sqrt(1.0 - f * f) * p[i];
                np++;
                p[i] *= f;
            }
            i++;
        }
        i = 0;
        while ((np < Npmod) && (i < i0))
        {
            if (fabs(p[i]) > (2.43 + 0.3 * np / Npmod) * 5.24684E-24)
            {
                sqrtp2[np] = sqrt(1.0 - f * f) * p[i];
                np++;
                p[i] *= f;
            }
            i++;
        }
        printf("np=%d   (2.43-2.73)sigma\n", np);
        np = 0; // repartimos 0.5*E+ en 2 partes iguales
        while ((np < Npmod) && (i < N_PART))
        {
            int signopr = copysign(1.0, sqrtp2[np]);
            if ((signopr * p[i] > 0) && (fabs(p[i]) > 0.15 * 5.24684E-24) && (fabs(p[i]) < 0.9 * 5.24684E-24))
            {
                p[i] = sqrt(p[i] * p[i] + sqrtp2[np] * sqrtp2[np] / 2.0);
                np++;
            }
            i++;
        }
        i = 0;
        while (np < Npmod)
        {
            int signopr = copysign(1.0, sqrtp2[np]);
            if ((signopr * p[i] > 0) && (fabs(p[i]) > 0.15 * 5.24684E-24) && (fabs(p[i]) < 0.9 * 5.24684E-24))
            {
                p[i] = sqrt(p[i] * p[i] + sqrtp2[np] * sqrtp2[np] / 2.0);
                np++;
            }
            i++;
        }
        np = 0; // otra vez la busqueda de p chicos, porque repartimos la otra
                // mitad de E+
        while ((np < Npmod) && (i < N_PART))
        {
            int signopr = copysign(1.0, sqrtp2[np]);
            if ((signopr * p[i] > 0) && (fabs(p[i]) > 0.15 * 5.24684E-24) && (fabs(p[i]) < 0.9 * 5.24684E-24))
            {
                p[i] = sqrt(p[i] * p[i] + sqrtp2[np] * sqrtp2[np] / 2.0);
                np++;
            }
            i++;
        }
        i = 0;
        while (np < Npmod)
        {
            int signopr = copysign(1.0, sqrtp2[np]);
            if ((signopr * p[i] > 0) && (fabs(p[i]) > 0.15 * 5.24684E-24) && (fabs(p[i]) < 0.9 * 5.24684E-24))
            {
                p[i] = sqrt(p[i] * p[i] + sqrtp2[np] * sqrtp2[np] / 2.0);
                np++;
            }
            i++;
        }
        free(sqrtp2);
    }
    fwrite(p, sizeof(p[0]) * N_PART, 1, saveFile);
    fclose(saveFile);
}

void iter_in_range(int n, int s, int e, double *x, double *p, double DT, double M, double alfa, double pmin075,
                   double pmax075)
{
    long int k;
    int signop;
    for (int i = s; i < e; i++)
    {
        double x_tmp = x[i], p_tmp = p[i];
        for (int step = 0; step < n; step++)
        {
            x_tmp = x_tmp + p_tmp * DT / M;
            signop = copysign(1.0, p_tmp);
            k = trunc(x_tmp + 0.5 * signop);
            if (k != 0)
            {
                x_tmp = (k % 2 ? -1.0 : 1.0) * (x_tmp - k);
                if (fabs(x_tmp) > 0.502)
                {
                    x_tmp = 1.004 * copysign(1.0, x_tmp) - x_tmp;
                }
                for (int j = 1; j <= labs(k); j++)
                {
                    double ptmp075 = pow((fabs(p_tmp)), 0.75);
                    double DeltaE = alfa * pow((ptmp075 - pmin075) * (pmax075 - ptmp075), 4); //+d;
                    //    if (fabs(DeltaE) > p_tmp*p_tmp ) {
                    //         printf("i=%d step=%d  p_tmp=%le  rpmin=%le
                    //         rpmax=%le   DeltaE=%le  fabs(DE)=%le
                    //         k=%ld\n",i,step,p_tmp,rpmin,rpmax,DeltaE,fabs(DeltaE),k);
                    //       }
                    p_tmp = sqrt(p_tmp * p_tmp + DeltaE * (d_rand() - 0.5));
                }
                p_tmp = (k % 2 ? -1.0 : 1.0) * signop * p_tmp;
            }
        }
        x[i] = x_tmp;
        p[i] = p_tmp;
    }
}

int make_hist(int *h, int *g, int *hg, double *DxE, double *DpE, const char *filename, int BINS)
{
    double chi2x = 0.0, chi2xr = 0.0, chi2p = 0.0, chiIp = 0.0, chiPp = 0.0, chiIx = 0.0, chiPx = 0.0;

    if (strcmp(filename, "X0000000.dat") == 0)
    {
        for (int i = BINS + 1; i <= 2 * BINS; i++)
        {
            chi2x += pow(h[i] - 2 * DxE[i], 2) / (2 * DxE[i]);
        }
        chi2x = (chi2x + pow(h[BINS] - DxE[BINS], 2) / DxE[BINS]) / (BINS + 1);
    }
    else
    {
        for (int i = 2; i <= 2 * (BINS + 1); i++)
        {
            chi2x += pow(h[i] - DxE[i], 2) / DxE[i];
        }
        //    chi2xr = chi2x;
        //    for (int i = 0; i <= 2; i++) {
        //      chi2xr -=  pow(h[i]-DxE[i],2)/DxE[i] +
        //      pow(h[2*BINS+4-i]-DxE[2*BINS+4-i],2)/DxE[2*BINS+4-i] ;
        //    }
        chi2x = chi2x / (2.0 * BINS + 1);
        chi2xr = chi2x; // chi2xr = chi2xr/(2.0*BINS-1) ;
    }
    for (int i = 0; i <= 2 * (BINS - BORDES); i++)
    {
        chi2p += pow(g[i + BORDES] - DpE[i + BORDES], 2) / DpE[i + BORDES];
    }
    for (int i = 0; i < (BINS - BORDES); i++)
    {
        chiIp += pow(g[i + BORDES] - g[2 * BINS - BORDES - i], 2) / DpE[i + BORDES];
        chiPp += pow(g[i + BORDES] + g[2 * BINS - BORDES - i] - 2.0 * DpE[i + BORDES], 2) / DpE[i + BORDES];
    }
    for (int i = 2; i < BINS + 1; i++)
    {
        chiIx += pow(h[i] - h[2 * BINS + 4 - i], 2) / DxE[i];
        chiPx += pow(h[i] + h[2 * BINS + 4 - i] - 2.0 * DxE[i], 2) / DxE[i];
    }
    chiIx = chiIx / (2.0 * BINS);
    chiPx = chiPx / (2.0 * BINS);
    chi2p = chi2p / (2.0 * (BINS - BORDES) + 1);
    chiIp = chiIp / (2.0 * (BINS - BORDES));
    chiPp = chiPp / (2.0 * (BINS - BORDES));
    //  printf("chi2x =%9.6f   chi2p =%9.6f   chiIp =%9.6f   chiPp =%9.6f\n",
    //          chi2x, chi2p, chiIp, chiPp);
    FILE *hist = fopen(filename, "w");
    fprintf(hist,
            "#   x    poblacion       p      poblacion    chi2x =%9.6f  chi2xr "
            "=%9.6f  chiIx =%9.6f  chiPx =%9.6f  chi2p =%9.6f  chiIp =%9.6f  "
            "chiPp =%9.6f\n",
            chi2x, chi2xr, chiIx, chiPx, chi2p, chiIp, chiPp);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", -0.502, h[0], -3.0e-23, g[0]);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", -0.501, h[1], -3.0e-23, g[0]);
    for (int i = 0; i <= BINS << 1; i++)
    { // BINS << 1 (shift-izq) equivale a 2*BINS
        fprintf(hist, "%8.5f %6d %24.12E %6d\n", (0.5 * i / BINS - 0.5), h[i + 2], (3.0e-23 * i / BINS - 3.0e-23),
                g[i]);
    }
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", 0.501, h[2 * BINS + 3], 3.0e-23, g[2 * BINS]);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", 0.502, h[2 * BINS + 4], 3.0e-23, g[2 * BINS]);
    //      OJO: DESCOMENTAR      //
    //  for (int i = 0; i <= (BINS+2) << 1; i++)
    //    for (int j = 0; j <= BINS << 1; j++)
    //      fprintf(hist, "%6d", hg[(2*BINS+1)*i+j]); fprintf(hist,"\n");
    fclose(hist);

    memset(h, 0, (2 * BINS + 5) * sizeof(int));
    memset(g, 0, (2 * BINS + 1) * sizeof(int));
    memset(hg, 0, (2 * BINS + 5) * (2 * BINS + 1) * sizeof(int));

    return 0; // avisa que se cumplió la condición sobre los chi2
}

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
        sem_wait(&iter_sem); // semaforo que señala que un hilo queda reservado
                             // para ejecución

        iter_in_range(steps[j], s, e, x, p, DT, M, alfa, pmin075,
                      pmax075); // avanza steps[j] pasos en el rango de partículas [s, e)

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
