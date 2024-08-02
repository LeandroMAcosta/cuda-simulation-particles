#include "../include/utils.h"

static double d_rand()
{
    srand(time(NULL));
    return (double)rand() / (double)RAND_MAX;
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
#pragma omp parallel for reduction(+ : sumEnergy) schedule(static)
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

int make_hist(int *h, int *g, int *hg, double *DxE, double *DpE, const char *filename, int BINS)
{
    double chi2x = 0.0, chi2xr = 0.0, chi2p = 0.0, chiIp = 0.0, chiPp = 0.0, chiIx = 0.0, chiPx = 0.0;

    if (strcmp(filename, "X0000000.dat") == 0)
    {
#pragma omp parallel for reduction(+ : chi2x) schedule(static)
        for (int i = BINS + 1; i <= 2 * BINS; i++)
        {
            chi2x += pow(h[i] - 2 * DxE[i], 2) / (2 * DxE[i]);
        }
        chi2x = (chi2x + pow(h[BINS] - DxE[BINS], 2) / DxE[BINS]) / (BINS + 1);
    }
    else
    {
#pragma omp parallel for reduction(+ : chi2x) schedule(static)
        for (int i = 2; i <= 2 * (BINS + 1); i++)
        {
            chi2x += pow(h[i] - DxE[i], 2) / DxE[i];
        }
        chi2x = chi2x / (2.0 * BINS + 1);
        chi2xr = chi2x; // chi2xr = chi2x reducido
    }
#pragma omp parallel for reduction(+ : chi2p) schedule(static)
    for (int i = 0; i <= 2 * (BINS - BORDES); i++)
    {
        chi2p += pow(g[i + BORDES] - DpE[i + BORDES], 2) / DpE[i + BORDES];
    }
#pragma omp parallel for reduction(+ : chiIp, chiPp) schedule(static)
    for (int i = 0; i < (BINS - BORDES); i++)
    {
        chiIp += pow(g[i + BORDES] - g[2 * BINS - BORDES - i], 2) / DpE[i + BORDES];
        chiPp += pow(g[i + BORDES] + g[2 * BINS - BORDES - i] - 2.0 * DpE[i + BORDES], 2) / DpE[i + BORDES];
    }
#pragma omp parallel for reduction(+ : chiIx, chiPx) schedule(static)
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

    FILE *hist = fopen(filename, "w");
    fprintf(hist,
            "#   x    poblacion       p      poblacion    chi2x =%9.6f  chi2xr "
            "=%9.6f  chiIx =%9.6f  chiPx =%9.6f  chi2p =%9.6f  chiIp =%9.6f  "
            "chiPp =%9.6f\n",
            chi2x, chi2xr, chiIx, chiPx, chi2p, chiIp, chiPp);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", -0.502, h[0], -3.0e-23, g[0]);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", -0.501, h[1], -3.0e-23, g[0]);
    for (int i = 0; i <= BINS << 1; i++)
    {
        fprintf(hist, "%8.5f %6d %24.12E %6d\n", (0.5 * i / BINS - 0.5), h[i + 2], (3.0e-23 * i / BINS - 3.0e-23),
                g[i]);
    }
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", 0.501, h[2 * BINS + 3], 3.0e-23, g[2 * BINS]);
    fprintf(hist, "%8.5f %6d %24.12E %6d\n", 0.502, h[2 * BINS + 4], 3.0e-23, g[2 * BINS]);

    fclose(hist);

    memset(h, 0, (2 * BINS + 5) * sizeof(int));
    memset(g, 0, (2 * BINS + 1) * sizeof(int));
    memset(hg, 0, (2 * BINS + 5) * (2 * BINS + 1) * sizeof(int));

    return 0; // avisa que se cumplió la condición sobre los chi2
}

bool check_memory_allocations(double *x, double *p, double *DxE, double *DpE, int *h, int *g, int *hg)
{
    bool result = true;
    if (x == NULL)
    {
        printf("Error: No se pudo reservar memoria para x\n");
        result = false;
    }
    if (p == NULL)
    {
        free(x);
        printf("Error: No se pudo reservar memoria para p\n");
        result = false;
    }
    if (DxE == NULL)
    {
        free(x);
        free(p);
        printf("Error: No se pudo reservar memoria para DxE\n");
        result = false;
    }
    if (DpE == NULL)
    {
        free(x);
        free(p);
        free(DxE);
        printf("Error: No se pudo reservar memoria para DpE\n");
        result = false;
    }
    if (h == NULL)
    {
        free(x);
        free(p);
        free(DxE);
        free(DpE);
        printf("Error: No se pudo reservar memoria para h\n");
        result = false;
    }
    if (g == NULL)
    {
        free(x);
        free(p);
        free(DxE);
        free(DpE);
        free(h);
        printf("Error: No se pudo reservar memoria para g\n");
        result = false;
    }
    if (hg == NULL)
    {
        free(x);
        free(p);
        free(DxE);
        free(DpE);
        free(h);
        free(g);
        printf("Error: No se pudo reservar memoria para hg\n");
        result = false;
    }
    return result;
}
