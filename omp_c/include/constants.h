#ifndef CONSTANTS_H
#define CONSTANTS_H

#define BORDES 237 // para NPART = 2^21 (+100 cuentas)
#define PI 3.14159265358979323846
#define epsmax2M 9.0E-46 // E maxima
#define DEmax2M                                                                                                        \
    6.0e-50              // mas grande para pmax(3.5964e-48: 1 canal de p menos que el max)
                         // OJO: ahora DEmax2M nos da 1.2e-50 (8/2/24) 6.296208e-49
                         // 2 * 5.24684E-24 * 6.0e-26
#define epsmin2M 9.0E-52 // 2m * E minima = pmin^2

#endif