# Simulación
Gas 1D con p discretos, SIN ruido en L del recipiente + ruido en p al rebotar con la pared.
A cada E_j le corresponde una Ej' sorteada con una distribución uniforme en [Ej-DeltaE/2, Ej+DeltaE/2],
DeltaE = alfa((p^0.75 - pmin^0.75)(pmax^0.75-p^0.75))^4+d

En la distribución en x permitimos que los bordes no sean filososÑ agregamos 2 canales
en cada extremo, redefiniendo los chi2x

NO: Al cabo de un cicle, antes de grabar el dmp, pasamos el 50% de la energía de (14) particulas con |p| > 3sigma a 14 particulas con 0.15sigma < |p| < .9 sigma

## Instrucciones para ejecutar el programa
Este programa esta diseñado para ser compilado con make.

### Compilación del programa
1. Abre una terminal
2. Clona el repositorio
```Bash
git clone git@github.com:ignabelitzky/gcas.git
```
3. Posicionate dentro del directorio clonado
```Bash
cd gcas
```
4. Ejecuta el siguiente comando para compilar el programa
```Bash
make
```
Esto generará el ejecutable (`main`) del programa.

### Ejecución del progama
Una vez que el programa ha sido compilado correctamente, puedes ejecutarlo siguiendo estos pasosÑ
1. Asegúrate de tener el archivo `datos.in` en el mismo directorio que el ejecutable del programa.
2. Ejecuta el programa previamente compilado:
```Bash
./main
```

**Nota:** El archivo .dmp guardan la info de 2^21 partículas.