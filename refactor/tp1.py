#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP1 FAA 2015-2016
# Auteur: Honoré NINTUNZE


import matplotlib.pyplot as plt
import numpy as np

from utils import *

### Données pour la Regression linéaire ###
temps = np.array(lire_fichier("data/t.txt"),float)
positions = np.array(lire_fichier("data/p.txt"),float)

# Nombres de données
N = len(temps)

# Coefficients de la droite donnée (A * x + b)
A = 2
b = 3

# pour les plots
abscisse = np.linspace(4,16,N)
ordonnee = A * abscisse + b

# Droite donnée 2x + 3
droite = np.array([b,A],float)

# Matrice des temps
matrice_temps = np.zeros((2,N))
matrice_temps[1,:] = temps
matrice_temps[0,:] = np.ones(N)

print "test" #np.linspace(4,16,N)

### Apprentissage ###

# mesures de perf
mesAbs = mesureAbsolue(matrice_temps, positions, droite) # Erreur absolue moyenne
mesJL1 = mesureNormale1(matrice_temps, positions, droite) # Erreur-type
mesJL2 = mesureNormale2(matrice_temps, positions, droite) # Erreur quadratique moyenne
mesJInf = mesureInfinie(matrice_temps, positions, droite) # Plus grand écart

### PRINT RESULTATS ###

print "_________ Mesures de performances __________________________________________"
print "droite donnée =", droite
print "jAbs =", mesAbs # Erreur absolue moyenne
print "jL1 =", mesJL1 # Erreur-type
print "jL2 (MSE) =", mesJL2 # Erreur quadratique moyenne
print "jLInf =", mesJInf # Plus grand écart


# Plots
plt.close('all')
plt.xlabel('Temps')
plt.ylabel('Positions')
plt.title('Tp1 FAA - Mesures de Perf')
plt.plot(temps, positions,'.')
plt.plot(abscisse,ordonnee)
plt.savefig("mesuresPerf_TP1.png")
plt.show()
