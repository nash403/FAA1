#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP2 FAA 2015-2016
# Auteur: Honoré NINTUNZE


import matplotlib.pyplot as plt
import numpy as np

from utils import *

### Données pour le moindre carré ###
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

### Apprentissage ###

# mesures de perf du TP1
mesAbs = mesureAbsolue(matrice_temps, positions, droite) # Erreur absolue moyenne
mesJL1 = mesureNormale1(matrice_temps, positions, droite) # Erreur-type
mesJL2 = mesureNormale2(matrice_temps, positions, droite) # Erreur quadratique moyenne
mesJInf = mesureInfinie(matrice_temps, positions, droite) # Plus grand écart

# calcul du moindre carre
mc = moindreCarre(matrice_temps,positions)
droite_mc = np.array(mc,float)

# mesures de perf avec moindre carre
mesAbs_mc = mesureAbsolue(matrice_temps, positions, droite_mc) # Erreur absolue moyenne
mesJL1_mc = mesureNormale1(matrice_temps, positions, droite_mc) # Erreur-type
mesJL2_mc = mesureNormale2(matrice_temps, positions, droite_mc) # Erreur quadratique moyenne
mesJInf_mc = mesureInfinie(matrice_temps, positions, droite_mc) # Plus grand écart

### PRINT RESULTATS ###

print "_________ Mesures de performances avec moindres carres __________________________________________"
print "droite moindres carres=", droite_mc
print "jAbs_mc =", mesAbs_mc # Erreur absolue moyenne
print "jL1_mc =", mesJL1_mc # Erreur-type
print "jL2_mc (MSE) =", mesJL2_mc # Erreur quadratique moyenne
print "jLInf_mc =", mesJInf_mc # Plus grand écart
print "_________ Différence entre les mesures de performances sans et avec moindres carres __________________________________________"
print "Diff -> jAbs - jAbs_mc =", (mesAbs - mesAbs_mc)
print "Diff -> jL1 - jL1_mc =", (mesJL1 - mesJL1_mc)
print "Diff -> jL2 - jL2_mc =", (mesJL2 - mesJL2_mc)
print "Diff -> jLInf - jLInf_mc =", -(mesJInf - mesJInf_mc)


# Plots
ordonneeMC = mc[1] * abscisse + mc[0]

plt.close('all')
plt.xlabel('Temps')
plt.ylabel('Positions')
plt.title('Tp2 FAA - Mesures de Perf avec moindre carre')
plt.plot(temps, positions,'.')
p1, = plt.plot(abscisse,ordonnee,'g')
p2, = plt.plot(abscisse,ordonneeMC,'r')
plt.legend([p1, p2], ["Theorique", "Moindres carres"])
plt.savefig("moindresCarres_TP2.png")
plt.show()
