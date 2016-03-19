#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP3 FAA 2015-2016
# Auteur: Honoré NINTUNZE


import matplotlib.pyplot as plt
import numpy as np

from utils import *


### Données pour la descente de gradiant ###

temps = np.array(lire_fichier("data/t.txt"),float)
positions = np.array(lire_fichier("data/p.txt"),float)

N = len(temps)

# Droite donnée (A * x + b)
A = 2
b = 3

droite = np.array([b,A],float)
droiteSto = np.array([rand.random(),rand.random()],float)

matrice_temps = np.zeros((2,N))
matrice_temps[1,:] = temps
matrice_temps[0,:] = np.ones(N)

### Apprentissage ###

# Descentes de gradiant
dg_droite, dg_tps, dg_mesures = descenteGradiant(matrice_temps,positions,droite)
dgs_droite, dgs_tps, dgs_mesures = descenteGradiantStochastique(matrice_temps,positions, droiteSto)

# Mesures pour la descente de gradiant
mesJL2_dg = mesureNormale2(matrice_temps, positions, dg_droite)
mesJL2_dgs = mesureNormale2(matrice_temps, positions, dgs_droite)

### PRINT RESULTATS ###

print "_________ Descentes de gradiant __________________________________________"
print "Descente gradiant =", dg_droite
print "Descente gradiant stochastique = ", dgs_droite
print "jL2_dg (MSE) =", mesJL2_dg # Erreur quadratique moyenne
print "jL2_dgs (MSE) =", mesJL2_dgs # Erreur quadratique moyenne


# Plots
def plot_descente_simple():
    plt.close('all')
    plt.xlabel('Temps')
    plt.ylabel('Mesures d\'erreur')
    plt.title('Tp3 FAA - Descente de gradiant')
    plt.plot(dg_tps,dg_mesures,'b')
    plt.savefig("descenteSimple_TP3.png")
    plt.show()

def plot_descente_simple_variation():
    plt.close('all')
    plt.xlabel('Temps')
    plt.ylabel('Mesures d\'erreur')
    plt.title('Tp3 FAA - Descente de gradiant, variation d\'alpha')
    for i in range(1,11):
        _, xs, ys = descenteGradiant(matrice_temps,positions,droite,facteur=i)
        plt.plot(xs,ys,label="Alpha 1/(1 + "+str(i*100)+" * temps)")
    plt.savefig("descenteSimpleVariation_TP3.png")
    plt.legend()
    plt.show()

def plot_descente_stochastique():
    plt.close('all')
    plt.xlabel('Temps')
    plt.ylabel('Mesures d\'erreur')
    plt.title('Tp3 FAA - Descente de gradiant stochastique')
    plt.plot(dgs_tps,dgs_mesures,'r')
    plt.ylim(0,2)
    plt.savefig("descenteStochastique_TP3.png")
    plt.show()

def plot_descente_stochastique_variation():
    plt.close('all')
    plt.xlabel('Temps')
    plt.ylabel('Mesures d\'erreur')
    plt.title('Tp3 FAA - Descente de gradiant, variation d\'alpha')
    for i in range(1,4):
        _, xs, ys = descenteGradiantStochastique(matrice_temps,positions,droiteSto,facteur=i)
        plt.plot(xs,ys,label="Alpha 1/(1 + "+str(i*100)+" * temps)")
    plt.savefig("descenteStochastiqueVariation_TP3.png")
    plt.legend()
    plt.show()

plot_descente_simple()
plot_descente_stochastique()
plot_descente_simple_variation()
plot_descente_stochastique_variation()
