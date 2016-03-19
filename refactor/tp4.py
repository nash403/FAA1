#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP4 FAA 2015-2016
# Auteur: Honoré NINTUNZE


import matplotlib.pyplot as plt
import numpy as np

from utils import *


### Données pour l'approxximation de fonction ###

x0 = np.array(lire_fichier("data/x0.txt"),float)
x1 = np.array(lire_fichier("data/x1.txt"),float)
x2 = np.array(lire_fichier("data/x2.txt"),float)

y0 = np.array(lire_fichier("data/y0.txt"),float)
y1 = np.array(lire_fichier("data/y1.txt"),float)
y2 = np.array(lire_fichier("data/y2.txt"),float)

lenx0 = len(x0)

mx0 = np.zeros((2,lenx0))

mx0[1,:] = x0

mx0[0,:] = np.ones(lenx0)

### Apprentissage ###

teta0 = moindreCarre(mx0,y0)

phi1 = get_phi(2,x1)
teta1 = moindreCarre(phi1,y1)

# Les données sont désordonnées, il faut les trier
x2.sort()
y2.sort()

phi2 = get_phi(10,x2)
teta2 = moindreCarre(phi2,y2)

### PRINT RESULTATS ###

print "_________ Approximation de fonctions __________________________________________"
print "Moindre carre pour les données (x0,y0) avec polynôme degré 1 =", teta0
print "Moindre carre pour les données (x1,y1) avec polynôme degré 2 =", teta1
print "Moindre carre pour les données (x2,y2) avec polynôme degré 10 =", teta2

# Plots

def plot_donnees0():
    plt.close('all')
    plt.plot(x0,get_formule(teta0,mx0),label="Approximation donnees (x0,y0)")
    plt.plot(x0,y0,'+',label='(x0,y0)')
    plt.savefig("approximation0_TP4.png")
    plt.legend()
    plt.show()

def plot_donnees1():
    plt.close('all')
    # La ligne suivante est équivalente à : plt.plot(x1,teta1[2]*phi1[2] + teta[1]*phi1[1] + teta[0]*phi1[0],'g.')
    plt.plot(x1,get_formule(teta1,phi1),label="Approximation donnees (x1,y1)")
    plt.plot(x1,y1,'+',label='(x1,y1)')
    plt.savefig("approximation1_TP4.png")
    plt.legend()
    plt.show()


def plot_donnees2():
    plt.close('all')
    plt.plot(x2,get_formule(teta2,phi2),label="Approximation donnees (x2,y2)")
    plt.plot(x2,y2,'+',label='(x2,y2)')
    plt.savefig("approximation2_TP4.png")
    plt.legend()
    plt.show()

plot_donnees0()
plot_donnees1()
plot_donnees2()
