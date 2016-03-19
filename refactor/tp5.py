#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP5 FAA 2015-2016
# Auteur: Honoré NINTUNZE


import matplotlib.pyplot as plt
import numpy as np
import math
import random as rand

from utils import *

### Données pour la Regression logistique (Classification) ###

# lecture des 2 dimensions pour les données
res_taille_f,res_poids_f = lire_fichierClasse("data/taillepoids_f.txt")
res_taille_h,res_poids_h = lire_fichierClasse("data/taillepoids_h.txt")

# taille et poids des femmes
taille_f = np.array(res_taille_f,float)
poids_f = np.array(res_poids_f,float)
# taille et poids des hommes
taille_h = np.array(res_taille_h,float)
poids_h = np.array(res_poids_h,float)

nombre_hommes = len(taille_h)
nombre_femmes = len(taille_f)

# classes, 0 femmes et 1 hommes
classe0 = np.zeros(nombre_femmes)
classe1 = np.ones(nombre_hommes)

matrice_classification = np.ones((2,(nombre_hommes+nombre_femmes)))
matrice_classification[1,:] = np.concatenate((taille_f,taille_h))
classesHF = np.concatenate((classe0,classe1))

SEUIL = 0.5

### Apprentissage ###

tetaDepart = np.array([3,2],float)#np.array([rand.randint(0,1),rand.randint(0,1)],float)

# Descante de gradiant pour la classification
dgc_droite, dgc_tps, dgc_mesures  = descenteGradiantSigmoide(matrice_classification, classesHF, tetaDepart, N=(nombre_hommes+nombre_femmes))

# Mesure pour la descante de gradiant sigmoide
# mesJL2_dgc = mesureNormale2(matrice_classification, classesHF, dgc_droite)

tetas = plusieurs_teta_random(10)
learning = descenteSigmoideMultiple(matrice_classification,classesHF,tetas,(nombre_hommes+nombre_femmes))


### PRINT RESULTATS ###

print "_________ Classification __________________________________________"
print "descante gradiant sigmoide =", dgc_droite

# Plots

plt.ylim(-1,2)

# affichage des classes pour la taille des hommes (classe1) et des femmes (classe0)
plt.plot(taille_f,classe0,'o',label="Femmes")
plt.plot(taille_h,classe1,'o',label="Hommes")

# pour tracer des lignes verticales
for tt in learning :
    A = tt[1]
    b = tt[0]
    Y = (1. / float(A * math.log(1 - SEUIL) - math.log(SEUIL) + b))
    plt.axvline(Y)

plt.savefig("classes_TP5.png")

plt.legend()
plt.show()
