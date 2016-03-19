#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP FAA 2015-2016
# Auteur: Honoré NINTUNZE

import matplotlib.pyplot as plt
import numpy as np
import math
import random as rand


### LECTURE DATA ###
def lire_fichier(file):
    fichier = open(file,'r')
    return fichier.read().split()

def lire_fichierClasse(file):
    fichier = open(file,'r')
    (d1,d2) = ([],[])
    for line in fichier:
        (taille,poids) = line.split()
        d1.append(float(taille))
        d2.append(float(poids))
    return (d1,d2)


### UTILS ###

# Mesures de performances

# Erreur absolue moyenne (jAbs)
def mesureAbsolue(x,y,teta,N=100) :
    vecteur = y - np.dot(x.T,teta)
    return np.sum(np.absolute(vecteur))/N

# Erreur quadratique moyenne (MSE) (jL2)
def mesureNormale2(x,y,teta,N=100):
    vecteur = y - np.dot(x.T,teta)
    return np.dot(vecteur.T, vecteur)/N

# Erreur-type (jL1)
def mesureNormale1(x,y,teta,N=100):
    vecteur = y - np.dot(x.T,teta)
    interm = np.dot(vecteur.T, vecteur)
    return math.sqrt(interm)/N

# Plus grand écart (jLInf)
def mesureInfinie(x,y,teta,N=100):
    vecteur = y - np.dot(x.T,teta)
    return np.amax(np.absolute(vecteur))

## Moindres Carres ##

# Cacul des Moindres Carres, on veut minimiser la MSE
def moindreCarre(x,y):
    return np.dot(np.linalg.inv(np.dot(x,x.T)), np.dot(x,y))

## Descente de Gradiant ##

# Calcul de la variation du pas d'apprentissage pour la descente de gradiant
def alpha(temps,val=1.,facteur=4):
    A = val
    B = A
    C = float(facteur)*100.
    return A / (B + C * float(temps))

# Simple descente de gradiant
def descenteGradiant(x, y, teta, N=100, epsi=0.00000001, tps=1,facteur=4):
    resTeta = teta
    resTps = tps

    l_tps, l_teta  = [resTps],[mesureNormale2(x,y,resTeta)]
    go_on = True

    while(go_on):
        gradteta = np.dot(x,(y - np.dot(x.T,resTeta)))
        tetaplus = resTeta + np.dot(np.dot(alpha(tps,facteur=facteur),gradteta),1./float(N))

        if math.fabs(mesureNormale2(x,y,tetaplus) - mesureNormale2(x,y,resTeta)) <= epsi:
            go_on = False
        else:
            resTeta = tetaplus
            resTps += 1

        l_tps.append(resTps)
        l_teta.append(mesureNormale2(x,y,resTeta))

    return resTeta, l_tps, l_teta

# Descente de gradiant stochastique
def descenteGradiantStochastique(x, y, teta, N=100, epsi=0.00000001, tps=1, facteur=1):
    resTeta = teta
    resTps = tps

    l_tps, l_teta  = [],[]
    go_on = True

    while(go_on):
        idx = rand.randint(0,N-1)
        vectX = np.array([x[0][idx],x[1][idx]])
        gradteta = np.dot(vectX,(y[idx] - np.dot(resTeta.T,vectX)))
        tetaplus = resTeta + np.dot(np.dot(alpha(tps,facteur=facteur),gradteta),1./float(N))

        if math.fabs(mesureNormale2(vectX,y[idx],tetaplus) - mesureNormale2(vectX,y[idx],resTeta)) <= epsi:
            go_on = False
        else:
            resTeta = tetaplus
            resTps += 1

        l_tps.append(resTps)
        l_teta.append(mesureNormale2(vectX,y[idx],resTeta))

    return resTeta, l_tps, l_teta

# Classification

# Calcul de la matrice Phi, matrice à la puissance <degre>
def get_phi(degre,x):
    mx = np.zeros((degre+1,len(x)))
    mx[0,:] = np.ones(len(x))
    for i in range(1,degre+1):
        mx[i,:] = np.array([float(v) ** i for v in x],float)
    return mx

# Fonction pour récupérer
def get_formule(teta,matrix):
    return np.dot(matrix.T,teta)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoidVect(z):
    res = np.array([])
    for i in range(len(z)):
        res = np.append(res,sigmoid(z[i]))
    return res

def risqueEmpirique(x,y,teta,N):
    sigma = sigmoidVect(np.dot(x.T,teta))
    return 1./float(N) * np.sum(-y * np.log(sigma) - (1.0 - y) * np.log(1.0 - sigma))

def descenteGradiantSigmoide(x, y, teta, N, epsi=0.0000001, tps=1):
    resTeta = teta
    resTps = tps

    l_tps, l_teta  = [resTps],[resTeta]
    go_on = True

    while(go_on):
        gradteta = np.dot(x,(y - sigmoidVect(np.dot(x.T,resTeta))))
        tetaplus = resTeta + np.dot(np.dot(alpha(tps),gradteta),1./float(N))

        if risqueEmpirique(x,y,resTeta,N) - risqueEmpirique(x,y,tetaplus,N) <= epsi:
            go_on = False
        else:
            resTeta = tetaplus
            resTps += 1

        l_tps.append(resTps)
        l_teta.append(risqueEmpirique(x,y,resTeta,N))

    return resTeta, l_tps, l_teta

def plusieurs_teta_random(n) :
    res = []
    for i in range(n) :
        res.append(np.array([rand.random(),rand.random()],float))
    return res

def descenteSigmoideMultiple(matrix,y,tetas,N,epsi=0.0000001, tps=1):
    res = []
    for teta in tetas :
        r0,r1,r2 = descenteGradiantSigmoide(matrix,y,teta,N,epsi,tps)
        res.append(r0)
    return res
