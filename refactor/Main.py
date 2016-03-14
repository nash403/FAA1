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

### DATA ###

### Données pour la Regression linéaire ###
temps = np.array(lire_fichier("data/t.txt"),float)
positions = np.array(lire_fichier("data/p.txt"),float)

N = len(temps)

# Droite donnée (A * x + b)
A = 2
b = 3

droite = np.array([b,A],float)

matrice_temps = np.zeros((2,N))
matrice_temps[1,:] = temps
matrice_temps[0,:] = np.ones(N)

# print "### DATA PRINT Regression linéaire ###"
# print "N=", N
# print "temps=", temps
# print "pos=", positions
# print "matrix=", matrice_temps

### Données pour la Regression logistique (Classification) ###

x0 = np.array(lire_fichier("data/x0.txt"),float)
x1 = np.array(lire_fichier("data/x1.txt"),float)
x2 = np.array(lire_fichier("data/x2.txt"),float)

y0 = np.array(lire_fichier("data/y0.txt"),float)
y1 = np.array(lire_fichier("data/y1.txt"),float)
y2 = np.array(lire_fichier("data/y2.txt"),float)

lenx0 = len(x0)
lenx1 = len(x1)
lenx2 = len(x2)

mx0 = np.zeros((2,lenx0))
mx1 = np.zeros((2,lenx1))
mx2 = np.zeros((2,lenx2))

mx0[1,:] = x0
mx1[1,:] = x1
mx2[1,:] = x2

mx0[0,:] = np.ones(lenx0)
mx1[0,:] = np.ones(lenx1)
mx2[0,:] = np.ones(lenx2)

#lecture des 2 dimensions
res_taille_f,res_poids_f = lire_fichierClasse("data/taillepoids_f.txt")
res_taille_h,res_poids_h = lire_fichierClasse("data/taillepoids_h.txt")

#taille et poids des femmes
taille_f = np.array(res_taille_f,float)
poids_f = np.array(res_poids_f,float)
#taille et poids des hommes
taille_h = np.array(res_taille_h,float)
poids_h = np.array(res_poids_h,float)

nombre_hommes = len(taille_h)
nombre_femmes = len(taille_f)

#classes, 0 femmes et 1 hommes
classe0 = np.zeros(nombre_femmes)
classe1 = np.ones(nombre_hommes)

matrice_classification = np.ones((2,(nombre_hommes+nombre_femmes)))
matrice_classification[1,:] = np.concatenate((taille_f,taille_h))
classesHF = np.concatenate((classe0,classe1))

# print "### DATA PRINT Classification (Régression logistique) ###"
# print "matrice_classification", matrix
# print "classes Homme/Femme", classesHF


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

## Descante de Gradiant ##

# Calcul de la variation du pas d'apprentissage pour la descante de gradiant
def alpha(temps,val=1.):
    A = val
    B = A
    C = 4000.
    return A / (B + C * float(temps))

# Simple descante de gradiant
def descanteGradiant(x, y, teta, N=100, epsi=0.0000001, tps=1):
    resTeta = teta
    resTps = tps

    l_tps, l_teta  = [resTps],[resTeta]
    go_on = True

    while(go_on):
        gradteta = np.dot(x,(y - np.dot(x.T,resTeta)))
        tetaplus = resTeta + np.dot(np.dot(alpha(tps),gradteta),1./float(N))

        if math.fabs(mesureNormale2(x,y,tetaplus) - mesureNormale2(x,y,resTeta)) <= epsi:
            go_on = False
        else:
            resTeta = tetaplus
            resTps += 1

        l_tps.append(resTps)
        l_teta.append(mesureNormale2(x,y,resTeta))

    return resTeta, l_tps, l_teta

# Descante de gradiant stochastique
def descanteGradiantStochastique(x, y, teta, N=100, epsi=0.0000001, tps=1):
    resTeta = teta
    resTps = tps

    l_tps, l_teta  = [resTps],[resTeta]
    go_on = True

    while(go_on):
        idx = rand.randint(0,N-1)
        vectX = np.array([x[0][idx],x[1][idx]])
        gradteta = np.dot(vectX,(y[idx] - np.dot(resTeta.T,vectX)))
        tetaplus = resTeta + np.dot(np.dot(alpha(tps),gradteta),1./float(N))

        if math.fabs(mesureNormale2(vectX,y[idx],tetaplus) - mesureNormale2(vectX,y[idx],resTeta)) <= epsi:
            go_on = False
        else:
            resTeta = tetaplus
            resTps += 1

        l_tps.append(resTps)
        l_teta.append(mesureNormale2(vectX,y[idx],resTeta))

    return resTeta, l_tps, l_teta

# Classification

# Calcul de la matrice Phi
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

def descanteGradiantSigmoide(x, y, teta, N, epsi=0.0000001, tps=1):
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

### Apprentissage ###

# mesures de perf
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

# descantes de gradiant
dg_droite, dg_tps, dg_mesures = descanteGradiant(matrice_temps,positions,droite)
dgs_droite, dgs_tps, dgs_mesures = descanteGradiantStochastique(matrice_temps,positions,droite)

# Mesures pour la descante de gradiant
mesJL2_dg = mesureNormale2(matrice_temps, positions, dg_droite)
mesJL2_dgs = mesureNormale2(matrice_temps, positions, dgs_droite)

# Descante de gradiant pour la classification
dgc_droite, dgc_tps, dgc_mesures  = descanteGradiantSigmoide(matrice_classification, classesHF, droite, N=(nombre_hommes+nombre_femmes))

# Mesure pour la descante de gradiant sigmoide
mesJL2_dgc = mesureNormale2(matrice_classification, classesHF, dgc_droite)

### PRINT RESULTATS ###

print "_________ Mesures de performances __________________________________________"
print "droite donnée=", droite
print "jAbs=", mesAbs # Erreur absolue moyenne
print "jL1=", mesJL1 # Erreur-type
print "jL2(MSE)=", mesJL2 # Erreur quadratique moyenne
print "jLInf=", mesJInf # Plus grand écart
print "_________ Mesures de performances avec moindres carres __________________________________________"
print "droite moindres carres=", droite_mc
print "jAbs_mc=", mesAbs_mc # Erreur absolue moyenne
print "jL1_mc=", mesJL1_mc # Erreur-type
print "jL2_mc(MSE)=", mesJL2_mc # Erreur quadratique moyenne
print "jLInf_mc=", mesJInf_mc # Plus grand écart
print "_________ Différence entre les mesures de performances sans et avec moindres carres __________________________________________"
print "Diff -> jAbs - jAbs_mc=", (mesAbs - mesAbs_mc)
print "Diff -> jL1 - jL1_mc=", (mesJL1 - mesJL1_mc)
print "Diff -> jL2 - jL2_mc=", (mesJL2 - mesJL2_mc)
print "Diff -> jLInf - jLInf_mc=", (mesJInf - mesJInf_mc)
print "_________ Descantes de gradiant __________________________________________"
print "descante gradiant =", dg_droite
print "descante gradiant stochastique = ", dgs_droite
print "jL2_dg(MSE)=", mesJL2_dg # Erreur quadratique moyenne
print "jL2_dgs(MSE)=", mesJL2_dgs # Erreur quadratique moyenne
print "_________ Classification __________________________________________"
print "descante gradiant sigmoide =", dgc_droite
print "jL2_dgc(MSE)=", mesJL2_dgc # Erreur quadratique moyenne

### Plot des données ###
def plot_descantes_gradiant():
    plt.close('all')
    plt.plot(dg_tps,dg_mesures,'r.')
    plt.title('Descante de Gradiant')
    plt.xlabel('Temps')
    plt.ylabel('Mesures')
    plt.show()
