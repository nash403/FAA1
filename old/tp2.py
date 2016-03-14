#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP FAA 2016
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
class0 = np.zeros(nombre_femmes)
class1 = np.ones(nombre_hommes)


### UTILS ###
def get_phi(degre,x):
    mx = np.zeros((degre+1,len(x)))
    mx[0,:] = np.ones(len(x))
    for i in range(1,degre+1):
        mx[i,:] = np.array([float(v) ** i for v in x],float)
    return mx

def get_formule(teta,matrix):
    return np.dot(matrix.T,teta)

def alpha(temps,val=1.):
    A = val
    B = A
    C = 4000.
    return A / (B + C * float(temps))

def sigmoid(x):
    # print "sig", x
    return 1 / (1 + math.exp(-x))

def sigmoidVect(z):
    # print "sigVect", z
    res = np.array([])
    for i in range(len(z)):
        res = np.append(res,sigmoid(z[i]))
    return res

def sigmoidMatrix(z):
    pass

def risqueEmpirique(x,y,teta,N):
    # print "risque", x, y
    sigma = sigmoidVect(np.dot(x.T,teta))
    return 1./float(N) * np.sum(-y * np.log(sigma) - (1.0 - y) * np.log(1.0 - sigma))

### MESURES ###
def jAbs(x1,y,teta,N=100) :
    vecteur = y - np.dot(x1.T,teta)
    return np.sum(np.absolute(vecteur))/N

def jl2(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    return np.dot(vecteur.T, vecteur)/N

def jl1(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    interm = np.dot(vecteur.T, vecteur)
    return math.sqrt(interm)/N

def jInf(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    return np.amax(np.absolute(vecteur))

def moindreCarre(x1,y): #jGradiant
    return np.dot(np.linalg.inv(np.dot(x1,x1.T)), np.dot(x1,y))

def descanteGradiant(x1, y, teta, tps, epsi=0.0000001, N=100):
    resTeta = teta
    resTps = tps
    while(True):
        gradteta = np.dot(x1,(y - np.dot(x1.T,resTeta)))
        tetaplus = resTeta + np.dot(np.dot(alpha(tps),gradteta),1./float(N))

        if math.fabstetaDescante(jl2(x1,y,tetaplus) - jl2(x1,y,resTeta)) <= epsi:
            return resTeta
        else:
            resTeta = tetaplus
            resTps += 1

def descanteGradiantStochastique(x1, y, teta, tps, epsi=0.0000001, N=100):
    resTeta = teta
    resTps = tps
    while(True):
        idx = rand.randint(0,N-1)
        vectX = np.array([x1[0][idx],x1[1][idx]])
        gradteta = np.dot(vectX,(y[idx] - np.dot(resTeta.T,vectX)))
        tetaplus = resTeta + np.dot(np.dot(alpha(tps),gradteta),1./float(N))

        if math.fabs(jl2(vectX,y[idx],tetaplus) - jl2(vectX,y[idx],resTeta)) <= epsi:
            return resTeta
        else:
            resTeta = tetaplus
            resTps += 1

def descanteGradiantSigmoide(x, y, teta, tps, epsi=0.0000001, N=100):
    resTeta = teta
    resTps = tps
    while(True):
        # print "avant", x, resTeta, np.dot(x.T,resTeta)
        gradteta = np.dot(x,(y - sigmoidVect(np.dot(x.T,resTeta))))
        tetaplus = resTeta + np.dot(np.dot(alpha(tps),gradteta),1./float(N))

        if risqueEmpirique(x,y,resTeta,N) - risqueEmpirique(x,y,tetaplus,N) <= epsi:
            return resTeta
        else:
            resTeta = tetaplus
            resTps += 1

##
# mcarres = moindreCarre(x1, p)
# print "ABS = ", jAbs(x1,p,teta)
# print "JL1 = ", jl1(x1,p,teta)
# print "JL2 = ", jl2(x1,p,teta)
# print "JLINF = ", jInf(x1,p,teta)
# print "Grad = ", mcarres

##
#pour la descante de gradiant, np.array([b,a],float)
tetaDescante = np.array([3,2],float)
# print "descante gradiant =", descanteGradiant(x1,p,teta,1)
# print "descante gradiant stochastique = ", descanteGradiantStochastique(x1,p,teta,1)
# pour récup les valeurs du gradiant
# c = mcarres.item(1)
# d = mcarres.item(0)
#
# u = c*x + d

##
# plt.plot(t,p,'.')
# # l1, =
# plt.plot(x,y,label="Theorique")
# # l2, =
# plt.plot(x,u,label="Moindres carres")
# # plt.legend(handles=[l1,l2])
# plt.xlabel('t (s)')
# plt.ylabel('p (m)')
# plt.title('TP1 FAA')

# teta0 = moindreCarre(mx0,y0);
# print "moindre carres x0 ->", teta0
# plt.plot(x0,y0,'ro')
# plt.plot(x0,get_formule(teta0,mx0),'bo')

# mx = get_phi(2,x1)
# teta1 = moindreCarre(mx,y1)
# print "moindre carres x1 ->", teta1
## plt.plot(x1,teta[2]*mx[2] + teta[1]*mx[1] + teta[0]*mx[0],'g+')
# plt.plot(x1,get_formule(teta1,mx),'g+')
# plt.plot(x1,y1,'r+')

# affichage des classes pour la taille des hommes (class1) et des femmes (class0)
# plt.plot(taille_f,class0,'ro')
# plt.plot(taille_h,class1,'bo')
# plt.ylim(-1,2)


# sigmoid
matrix = np.ones((2,(nombre_hommes+nombre_femmes)))
matrix[1,:] = np.concatenate((taille_f,taille_h))
classesHF = np.concatenate((class0,class1))
print "matrix", matrix
print"classesHF", classesHF
# print "data", matrix.T, tetaDescante, np.dot(matrix.T,tetaDescante)
print "sig", sigmoidVect(np.dot(matrix.T,tetaDescante))
print "descante", descanteGradiantSigmoide(matrix,classesHF,tetaDescante,1,N=(nombre_hommes+nombre_femmes))

# plt.show()
