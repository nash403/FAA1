#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP FAA 2015-2016
# Auteur: Honoré NINTUNZE

import matplotlib.pyplot as plt
import numpy as np
import math
import random as rand


# Droite à approximer b*x + a
a = 2
b = 3

e = 1

N = 100

x = np.linspace(4,15,N)
y = a*x + b
e = np.ones(N)

# alphaT = A / (B + C*t)


f1 = open("data/t.txt",'r')

f2 = open("data/p.txt",'r')
t = np.array(f1.read().split(),float)
p = np.array(f2.read().split(),float)

teta = np.array([b,a],float)

x1 = np.zeros((2,N))
x1[1,:] = t
x1[0,:] = e

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

def jGradiant(x1,y):
    return np.dot(np.linalg.inv(np.dot(x1,x1.T)), np.dot(x1,y))

# alpha est le pas d'apprentissage
def alpha(temps,val=1.):
    A = val
    B = A
    C = 4000.
    return A / (B + C * float(temps))

# Il faut que je trace les vitesses de convergence pour descente gradiant et descente gradiant stochastique et comparer les jL2 des 2 methodes
def descanteGradiant(x1, y, teta, tps, epsi=0.0000001, N=100):
    resTeta = teta
    resTps = tps
    while(True):
        gradteta = np.dot(x1,(y - np.dot(x1.T,resTeta)))
        tetaplus = resTeta + np.dot(np.dot(alpha(tps),gradteta),1./float(N))

        if math.fabs(jl2(x1,y,tetaplus) - jl2(x1,y,resTeta)) <= epsi:
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

mcarres = jGradiant(x1, p)
print "ABS = ", jAbs(x1,p,teta)
print "JL1 = ", jl1(x1,p,teta)
print "JL2 = ", jl2(x1,p,teta)
print "JLINF = ", jInf(x1,p,teta)
print "Grad = ", mcarres

#pour la descante de gradiant, np.array([b,a],float)
# tetaDescante = np.array([3,2],float)
print "descante gradiant =", descanteGradiant(x1,p,teta,1)
print "descante gradiant stochastique = ", descanteGradiantStochastique(x1,p,teta,1)
# pour récup les valeurs du gradiant
c = mcarres.item(1)
d = mcarres.item(0)

u = c*x + d

# plt.plot(t,p,'.')
# # l1, =
# plt.plot(x,y,label="Theorique")
# # l2, =
# plt.plot(x,u,label="Moindres carres")
# # plt.legend(handles=[l1,l2])
# plt.xlabel('t (s)')
# plt.ylabel('p (m)')
# plt.title('TP1 FAA')
# plt.show()
