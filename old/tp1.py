#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TP1 FAA
# Auteur : Honoré NINTUNZE, 2016

import matplotlib.pyplot as plt
import numpy as np
import math

a = 2
b = 3
e = 1
N = 100

t = np.array([b,a],float)

f1 = open("data/t.txt")

f2 = open("data/p.txt")

x = np.linspace(4,16,N)
y = a*x + b

t = [ float(x) for x in f1.read().split()]
p = [ float(x) for x in f2.read().split()]

f1.close()
f2.close()


# xI = np.matrix([[x for x in t], [e]*N])
xI = np.zeros((2,N))
xI[1,:] = t
xI[0,:] = [e]*N
# def jl1(teta):
#     res = p - np.dot(xI.T,teta)
#     return math.sqrt(np.dot(res.T,res))/N

def jl2(teta):
    res = np.array(p) - np.dot(xI.T,teta)
    return (1.0/N) * np.dot(res.T, res)

def jAbs(teta):
    res = p - np.dot(xI.T,teta)
    return (1.0/N) * np.sum(np.absolute(res))

def jInf(teta):
    res = p - np.dot(xI.T,teta)
    return np.amax(np.absolute(res))

# print "abs", jAbs(t)
# print "jl1", jl1(np.array([b,a],float))
print "jl2", jl2(t)
# print "jinf", jInf(t)
#
#
# plt.xlabel('time (s)')
# plt.ylabel('position (m)')
# plt.title('Tp1 FAA')
# plt.plot(t, p,'.')
# plt.plot(x,y)
# plt.savefig("figure.png")
# plt.show()
