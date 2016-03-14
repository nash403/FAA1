#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math

a = 2
b = 3
e = 1
N = 100

f1 = open("data/t.txt")

f2 = open("data/p.txt")

t = [ float(x) for x in f1.read().split()]
p = [ float(x) for x in f2.read().split()]

f1.close()
f2.close()

def f(x):
    return 2*x +3

# print(t)
# print(p)
# plt.xlabel('time (s)')
# plt.ylabel('position (m)')
# plt.title('Test')
# plt.grid(True)
# plt.plot(t, p,'+')
# plt.plot([f(x) for x in range(20)],'r-')
# plt.show()

def fXi(teta):
    x = np.matrix([[x for x in t], [e]*N])
    return (np.transpose(x) * teta)

def jl1(teta):
    res = p - fXi(teta)
    return (1.0/N) * math.sqrt(np.transpose(res) * res)

def jl2(teta):
    res = p - fXi(teta)
    m = (1.0/N) * np.transpose(res) * res
    return m[0,0]

print "jl1", jl1(np.matrix([[a],[b]]))
print "jl2", jl2(np.matrix([[a],[b]]))
