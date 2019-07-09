# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 07:40:38 2019

@author: Sofonias Alemu
"""

import numpy as np
from matplotlib import pyplot as plt

### Exercise 1

def pb1(n):
    A=np.random.normal(0,1,size=(n,n))
    M=np.mean(A,axis=0)
    return np.var(M)

def pb1_2(N):
    V=np.zeros(np.shape(N)[0])
    for i in range(np.shape(N)[0]):
        V[i]=pb1(N[i])
    return V

N=np.linspace(100,10,1000,dtype=int)

V=pb1_2(N)  
plt.figure(1)
plt.plot(V)      

### Exercise 2

X=np.linspace(-2*np.pi,2*np.pi,100)
Y_cos=np.sin(X)
Y_sin=np.cos(X)
Y_arctan=np.arctan(X)

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(X,Y_cos)
plt.subplot(3,1,2)
plt.plot(X,Y_sin)
plt.subplot(3,1,3)
plt.plot(X,Y_arctan)

### Exercise 3

f=lambda x: 1/(x-1)
X=np.linspace(-2,6,100)

a=(X>=-2 - X<1).any()
b=X>1 and X<=6
X1=X[a]
X2=X[b]

Y1=np.nan(100)
Y1[X>=-2 and X<1]=f(X>=-2 and X<1)
Y2=np.nan(100)
Y2[X>1 and X<=6]=f(X>1 and X<=6)

plt.figure(3)
plt.plot(X,Y1)
plt.plot(X,Y2)



