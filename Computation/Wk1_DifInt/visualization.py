# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 06:03:29 2019

@author: Sofonias Alemu
"""
import numpy as np
from matplotlib import pyplot as plt
###  Exercise 2

def bern(n,x):
    import scipy.special as sp
    Res=np.zeros([n+1,np.shape(x)[0]])
    for v in range(n+1):
        Res[v]=sp.binom(n,v)*x**v*(1-x)**(n-v)
    return Res
N=100
X=np.linspace(0,1,N)

plt.figure(1)
plt.subplot(4,4,1)
plt.plot(X,np.ones(N))
plt.subplot(4,4,5)
plt.plot(X,bern(1,X)[0,:])
plt.subplot(4,4,6)
plt.plot(X,bern(1,X)[1,:])
plt.subplot(4,4,9)
plt.plot(X,bern(2,X)[0,:])
plt.subplot(4,4,10)
plt.plot(X,bern(2,X)[1,:])
plt.subplot(4,4,11)
plt.plot(X,bern(2,X)[2,:])
plt.subplot(4,4,13)
plt.plot(X,bern(3,X)[0,:])
plt.subplot(4,4,14)
plt.plot(X,bern(3,X)[1,:])
plt.subplot(4,4,15)
plt.plot(X,bern(3,X)[2,:])
plt.subplot(4,4,16)
plt.plot(X,bern(3,X)[3,:])

#### Exercise 3        

A=np.load("MLB.npy")
plt.figure(1)
plt.subplot(1,3,1)
plt.scatter(A[:,0],A[:,1])
plt.xlabel("height(inches)")
plt.ylabel("weight(pounds)")
plt.subplot(1,3,2)
plt.scatter(A[:,0],A[:,2])
plt.xlabel("height(inches)")
plt.ylabel("age(years)")
plt.subplot(1,3,3)
plt.scatter(A[:,1],A[:,2])
plt.xlabel("weight(pounds)")
plt.ylabel("age(years)")

#### Exercise 5

x=np.linspace(-1.5,1.5,200)
X,Y=np.meshgrid(x,x)
Z=(1-X)**2+100*(Y-X**2)**2

plt.figure(2)
plt.subplot(2,1,1)
plt.pcolormesh(X,Y,Z)
plt.subplot(2,1,2)
plt.contour(X,Y,Z)
plt.plot(1,1,marker='o')

### Exercise 6

A=np.load("countries.npy")

