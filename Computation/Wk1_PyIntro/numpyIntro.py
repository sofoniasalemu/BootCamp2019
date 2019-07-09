# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:45:46 2019

@author: Sofonias Alemu
"""

import numpy as np

##### Problem 1 ####

def prob1(A1,B1):
    A=np.array(A1)
    B=np.array(B1)

    return A@B

A1=[[3,-1,4],[1,5,-9]]
B1=[[2,6,-5,3],[5,-8,9,7],[9,-3,-2,-3]]
C1=prob1(A1,B1)

####  Problem 2 ####

def prob2(A1):
    A=np.array(A1)
    return -(A@A)@A+9*A@A-15*A

C2=prob2([[3,1,4],[1,5,9],[-5,3,1]])

###  Problem 3 ####

def prob3(d2,d1):
    D1=np.ones((d2,d2),dtype=np.int)
    A=np.triu(D1)
    B=-1*A.transpose()+d1*np.triu(D1)-np.eye(d2,dtype=np.int)*d1
    return A.astype(np.float64), B.astype(np.float64)

C31,C32=prob3(7,5)

####  Problem 4 ####

def prob4(A1):
    A1=np.array(A1)
    A2=A1
    A2[A2<0]=0
    return A2

A1=[1,2,3,-3]
C4=prob4(A1)

#### Problem 5 ####

def prob5(a1,b1,c1):
    A1=np.arange(a1).reshape((3,2)).transpose()
    B10=np.ones((3,3),dtype=np.int)
    B1=np.tril(b1*B10)
    C1=c1*np.eye(3,dtype=np.int)
    col1=np.vstack((np.zeros((3,3),dtype=np.int),A1,B1))
    col2=np.vstack((A1.transpose(),np.zeros((5,2),dtype=np.int)))
    col3=np.vstack((np.eye(3,dtype=np.int),np.zeros((2,3),dtype=np.int),C1))

    return np.hstack((col1,col2,col3))

C5=prob5(6,3,-2)
    

#### Problem 6 #####

def prob6(A1):
    A1=np.array(A1)
    A2=A1.transpose()
    A3=A2/A2.sum(axis=0)
    return A3.transpose()
A1=[[1,2],[3,4]]
C6=prob6(A1)

#### Problem 7 #####

def prob7_diag(A1,t):
    V=np.zeros((A1.shape[1],1))
    for i in np.arange(A1.shape[1]-t):
        V[i]=np.prod(A1[i:i+t,i:i+t].diagonal())
    return np.max(V)

def prob7_vert(A1,t):
    V=np.zeros((A1.shape[1],1))
    for i in np.arange(A1.shape[1]-t):
        V[i]=max(np.prod(A1[i:i+t,:],axis=0))
    return np.max(V)


def prob7_hor(A1,t):
    V=np.zeros((A1.shape[1],1))
    for i in np.arange(A1.shape[1]-t):
        V[i]=max(np.prod(A1[:,i:i+t],axis=1))
    return np.max(V)

grid = np.load('grid.npy')

max_diag=prob7_diag(grid,4)
max_hor=prob7_hor(grid,4)
max_vert=prob7_vert(grid,4)
