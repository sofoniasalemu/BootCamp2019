# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 21:13:19 2019

@author: Sofonias Alemu
"""

import numpy as np
from scipy.optimize import  root as eqsolver

def v_pr(x,gamma):
    return np.power(x,-gamma)

def eq(e0,e1,e2,a1,a2,var,gamma):
    thet=var[0:2]
    q=var[2:4]
    res=np.zeros(4)
    res[0]=-v_pr(e0-thet[0]*q[0]-thet[1]*q[1],gamma)*q[0]+np.average(v_pr(e1+thet[0]*a1+thet[1]*a2,gamma)*a1)
    res[1]=-v_pr(e0+thet[0]*q[0]+thet[1]*q[1],gamma)*q[0]+np.average(v_pr(e2-thet[0]*a1-thet[1]*a2,gamma)*a1)
    res[2]=-v_pr(e0-thet[0]*q[0]-thet[1]*q[1],gamma)*q[1]+np.average(v_pr(e1+thet[0]*a1+thet[1]*a2,gamma)*a2)
    res[3]=-v_pr(e0+thet[0]*q[0]+thet[1]*q[1],gamma)*q[1]+np.average(v_pr(e2-thet[0]*a1-thet[1]*a2,gamma)*a2)
    return res
    
def ex1(gamma):    
    e0=1.
    e1=np.array([1,2,1,2])
    e2=np.array([3,1,3,1])
    a1=np.array([1,1,1,1])
    a2=np.array([1,1,1.5,1.5])
    guess=np.zeros(4)
    obj=lambda var: eq(e0,e1,e2,a1,a2,var,gamma)    
    sol=eqsolver(obj,guess)
    return sol.x

Gamma=np.array([2,4,8,166])
Sol=np.zeros([4,4])
for i in range(4):
    Sol[:,i]=ex1(Gamma[i])
    
###  Investor 1

theta_1_1=Sol[0,:] #Investment on risk free asset
theta_1_2=Sol[1,:] #Investment on risky asset

###  Investor 1

theta_2_1=-Sol[0,:] #Investment on risk free asset
theta_2_2=-Sol[1,:] #Investment on risky asset    

### Equilibrium Price

q_1=Sol[2,:]
q_2=Sol[3,:]