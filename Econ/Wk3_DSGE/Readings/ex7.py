# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 04:10:18 2019

@author: Sofonias Alemu
"""

import autograd.numpy as np
from scipy.optimize import  root as eqsolver
#from scipy.optimize import  minimize

def obj1(c,w,l,k,r,tau,alpha,gamma,xi,delta,a,beta):
        V=np.zeros(5)
        V[0]=w*l+(r-delta)*k-c
        V[1]=beta*((r-delta)*(1-tau)+1)-1
        V[2]=a*(1-l)**(-xi)-c**(-gamma)*w*(1-tau)
        V[3]=alpha*k**(alpha-1)*l**(1-alpha)-r
        V[4]=(1-alpha)*k**alpha*l**(-alpha)-w
        return V

def steady(tau,alpha,gamma,xi,delta,a,beta,obj1):
    obj=lambda x: obj1(x[0],x[1],x[2],x[3],x[4],tau,alpha,gamma,xi,delta,a,beta)
    sol=eqsolver(obj,np.array([1,1.5,.1,1.2,1])).x
    sol=np.append(sol,tau0*(sol[1]*sol[2]+(sol[4]-delta)*sol[3]))
    return sol


def der_cen_2(f,x,h,i,j):
    return (f(x+h*np.eye(1,len(x),i)[0])[j]-f(x-h*np.eye(1,len(x),i+1)[0])[j])/(2*h)

def jacob(f,x0,h):
    Jac=np.zeros([7,6])
    for i in range(7):
        for j in range(6):                     
                Jac[i,j]=der_cen_2(f,x0,h,i,j)
    return Jac

f=lambda x:steady(x[0],x[1],x[2],x[3],x[4],x[5],x[6],obj1)

tau0=.05
alpha0=.4
gamma0=2.5
xi0=1.5
delta0=.10
a0=.5
beta0=.98

x0=np.array([tau0,alpha0,gamma0,xi0,delta0,a0,beta0])
jjacob=jacob(f,x0,1e-4)

##################################################################################################################
K=10
h0=1e-5

tau1=np.linspace(0.01,.08,K)#.05
alpha1=np.linspace(0.1,.7,K)#.4
gamma1=np.linspace(2,3,K)#2.5
xi1=np.linspace(1,2,K)#1.5
delta1=np.linspace(0.05,.15,K)#.10
a1=np.linspace(0.2,.7,K)#.5
beta1=np.linspace(0.9,.99,K)#.98

wrt_tau,wrt_alpha,wrt_gamma,wrt_xi,wrt_delta,wrt_a,wrt_beta=np.zeros([K,7,6]),np.zeros([K,7,6]),np.zeros([K,7,6]),np.zeros([K,7,6]),np.zeros([K,7,6]),np.zeros([K,7,6]),np.zeros([K,7,6])

def fun_1(f,h0,K,tau1,alpha1,gamma1,xi1,delta1,a1,beta1,tau0,alpha0,gamma0,xi0,delta0,a0,beta0):
    for i in range(K):
        wrt_tau[i,:,:]=jacob(f,np.array([tau1[i],alpha0,gamma0,xi0,delta0,a0,beta0]),h0)
        wrt_alpha[i,:,:]=jacob(f,np.array([tau0,alpha1[i],gamma0,xi0,delta0,a0,beta0]),h0)
        wrt_gamma[i,:,:]=jacob(f,np.array([tau0,alpha0,gamma1[i],xi0,delta0,a0,beta0]),h0)
        wrt_xi[i,:,:]=jacob(f,np.array([tau0,alpha0,gamma0,xi1[i],delta0,a0,beta0]),h0)
        wrt_delta[i,:,:]=jacob(f,np.array([tau0,alpha0,gamma0,xi0,delta1[i],a0,beta0]),h0)
        wrt_a[i,:,:]=jacob(f,np.array([tau0,alpha0,gamma0,xi0,delta0,a1[i],beta0]),h0)
        wrt_beta[i,:,:]=jacob(f,np.array([tau0,alpha0,gamma0,xi0,delta0,a0,beta1[i]]),h0)
    return wrt_tau,wrt_alpha,wrt_gamma,wrt_xi,wrt_delta,wrt_a,wrt_beta

    
wrt_tau,wrt_alpha,wrt_gamma,wrt_xi,wrt_delta,wrt_a,wrt_beta=fun_1(f,h0,K,tau1,alpha1,gamma1,xi1,delta1,a1,beta1,tau0,alpha0,gamma0,xi0,delta0,a0,beta0)

from matplotlib import pyplot as plt

def plot_fun(fun,par,wrt_tau,wrt_alpha,wrt_gamma,wrt_xi,wrt_delta,wrt_a,wrt_beta):   
    fun_list = ['c','w','l','k','r','T']
    par_list = ['tau','alpha','gamma','xi','delta','a','beta']
    if fun not in fun_list:
            return 
    elif par not in par_list:
            return 
    else:
        ff=fun_list.index(fun)
        if par=='tau':
            plt.plot(wrt_tau[:,1,ff])
        if par=='alpha':
            plt.plot(wrt_alpha[:,2,ff])
        if par=='gamma':
            plt.plot(wrt_gamma[:,3,ff])
        if par=='xi':
            plt.plot(wrt_xi[:,4,ff])
        if par=='delta':
            plt.plot(wrt_delta[:,5,ff])
        if par=='a':
            plt.plot(wrt_a[:,6,ff])
        if par=='beta':
            plt.plot(wrt_beta[:,7,ff])     
        plt.xlabel(par, size=16)
        plt.ylabel(fun, size=16)
        plt.show()  


            
plot_fun('c','xi',wrt_tau,wrt_alpha,wrt_gamma,wrt_xi,wrt_delta,wrt_a,wrt_beta)   
            
    