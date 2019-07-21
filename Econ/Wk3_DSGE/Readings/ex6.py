import numpy as np
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
    return sol







tau0=.05
alpha0=.4
gamma0=2.5
xi0=1.5
delta0=.10
a0=.5
beta0=.98

res_num=steady(tau0,alpha0,gamma0,xi0,delta0,a0,beta0,obj1)
res_num=np.append(res_num,tau0*(res_num[1]*res_num[2]+(res_num[4]-delta0)*res_num[3]))

print("\n Numerical Solution:  ", "\nc=",res_num[0],"\n w=",res_num[1],"\n l=",res_num[2],"\n k=",res_num[3], "\n r=",res_num[4], "\n T=",res_num[5])

