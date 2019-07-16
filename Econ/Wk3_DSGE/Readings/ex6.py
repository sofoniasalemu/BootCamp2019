import numpy as np
from scipy.optimize import  root as eqsolver
from scipy.optimize import  minimize

def obj1(c,w,l,k,r,tau,alpha,gamma,xi,delta,a,beta):
    if c<0 or l>1:
        V=np.ones(5)
        return 1
    else:
        V=np.zeros(5)
        V[0]=w*l+(r-delta)*k-c
        V[1]=beta*((r-delta)*(1-tau)+1)-1
        V[2]=a*(1-l)**(-xi)-c**(-gamma)*w*(1-tau)
        V[3]=alpha*k**(alpha-1)*l**(1-alpha)-r
        V[4]=(1-alpha)*k**alpha*l**(-alpha)
        return V@V

def steady(tau,alpha,gamma,xi,delta,a,beta,obj1):
    obj=lambda x: obj1(x[0],x[1],x[2],x[3],x[4],tau,alpha,gamma,xi,delta,a,beta)
    sol=minimize(obj,np.array([1,1,.1,2,1])).x
    return sol

tau0=.05
alpha0=.5
gamma0=2.5
xi0=1.5
delta0=.10
a0=.5
beta0=.98

print(steady(tau0,alpha0,gamma0,xi0,delta0,a0,beta0,obj1))


#x=np.ones(5)*.2
#tau,alpha,gamma,xi,delta,a,beta=tau0,alpha0,gamma0,xi0,delta0,a0,beta0

#a*(1-x[2])**(-xi)-x[0]**(-gamma)*x[1]*(1-x[4])