import numpy as np
from scipy.optimize import  root as eqsolver
from scipy.optimize import  minimize

def obj1(c,w,k,r,tau,alpha,gamma,delta,beta):
        V=np.zeros(4)
        V[0]=w+(r-delta)*k-c
        V[1]=beta*((r-delta)*(1-tau)+1)-1
        V[2]=alpha*k**(alpha-1)-r
        V[3]=(1-alpha)*k**alpha-w
        return V

def steady(tau,alpha,gamma,delta,beta,obj1):
    obj=lambda x: obj1(x[0],x[1],x[2],x[3],tau,alpha,gamma,delta,beta)
    sol=eqsolver(obj,np.array([1,1,.1,2])).x
    return sol

tau0=.05
alpha0=.4
gamma0=2.5
delta0=.10
beta0=.98

res_num=steady(tau0,alpha0,gamma0,delta0,beta0,obj1)
res_num=np.append(res_num,tau0*(res_num[1]+(res_num[3]-delta0)*res_num[2]))

res_ana=np.zeros(4)

res_ana[3]=((1/beta0)-1)/(1-tau0)+delta0 ## r
res_ana[2]=(res_ana[3]/alpha0)**(1/(alpha0-1)) ## k
res_ana[1]=(res_ana[2]**alpha0)*(1-alpha0)  ## w
res_ana[0]=res_ana[1]+(res_ana[3]-delta0)*res_ana[2] ## c

res_ana=np.append(res_ana,tau0*(res_ana[1]+(res_ana[3]-delta0)*res_ana[2]))

print("\n Analytical Solution:  ", "\nc=",res_ana[0],"\n w=",res_ana[1], "\n k=",res_ana[2], "\n r=",res_ana[3], "\n T=",res_ana[4])

print("\n Numerical Solution:  ", "\nc=",res_num[0],"\n w=",res_num[1], "\n k=",res_num[2], "\n r=",res_num[3], "\n T=",res_num[4])