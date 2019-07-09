# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:35:55 2019

@author: Sofonias Alemu
"""
import numpy as np

def ex_21(g,a,b,N,met):

    if met=="midpoint":
        P=np.linspace(a,b,N)
        ans= (b-a)/N*np.sum(g(P[:-1]))        
    elif met=="trapezoid":
        P=np.linspace(a,b,N)
        ans= ((b-a)/(2*N))*(g(P[0])+2*np.sum(g(P[1:-1]))+g(P[-1]))
    elif met=="Simpsons":    
        P=np.linspace(a,b,2*N)
        v1=np.sum(g(P[range(1,2*N-1,2)]))
        v2=np.sum(g(P[range(2,2*N-2,2)]))
        ans= ((b-a)/(6*N))*(g(P[0]) + 4*v1 + 2*v2 + g(P[-1]))
    return ans        

f=lambda x: .1*np.power(x,4)-1.5*np.power(x,3)+.53*np.power(x,2)+2*x+1    
print(ex_21(f,-10,10,1000,'midpoint')) 
print(ex_21(f,-10,10,1000,'trapezoid')) 
print(ex_21(f,-10,10,1000,'Simpsons')) #####

### Exercise 2

def ex_22(mu,sigma,N,k):
    from scipy.stats import norm as nor
    Z=np.linspace(mu-k*sigma,mu+k*sigma,N)
    w=np.zeros(N)
    w[0]=nor.cdf((Z[0]+Z[1])/2,mu,sigma)
    for i in range (N-2):
        Z_min=.5*(Z[i-1]+Z[i])
        Z_max=.5*(Z[i+1]+Z[i])
        #w[i]=nor(mu,sigma).cdf(Z_max)-nor(mu,sigma).cdf(Z_min)
        w[i]=(Z_max-Z_min)*nor(mu,sigma).pdf(Z[i])
        w[-1]=1-nor.cdf((Z[-2]+Z[-1])/2,mu,sigma)
    return w, Z
    
w,Z=ex_22(0,1,11,4)

###  Exercise 2.3    

def ex_23(mu,sigma,N,k):
    W,A=ex_22(mu,sigma,N,k)
    return W,np.exp(A)

### Exercise 2.4
    
mu,sigma=10.5,.8
N,k=100,10
W,A=ex_23(mu,sigma,100,k)

res1=np.exp(mu+.5*sigma**2)
res2=np.average(A,weights=W)
    

### Exercise 3.1 and 3.2



#def g(a,b,v,N):
#    w=v[0:N]
#    x=v[N:]
#    R=np.zeros(2*N) 
#    A=np.linspace(0,N-1,N)
#    for i in range(2*N-1):
#        R[i]=(A[i]+1)*(b**A[i]-a**A[i])-w@x**A[i]
#    return R
def g(a,b,v):
    w=v[0:3]
    x=v[3:]
    R=np.zeros(6)
    R[0]=(b-a)-(w[0]+w[1]+w[2])
    R[1]=(1/2)*(b**2-a**2)-(w[0]*x[0]+w[1]*x[1]+w[2]*x[2])
    R[2]=(1/3)*(b**3-a**3)-(w[0]*x[0]**2+w[1]*x[1]**2+w[2]*x[2]**2)
    R[3]=(1/4)*(b**4-a**4)-(w[0]*x[0]**3+w[1]*x[1]**3+w[2]*x[2]**3)
    R[4]=(1/5)*(b**5-a**5)-(w[0]*x[0]**4+w[1]*x[1]**4+w[2]*x[2]**4)
    R[5]=(1/6)*(b**6-a**6)-(w[0]*x[0]**5+w[1]*x[1]**5+w[2]*x[2]**5)
    return R

def Gauss_quad(a,b,f,N):       
    g_1=lambda v: g(a,b,v)
    from scipy.optimize import  root as eqsolver
    vstar=eqsolver(g_1,np.ones(2*N)).x
    w=vstar[0:N]
    x=vstar[N:]
    return f(x)@w
 
f=lambda x: .1*np.power(x,4)-1.5*np.power(x,3)+.53*np.power(x,2)+2*x+1  

print(Gauss_quad(-10,10,f,3)) 
    
import scipy as sc
print(sc.integrate.quad(f,-10,10))


### Exercise 4.1
def gg(X):
    d=np.shape(X)[0]
    V=np.zeros(d)
    for i in range(np.shape(X)[0]):
        if  np.linalg.norm(X[i,:],2)<=1:
            V[i]=1
        else:
            V[i]=0  
    return V
    
def f(g,N):
    np.random.seed(seed=25)
    #Ran=np.zeros([N,2])
    #Ran[:,0]=np.random.uniform(-1,1,size=(N))  
    #Ran[:,1]=np.random.uniform(-1,1,size=(N))     
    Ran=np.random.uniform(-1,1,size=(2,N)).transpose()
    v=g(Ran)
    return np.average(v)*4

f(gg,41900)
        
    
    


    
    
    
    
    
    
    
