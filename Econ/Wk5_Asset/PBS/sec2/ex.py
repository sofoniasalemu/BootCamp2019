# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 22:10:31 2019

@author: Sofonias Alemu
"""

import numpy as np
from scipy.optimize import broyden1 as br1
from scipy.optimize import minimize
import pandas as pd

###############  import data  ######################

defl=pd.read_csv("GDPDEF.csv")
cons=pd.read_csv("PCECC96.csv")
rf=pd.read_csv("TB3MS.csv")
index=pd.read_csv("WILL5000INDFC.csv")

dates_q=pd.to_datetime(cons['DATE'])
dates_m=pd.to_datetime(index['DATE'])

defl['DATE']=pd.to_datetime(defl['DATE'])
defl=defl.set_index(['DATE'])
defl=defl['1971-04-01':'2018-01-01']

cons['DATE']=pd.to_datetime(cons['DATE'])
cons=cons.set_index(['DATE'])
cons=cons['1971-04-01':'2018-01-01']

rf['DATE']=pd.to_datetime(rf['DATE'])
rf=rf.set_index(['DATE'])
rf=rf['1971-04-01':'2018-01-01']

index['DATE']=pd.to_datetime(index['DATE'])
index=index.set_index(['DATE'])
index=index['1971-04-01':'2018-01-01']


#defl=defl['GDPDEF'].values
#cons=cons['PCECC96'].values
#rf=rf['TB3MS'].values
#index=index['WILL5000INDFC'].values

############# change to quarterly ####################

v_i=index['WILL5000INDFC'].values[1:]/index['WILL5000INDFC'].values[:-1]
v_rf=(rf['TB3MS'].values[1:]/100+1)**(1/12)
index_a=np.empty([1,])
rf_a=np.empty([1,])
for i in range(2,560,3):
    index_a=np.append(index_a,v_i[i]*v_i[i-1]*v_i[i-2])
    rf_a=np.append(rf_a,v_rf[i]*v_rf[i-1]*v_rf[i-2])
defla=defl['GDPDEF'].values[1:]/defl['GDPDEF'].values[:-1] 

index_a=index_a*defla
cons_a=cons['PCECC96'].values[1:]
############  EX  ###################################


def mom1(index_a,cons_a,gamma,beta):
  return 1-np.mean(beta*(cons_a[1:]/cons_a[:-1])**(-gamma)*index_a[1:])

def mom2(rf_a,cons_a,gamma,beta):
  return 1-np.mean(beta*(cons_a[1:]/cons_a[:-1])**(-gamma)*rf_a[1:])


beta=.99
f_1=lambda gamma: mom1(index_a,cons_a,gamma,beta)
f_2=lambda gamma: mom2(rf_a,cons_a,gamma,beta)

def gmm_moms(x,M):
    V=np.empty([2,])
    V[0]=mom1(index_a,cons_a,x[0],x[1])
    V[1]=mom2(rf_a,cons_a,x[0],x[1])
    return np.dot(np.dot(V,M),V)

gamma_1=br1(f_1,.1)
gamma_2=br1(f_2,.5)

M=np.eye(2)
M_1=np.cov(index_a,rf_a)
f_3=lambda x: gmm_moms(x,M)
f_4=lambda x: gmm_moms(x,M_1)

x_sol=minimize(f_3,np.array([.9,.1])).x
gamma_3=x_sol[0]
beta_3=x_sol[1]

x_sol=minimize(f_4,np.array([.9,.1])).x
gamma_4=x_sol[0]
beta_4=x_sol[1]

print("\n################ \
      Estimation of gamma from the euler equation with market return\
      #################\n")
print("gamma = ",gamma_1,"\n")

print("################ \
      Estimation of gamma from the euler equation with risk free return\
      #################\n")
print("gamma = ",gamma_2,"\n")

print("################\
      GMM estimation of gamma and beta with Identity matrix\
      ################\n")
print("gamma = ",gamma_3)
print("beta = ",beta_3,"\n")

print("################ \
      GMM estimation of gamma and beta with Covariance matrix \
      ###############\n")
print("gamma = ",gamma_4)
print("beta = ",beta_4)