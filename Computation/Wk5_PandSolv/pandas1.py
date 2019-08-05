# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:16:04 2019

@author: Sofonias Alemu
"""
import pandas as pd
import numpy as np

####### problem 1 #######

s1 = pd.Series(np.array(range(0,50))**2-1)

s1=s1[np.array([i % 2==0 for i in s1.index])]

s1[np.array([i % 3==0 for i in s1.index])]=0

##### problem 2  #########

def fun_2(p,d):
    dates=pd.date_range("1/1/2000","12/31/2000", freq='D')
    s2=pd.Series(d*np.ones([len(dates),]),index=dates)
    v=np.random.binomial(1,p,len(dates)-1)
    v[v==0]=-1
    v=np.append(0,v)
    s2_2=pd.Series(v,index=dates)
    
    s2_2=s2_2.cumsum()    
    s3=s2+s2_2

    return s3

s3=fun_2(.5,100)
    
from matplotlib import pyplot as plt
plt.figure(1)
plt.plot(fun_2(.2,100).values)
plt.ylabel('Wealth')
plt.figure(2)
plt.plot(fun_2(.5,100).values)
plt.ylabel('Wealth')
plt.figure(3)
plt.plot(fun_2(.7,100).values)
plt.ylabel('Wealth')

####### Problem 3 ######


name = ['Mylan', 'Regan', 'Justin', 'Jess', 'Jason', 'Remi', 'Matt', '\
Alexander', 'JeanMarie']
sex = ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'F']
age = [20, 21, 18, 22, 19, 20, 20, 19, 20]
rank = ['Sp', 'Se', 'Fr', 'Se', 'Sp', 'J', 'J', 'J', 'Se']
ID = range(9)
aid = ['y', 'n', 'n', 'y', 'n', 'n', 'n', 'y', 'n']
GPA = [3.8, 3.5, 3.0, 3.9, 2.8, 2.9, 3.8, 3.4, 3.7]
mathID = [0, 1, 5, 6, 3]
mathGd = [4.0, 3.0, 3.5, 3.0, 4.0]
major = ['y', 'n', 'y', 'n', 'n']
studentInfo = pd.DataFrame({'ID': ID, 'Name': name, 'Sex': sex, 'Age': age, \
'Class': rank})
otherInfo = pd.DataFrame({'ID': ID, 'GPA': GPA, 'Financial_Aid': aid})
mathInfo = pd.DataFrame({'ID': mathID, 'Grade': mathGd, 'Math_Major': major})

studentInfo[(studentInfo['Age']>19) & (studentInfo['Sex']=='M')][['ID','Age']]

###### Probem 4 ##########

pd.merge(otherInfo[studentInfo['Sex']=='M']\
         ,studentInfo[studentInfo['Sex']=='M']\
         , on='ID')[['ID','Age','GPA']]