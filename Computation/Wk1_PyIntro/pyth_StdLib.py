# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 18:29:27 2019

@author: Sofonias Alemu
"""

import numpy as np

#### Problem 1 ####
def prob1(L):
    return min(L),max(L), sum(L)/len(L)

### Problem 2 ####

# Int
    
a1_int=np.array([1,2,3],dtype=np.int)
a2=a1_int
a2[1]=2
a1_int==a2

a1_tuple = (4, 5, 6)
a2=a1_tuple
a2[1]=6

a1_set = {4, 5, 6}
a2=a1_set
a2[1]=6

#### Problem 3 ####

import calculator as cal

def hypo(a,b):
    return cal.sqrt1(cal.sum1(cal.prod1(a,a),cal.prod1(b,b)))

hypo(2,3)

### Problem 4 #####

from itertools import combinations, chain

def prob4(A):
    lis={"empty"}
    for i in np.arange(len(A)):
        lis=list(chain(lis,combinations(A,i+1)))
        
    return lis

prob4("ABC")

#### Problem 5 ####

run shutthebox.py Jack 300





