# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 22:04:29 2019

@author: Sofonias Alemu
"""

import box
import random
import time
import sys

if len(sys.argv)==3:
    print(sys.argv[1])
    print(sys.argv[2])
else:
    sys.exit('Exactly two extra command argument is required')
    
numbers=list(range(1,7))

remaining=numbers
tim0=time.time()
tim1=0
tt=float(sys.argv[2])
name=sys.argv[1]

while len(remaining)>0 and tim1<=tt:
    tim1=time.time()-tim0
    if sum(remaining)<=6:
        roll=random.choice(numbers)
    else:
        roll=random.choice(numbers)+random.choice(numbers)
    print("Numbers left: ",remaining)
    print("Roll: ",roll)
    print("Seconds left",tt-tim1)      
        
    if box.isvalid(roll, remaining)==1:
        print(remaining)   
        x = input("Numbers to eliminate: ")
        choice=box.parse_input(x, remaining)
        if sum(choice)!=roll:
            x = input("Numbers to eliminate: ")
        print("Numbers to eliminate: ",choice)
        remaining= [x for x in remaining if (x not in choice)]


    else:
        print("Score for player "+name,sum(remaining),"points")
        print("Time played: ",tt-tim1)
        sys.exit('you lost')

    if tim1>=tt:
        print("Score for player "+name,sum(remaining),"points")
        print("Time played: ",tt-tim1)
        sys.exit('you lost')

print("Score for player "+name,sum(remaining),"points")
print("Time played: ",tt-tim1)   
sys.exit('you won')