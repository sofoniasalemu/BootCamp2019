# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 00:42:08 2019

@author: Sofonias Alemu
"""

import numpy as np
import pandas as pd
from datetime import datetime

from matplotlib import pyplot as plt

###### Problem 1 #############
pb1=pd.read_csv("DJIA.csv")

pb1=pb1[pb1.VALUE!="."]


pb1.VALUE=pd.to_numeric(pb1.VALUE)
pb1.DATE=pd.to_datetime(pb1.DATE)
pb1.plot(x='DATE',y='VALUE')

##### Problem 2 ############

pb2=pd.read_csv("paychecks.csv")
pb2=pb2.rename(columns={"1122.26":"Wage"})
l=len(pb2.Wage)
pb2['DATE']=pd.date_range(start='3/13/2008',periods=l,freq="WOM-3FRI")
pb2.plot(x='DATE',y='Wage')


## Problem 5 ###########

largest_gain_day = pb1.max().DATE
largest_loss_day = pb1.min().DATE

### Problem 6 ############

ax1=plt.subplot(111)
pb1.plot(x='DATE',y='VALUE',ax=ax1)
pb1=pd.read_csv("DJIA.csv")
pb1=pb1[pb1.VALUE!="."]
pb1.VALUE=pd.to_numeric(pb1.VALUE)
pb1.rolling(window=20).mean().plot(color='g',ax=ax1)
pb1.rolling(window=20).min().plot(color='r',ax=ax1)
pb1.rolling(window=20).max().plot(color='b',ax=ax1)
pb1.ewm(span=20).mean().plot(color='r',ax=ax1)
ax1.legend(["actual","rolling av","rolling min","rolling max","exp av"])