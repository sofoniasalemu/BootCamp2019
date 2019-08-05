# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:01:45 2019

@author: Sofonias Alemu
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pb1=pd.read_csv("titanic.csv")

plt.figure(1)
pb1.groupby('Sex')["Survived"].mean().plot(kind='bar',title="Survival Probability for male and females")



a=np.nanmin(pb1['Fare'].values)
b=np.nanmax(pb1['Fare'].values)
bins =  np.arange(a,b,(b-a)/4.)
plt.figure(2)
pb1.groupby(pd.cut(pb1["Fare"],bins))["Survived"].mean().plot(kind='bar',title="Survival Probability based on fares")