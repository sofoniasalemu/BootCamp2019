# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 00:23:16 2019

@author: Sofonias Alemu
"""

import numpy as np
import pandas as pd
from pydataset import data


##### Problem 2 ######
pb2=pd.read_csv("titanic.csv")

pb2.pivot_table(values="Survived",\
columns=["Embarked"],\
aggfunc="mean", fill_value='-')

pb2.pivot_table(values="Survived",\
index=["Sex"],columns=["Embarked"],\
aggfunc="mean", fill_value='-')


##### Problem 3 #########

