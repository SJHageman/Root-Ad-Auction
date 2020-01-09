# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 13:33:32 2020

@author: Tebe
"""
import os, sys, time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.rcParams['figure.dpi'] = 240 # fix high-dpi display scaling issues

sys.path.append(os.getcwd()) # add cwd to path

from zip_codes import ZC # zip code database
import load_file as lf # file i/o
import myplots as mp # my plotting functions
import file_reshaper as fr # file reshaper

zc = ZC(fdir='') # initialize zip code class

### define your data directory
data_dir = r'K:\Data Science\Root Ad Data\Data'

### convert files to gzip (skip if you already have gzips)
fr.reshape_files(data_dir=data_dir) # this will take 15-20 minutes
fr.local_hour_creator(data_dir=data_dir) # this will take another 15-20 minutes
