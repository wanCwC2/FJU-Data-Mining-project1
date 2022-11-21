import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
warnings.filterwarnings('ignore')

data = pd.read_csv('data/project1_train.csv')
test = pd.read_csv('data/project1_test.csv')