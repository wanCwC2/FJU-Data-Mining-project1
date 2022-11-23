import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('ignore')

#Read data
data = pd.read_csv('data/project1_train.csv')
test = pd.read_csv('data/project1_test.csv')

#Revise Male, Female
data.loc[data.Gender=='Male', 'Gender'] = 1
data.loc[data.Gender=='Female', 'Gender'] = 0
test.loc[test.Gender=='Male', 'Gender'] = 1
test.loc[test.Gender=='Female', 'Gender'] = 0

#Find analysis target
X_df = pd.DataFrame(data, columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio'])
y_df = pd.DataFrame(data, columns = ['Label'])

#Dvide the data into validation and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X_df, y_df, train_size=0.6, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, train_size=0.5, random_state=0)

#Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)
X_test_std = sc.transform(X_test)

#Delete too big data
#X_train_std.drop(X_train_std[X_train_std[:,5]>3].index) #pandas data, but change to array after standardizating
X_train_std = np.delete(X_train_std, X_train_std[:,5]>3, axis=0)
X_valid_std = np.delete(X_valid_std, X_valid_std[:,5]>3, axis=0)

sc_test = StandardScaler()
sc_test.fit(test)
test_std = sc_test.transform(test)

#多層感知器 Multi-layer Perceptron / 類神經網路 Neural network
model = MLPClassifier(
  hidden_layer_sizes=(10),
  max_iter=10,
  solver="adam",
  random_state=100002
)
model.fit(X_train_std, y_train)
print("Training set score: %f" % model.score(X_train_std, y_train))
print("Validation set score: %f" % model.score(X_valid_std, y_valid))