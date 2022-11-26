import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def data_standardization(df_input):
    sc = StandardScaler()
#    sc = MinMaxScaler()  
    df=sc.fit_transform(df_input)
    return df

warnings.filterwarnings('ignore')

#Read data
data = pd.read_csv('data/project1_train.csv')
test = pd.read_csv('data/project1_test.csv')

#data = data.drop(data[data['Albumin_and_Globulin_Ratio'].isnull()].index)
from sklearn.impute import SimpleImputer
# 建立以平均值填補缺損值的實體
imp = SimpleImputer(strategy='most_frequent')  #mean, median, most_frequent
# 填補缺損值
imp.fit(data)
data2 = imp.transform(data)
data = pd.DataFrame(data2, index = data.index, columns = data.columns)

#drop 29, 60
#data = data.drop([29, 60])

#Revise Male, Female
data.loc[data.Gender=='Male', 'Gender'] = 1
data.loc[data.Gender=='Female', 'Gender'] = 0
test.loc[test.Gender=='Male', 'Gender'] = 1
test.loc[test.Gender=='Female', 'Gender'] = 0

#Standardization
X_df = data_standardization(data.iloc[:,0:10])
y_df = data['Label'].astype('int')

#Dvide the data into validation and test sets
X_train , X_test , y_train , y_test = train_test_split(X_df ,y_df , test_size=0.3 , random_state=408570344)

#XGBoost
from xgboost import XGBClassifier
# declare parameters

params = { 'max_depth': 1,
           'learning_rate': 0.03,
           'n_estimators': 300,
           'colsample_bytree': 1}

#Find best parameters
# Best parameters: {'colsample_bytree': 1, 'learning_rate': 0.03, 'max_depth': 1, 'n_estimators': 300}
'''
params = { 'max_depth': [1, 3, 6, 10],
           'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.3, 0.5],
           'n_estimators': [100, 300, 600, 1000],
           'colsample_bytree': [1, 3, 6, 10]}

from sklearn.model_selection import GridSearchCV
xg2 = XGBClassifier(random_state=408570344)
clf = GridSearchCV(estimator = xg2,
                   param_grid = params,
                   scoring = 'neg_mean_squared_error',
                   verbose=1)
clf.fit(X_df, y_df)

print("Best parameters:", clf.best_params_)
'''

# instantiate the classifier 
xgb_clf = XGBClassifier(**params)
# fit the classifier to the training data
xgb_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = xgb_clf.predict(X_test)
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

'''
import xgboost as xgb
xgb.plot_importance(xgb_clf)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()
'''