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

y_df = y_df.drop(X_df[X_df['Albumin_and_Globulin_Ratio'].isnull()].index)
X_df = X_df.drop(X_df[X_df['Albumin_and_Globulin_Ratio'].isnull()].index)

#Dvide the data into validation and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X_df, y_df, train_size=0.6, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, train_size=0.5, random_state=0)

#Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)
X_test_std = sc.transform(X_test)

#Change numpy array to pandas
X_train_std = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
X_valid_std = pd.DataFrame(X_valid_std, index=X_valid.index, columns=X_valid.columns)
'''
#Delete too big data
#X_train_std.drop(X_train_std[X_train_std[:,5]>3].index) #pandas data, but change to array after standardizating
for i in range(0, X_train_std.shape[1]):
    #y_train.drop(np.where(X_train_std[:,5]>2)[0])
    y_train = y_train.drop(X_train_std[X_train_std.iloc[:,i]>2].index)
    X_train_std = X_train_std.drop(X_train_std[X_train_std.iloc[:,i]>2].index)
    y_valid = y_valid.drop(X_valid_std[X_valid_std.iloc[:,i]>2].index)
    X_valid_std = X_valid_std.drop(X_valid_std[X_valid_std.iloc[:,i]>2].index)
'''
sc_test = StandardScaler()
sc_test.fit(test)
test_std = sc_test.transform(test)

#XGBoost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
n_estimators = [10,20,30,40,50,60,70,80,90,100]
max_depth = [1,2,3,4,5,6]
parameters_to_search = {'n_estimators': n_estimators, 
              'max_depth': max_depth} #設定要訓練的值
xgbModel = xgb.XGBRegressor(n_estimators = 100, max_depth = 6)
xgbModel_cv = GridSearchCV(xgbModel, parameters_to_search, cv=5) #可以直接找出最佳的訓練值
xgbModel_cv.fit(X_train_std, y_train.values.ravel())
#xgbScore = xgbModel_cv.score(X_test_std, y_test.values.ravel())
#print('Correct rate using XGBoost: {:.5f}'.format(xgbScore))
'''
#Use MSE sure whether it has overfitting.
#print("XGBoost's MSE")
from sklearn import metrics
train_pred = xgbModel_cv.predict(X_train_std)
xgbTrainMse = metrics.mean_squared_error(y_train.values.ravel(), train_pred)
#print('train data MSE: ', xgbTrainMse)
#test_pred = xgbModel_cv.predict(X_test_std)
#xgbTestMse = metrics.mean_squared_error(y_test.values.ravel(), test_pred)
#print('test data MSE: ', xgbTestMse)
'''
#SVR
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
#import numpy as np
svr_maxRate = 0 #驗證及訓練結果的最高正確率
svr_c = 2 #最適合用在此的SVR參數
svr_epsilon = 0.5 #最適合用在此的SVR參數
#rng = np.random.RandomState(0)
svrModel = make_pipeline(StandardScaler(), SVR(C=svr_c, epsilon=svr_epsilon))
svrModel.fit(X_train_std, y_train.values.ravel())
for i in range(1, 5):
    for j in range(0, 5):
        svrModel = make_pipeline(StandardScaler(), SVR(C=i, epsilon = j/10))
        svrModel.fit(X_train_std, y_train.values.ravel())
        if svr_maxRate < svrModel.score(X_valid_std, y_valid.values.ravel()):
            svr_maxRate = svrModel.score(X_valid_std, y_valid.values.ravel())
            svr_c = i
            svr_epsilon = j/10
svrModel = make_pipeline(StandardScaler(), SVR(C = svr_c, epsilon = svr_epsilon))
svrModel.fit(X_train_std, y_train.values.ravel())
#svrScore = svrModel.score(X_test_std, y_test.values.ravel())
#print('Correct rate using SVR: {:.5f}'.format(svrScore))
'''
#Use MSE sure whether it has overfitting.
print("SVR's MSE")
from sklearn import metrics
train_pred = svrModel.predict(X_train_std)
svrTrainMse = metrics.mean_squared_error(y_train.values.ravel(), train_pred)
print('train data MSE: ', svrTrainMse)
test_pred = svrModel.predict(X_test_std)
svrTestMse = metrics.mean_squared_error(y_test.values.ravel(), test_pred)
print('test data MSE: ', svrTestMse)
'''
# Random Forest
from sklearn.ensemble import RandomForestRegressor
'''
rf_maxRate = 0
rf_state = 0
for i in range (1,10):
    rfModel = RandomForestRegressor(random_state = i)
    rfModel.fit(X_train_std, y_train.values.ravel())
    if rf_maxRate < rfModel.score(X_valid_std, y_valid.values.ravel()):
        rf_maxRate = rfModel.score(X_valid_std, y_valid.values.ravel())
        rf_state = i
'''
rfModel = RandomForestRegressor(random_state = 408570344)
#RandomForestRegressor(random_state = rf_state)
#rfScore = round(rfModel.score(X_test_std, y_test.values.ravel()),5)
#print("Correct rate using Random Forest: ", rfScore)
'''
#Use MSE sure whether it has overfitting.
print("Random Forest's MSE")
from sklearn import metrics
train_pred = rfModel.predict(X_train_std)
rfTrainMse = metrics.mean_squared_error(y_train.values.ravel(), train_pred)
print('train data MSE: ', rfTrainMse)
test_pred = rfModel.predict(X_test_std)
rfTestMse = metrics.mean_squared_error(y_test.values.ravel(), test_pred)
print('test data MSE: ', rfTestMse)
'''
#Stacking
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
#弱學習器
estimators = [
    ('xgb', GridSearchCV(xgbModel, parameters_to_search, cv=5)),
    ('svr', make_pipeline(StandardScaler(), SVR(C = svr_c, epsilon = svr_epsilon))),
    ('rf', RandomForestRegressor(random_state = 408570344))
]
#Stacking將不同模型優缺點進行加權，讓模型更好。
#final_estimator：集合所有弱學習器訓練出最終預測模型。預設為LogisticRegression。
stackModel = StackingRegressor(
    estimators=estimators, final_estimator= MLPRegressor(activation = "relu", alpha = 0.1, hidden_layer_sizes = (8,8),
                            learning_rate = "constant", max_iter = 200, random_state = 100)
)
stackModel.fit(X_train_std, y_train.values.ravel())
#stackScore = stackModel.score(X_test_std, y_test.values.ravel())
#print("Correct rate after Stacking: ", stackScore)

#Output predict data
result = pd.DataFrame([], columns=['Id', 'Category'])
result['Id'] = [f'{i:03d}' for i in range(len(test))]
result['Category'] = stackModel.predict(test_std).astype(int)
result.to_csv("data/predict.csv", index = False)
