import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def data_standardization(df_input):
    sc = StandardScaler()   
    df=sc.fit_transform(df_input.iloc[:,0:10])
    return df

warnings.filterwarnings('ignore')

#Read data
data = pd.read_csv('data/project1_train.csv')
test = pd.read_csv('data/project1_test.csv')

#Revise Male, Female
data.loc[data.Gender=='Male', 'Gender'] = 1
data.loc[data.Gender=='Female', 'Gender'] = 0
test.loc[test.Gender=='Male', 'Gender'] = 1
test.loc[test.Gender=='Female', 'Gender'] = 0

#data = data.drop(data[data['Albumin_and_Globulin_Ratio'].isnull()].index)
from sklearn.impute import SimpleImputer
# 建立以平均值填補缺損值的實體
imp = SimpleImputer(strategy='most_frequent')  #mean, median, most_frequent
# 填補缺損值
imp.fit(data)
data2 = imp.transform(data)
data = pd.DataFrame(data2, index = data.index, columns = data.columns)

'''
#Find analysis target
X_df = pd.DataFrame(data, columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio'])
y_df = pd.DataFrame(data, columns = ['Label'])
'''
#Standardization
X_df = data_standardization(data)
y_df = data['Label'].astype('int')
'''
#Delete nan
y_df = y_df.drop(X_df[X_df['Albumin_and_Globulin_Ratio'].isnull()].index)
X_df = X_df.drop(X_df[X_df['Albumin_and_Globulin_Ratio'].isnull()].index)
'''

#Dvide the data into validation and test sets
X_train , X_val , y_train , y_val = train_test_split(X_df ,y_df , test_size=0.3 , random_state=408570344)
#X_train, X_valid, y_train, y_valid = train_test_split(X_df, y_df, train_size=0.6, random_state=0)
#X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, train_size=0.5, random_state=0)

'''
#Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)
X_test_std = sc.transform(X_test)

#Change numpy array to pandas
X_train_std = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
X_valid_std = pd.DataFrame(X_valid_std, index=X_valid.index, columns=X_valid.columns)
X_test_std = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns)
'''
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
#sc_test = StandardScaler()
#sc_test.fit(test)
sc = StandardScaler() 
test_std = sc.fit_transform(test)
#test_std = pd.DataFrame(test_std, index=test.index, columns=test.columns)

#XGBoost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
'''
n_estimators = [10,20,30,40,50,60,70,80,90,100]
max_depth = [1,2,3,4,5,6]
parameters_to_search = {'n_estimators': n_estimators, 
              'max_depth': max_depth} #設定要訓練的值
xgbModel = xgb.XGBRegressor(n_estimators = 50, max_depth = 6)
xgbModel_cv = GridSearchCV(xgbModel, parameters_to_search, cv=5) #可以直接找出最佳的訓練值
xgbModel_cv.fit(X_train_std, y_train.values.ravel())
xgbScore = xgbModel_cv.score(X_test_std, y_test.values.ravel())
print('Correct rate using XGBoost: {:.5f}'.format(xgbScore))
'''
'''
xgbModel_Classifier = xgb.XGBClassifier(n_estimators = 50, max_depth = 6)
xgbModel_Classifier.fit(X_train_std, y_train.values.ravel())
xgbScore_Classifier = xgbModel_Classifier.score(X_test_std, y_test.values.ravel())
print('Correct rate using XGBoost: {:.5f}'.format(xgbScore_Classifier))
'''
'''
params = { 'max_depth': [3,6,10],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [100, 500, 1000],
           'colsample_bytree': [0.3, 0.7]}

xg2 = xgb.XGBClassifier(random_state=408570344)
clf = GridSearchCV(estimator = xg2,
                   param_grid = params,
                   scoring = 'neg_mean_squared_error',
                   verbose=1)
clf.fit(X_train_std, y_train)

print("Best parameters:", clf.best_params_)
'''
'''
xg3 = xgb.XGBClassifier(colsample_bytree= 0.3, learning_rate=0.01, max_depth= 10, n_estimators=100)
xg3=xg3.fit(X_train_std, y_train)
xgbScore_Classifier = xg3.score(X_test_std, y_test)
print('Correct rate using XGBoost: {:.5f}'.format(xgbScore_Classifier))
'''
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
svrModel.fit(X_train_std, y_train)
for i in range(1, 5):
    for j in range(0, 5):
        svrModel = make_pipeline(StandardScaler(), SVR(C=i, epsilon = j/10))
        svrModel.fit(X_train_std, y_train.values.ravel())
        if svr_maxRate < svrModel.score(X_valid_std, y_valid):
            svr_maxRate = svrModel.score(X_valid_std, y_valid)
            svr_c = i
            svr_epsilon = j/10
svrModel = make_pipeline(StandardScaler(), SVR(C = svr_c, epsilon = svr_epsilon))
svrModel.fit(X_train_std, y_train)
svrScore = svrModel.score(X_test_std, y_test)
print('Correct rate using SVR: {:.5f}'.format(svrScore))
'''

#SVM
from sklearn import svm
# 建立 linearSvc 模型
#polyModel=svm.SVC(kernel='poly', degree=3, gamma='auto', C=1)
#polyModel=svm.SVC(kernel='rbf', gamma=0.7, C=1)
# 使用訓練資料訓練模型
max_pred = 0
degree_max = 5 #1
C_max = 2 #1
'''
for i in range (1,11):
    for j in range (1, 11):
        #svm.SVC(kernel='poly', degree=i, gamma='auto', C=j)
        svm.SVC(kernel='rbf', degree=i, C=j)
        polyModel.fit(X_train_std, y_train)
        if polyModel.score(X_valid_std, y_valid) > max_pred:
            max_pred = polyModel.score(X_valid_std, y_valid)
            degree_max = i
            C_max = j
#print(degree_max, C_max)
#svm.SVC(kernel='poly', degree=degree_max, gamma='auto', C=C_max)
svm.SVC(kernel='rbf', degree=degree_max, C=C_max)
polyModel.fit(X_train_std, y_train)
svmScore = polyModel.score(X_test_std, y_test)
print('Correct rate using SVR: {:.5f}'.format(svmScore))
'''

# Random Forest
from sklearn.ensemble import RandomForestClassifier
'''
rf_maxRate = 0
rf_state = 0
for i in range (1,10):
    rfModel = RandomForestClassifier(random_state = i)
    rfModel.fit(X_train_std, y_train.values.ravel())
    if rf_maxRate < rfModel.score(X_valid_std, y_valid.values.ravel()):
        rf_maxRate = rfModel.score(X_valid_std, y_valid.values.ravel())
        rf_state = i
'''
max_pred = 0
n_max = 200 #100
depth_max = 4 #3
'''
rfModel = RandomForestClassifier(n_estimators=200, max_depth=3, random_state = 408570344)
for i in range (100, 2000, 200):
    for j in range(2, 10):
        rfModel = RandomForestClassifier(n_estimators=i, max_depth=j, random_state = 408570344)
        rfModel.fit(X_train_std, y_train)
        if rfModel.score(X_valid_std, y_valid) > max_pred:
            max_pred = rfModel.score(X_valid_std, y_valid)
            depth_max = j
            n_max = i
'''
'''
#rfModel = RandomForestClassifier(n_estimators=n_max, max_depth=depth_max, random_state = 408570344)
rfModel = RandomForestClassifier(n_estimators=n_max, max_depth=depth_max, min_samples_split=3, random_state = 408570344)
rfModel.fit(X_train_std, y_train)
rfScore = round(rfModel.score(X_test_std, y_test),5)
print("Correct rate using Random Forest: ", rfScore)
'''
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
'''
model = MLPClassifier(
  hidden_layer_sizes=(10),
  max_iter=10,
  solver="adam",
  random_state=100002     #random_state請改成你的學號（純數字、不加B）
)
model.fit(X_train, y_train)
print("Training set score: %f" % model.score(X_train, y_train))
print("Validation set score: %f" % model.score(X_val, y_val))
kfold = KFold(n_splits=10, shuffle=True, random_state=100002)
score = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy').mean()
param_grid = [ {'hidden_layer_sizes':[(10,), (20,), (50,), (5,5), (10,10)], 
          'max_iter':[5, 10, 20, 50],
          'solver': ['sgd', 'adam'],
          }]

MLP_kfold = KFold(n_splits=10, shuffle=True, random_state=100002)
grid = GridSearchCV(MLPClassifier(), param_grid, cv=MLP_kfold, scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
#{'hidden_layer_sizes': (50,), 'max_iter': 50, 'solver': 'sgd'}
#MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, solver='sgd')

MLP_tuned = MLPClassifier(
  hidden_layer_sizes=(50,),
  max_iter=50,
  solver="sgd",
  random_state=408570344     #random_state請改成你的學號（純數字、不加B）
)
after_GridSearch_score = cross_val_score(MLP_tuned, X_train, y_train, cv=MLP_kfold, scoring='accuracy').mean()
print("After tuned accuracy: {}".format(after_GridSearch_score))
'''
#Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
#弱學習器

xgb_params = { 'max_depth': 1,
           'learning_rate': 0.03,
           'n_estimators': 200,
           'colsample_bytree': 0.3,
           'random_state': 408570344}
mlp_params = {'hidden_layer_sizes': (50,),
              'max_iter': 50,
              'solver': "sgd",
              'random_state': 408570344 }

estimators = [
    ('xgb', XGBClassifier(**xgb_params)),
    ('svc', svm.SVC(kernel='rbf', degree=degree_max, C=C_max, random_state=408570344)),
    ('rf', RandomForestClassifier(n_estimators=n_max, max_depth=depth_max, min_samples_split=2, random_state = 408570344)),
    ('dt', DecisionTreeClassifier(max_depth=depth_max, min_samples_split=2, random_state=408570344)),
    ('knn', KNeighborsClassifier(n_neighbors=2)),
    ('mlp', MLPClassifier(**mlp_params))
]
#Stacking將不同模型優缺點進行加權，讓模型更好。
#final_estimator：集合所有弱學習器訓練出最終預測模型。預設為LogisticRegression。
'''
stackModel = StackingClassifier(
    estimators=estimators, final_estimator= MLPClassifier(activation = "relu", alpha = 1, hidden_layer_sizes = (3,3),
                            learning_rate = "constant", max_iter = 20, random_state = 408570344)
)
'''
stackModel = StackingClassifier(estimators = estimators,
#                                final_estimator = ,
                                stack_method = 'predict')
stackModel.fit(X_train, y_train)
stackScore = stackModel.score(X_val, y_val)
print("Correct rate after Stacking: ", stackScore)

#Output predict data
result = pd.DataFrame([], columns=['Id', 'Category'])
result['Id'] = [f'{i:03d}' for i in range(len(test))]
result['Category'] = stackModel.predict(test_std).astype(int)
result.to_csv("data/predict.csv", index = False)
