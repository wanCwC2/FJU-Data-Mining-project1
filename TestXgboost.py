import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_standardization(df_input):
    sc = StandardScaler()   
    df=sc.fit_transform(df_input)
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

#Outliers
from scipy.stats import boxcox
out = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio'] 
for i in out:
    transform_data = data[i]**(1/3)
    transform_data2 = test[i]**(1/3)
    for j in range(0, data.shape[0]):
        data.loc[j, i] = transform_data[j]
    for j in range(0, test.shape[0]):
        test.loc[j, i] = transform_data2[j]

from sklearn.impute import SimpleImputer
# 建立以平均值填補缺損值的實體
imp = SimpleImputer(strategy='most_frequent')  #mean, median, most_frequent
# 填補缺損值
imp.fit(data)
data2 = imp.transform(data)
data = pd.DataFrame(data2, index = data.index, columns = data.columns)

#Standardization
X_df = data_standardization(data.iloc[:,0:10])
y_df = data['Label'].astype('int')
test = data_standardization(test)

#Dvide the data into validation and test sets
X_train , X_test , y_train , y_test = train_test_split(X_df ,y_df , test_size=0.1 , random_state=408570344)

#XGBoost
from xgboost import XGBClassifier

# declare parameters
#Find best parameters
#About run time 1.5 hours
'''
params = { 'max_depth': range (2, 15, 3),
           'learning_rate': [0.01, 0.1, 0.5, 1, 5, 10],
           'n_estimators': range(80, 500, 50),
           'colsample_bytree': [0.5, 1, 3, 6, 10],
           'min_child_weigh': range(1, 9, 1),
           'subsample': [0.5, 0.7, 0.9, 1.5, 2]}

from sklearn.model_selection import GridSearchCV
xg2 = XGBClassifier()
clf = GridSearchCV(estimator = xg2,
                   param_grid = params,
                   scoring = 'neg_log_loss')
clf.fit(X_train, y_train)

print("Best parameters:", clf.best_params_)
'''

params = {'colsample_bytree': 0.5,
          'learning_rate': 0.01,
          'max_depth': 5,
          'min_child_weigh': 1,
          'n_estimators': 280,
          'subsample': 0.7,
          'random_state': 408570344}

# instantiate the classifier 
#model = clf.best_estimator_
model = XGBClassifier(**params)
# fit the classifier to the training data
model.fit(X_train, y_train)

print("Training accuracy score: ", model.score(X_train, y_train))

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print('Testing accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred))) #0.6596

#Output predict data
result = pd.DataFrame([], columns=['Id', 'Category'])
result['Id'] = [f'{i:03d}' for i in range(len(test))]
result['Category'] = model.predict(test).astype(int)
result.to_csv("data/predict.csv", index = False)

import matplotlib.pyplot as plt
import xgboost as xgb
xgb.plot_importance(model)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()