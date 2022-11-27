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
#data = data.drop([29, 60, 297, 347])
#des = data.describe()
#des2 = test.describe()

#Outliers
from scipy.stats import boxcox
out = ['Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase'] 
for i in out:
    #transform_data = data[i]**(1/3)
    #transform_data2 = test[i]**(1/3)
    transform_data, lam = boxcox(data[i])
    transform_data2, lam = boxcox(test[i])
    for j in range(0, data.shape[0]):
        data.loc[j, i] = transform_data[j]
    for j in range(0, test.shape[0]):
        test.loc[j, i] = transform_data2[j]
#des = data.describe()
#des2 = test.describe()

#Revise Male, Female
data.loc[data.Gender=='Male', 'Gender'] = 1
data.loc[data.Gender=='Female', 'Gender'] = 0
test.loc[test.Gender=='Male', 'Gender'] = 1
test.loc[test.Gender=='Female', 'Gender'] = 0

#from sklearn import preprocessing
#lbl = preprocessing.LabelEncoder()
#test['Gender'] = lbl.fit_transform(test['Gender'].astype(int))

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

#X_df = pd.DataFrame(X_df, columns = data.columns[0:10])
#y_df = pd.DataFrame(y_df)

#Dvide the data into validation and test sets
X_train , X_test , y_train , y_test = train_test_split(X_df ,y_df , test_size=0.3 , random_state=408570344)

'''
#crossvalidator
from sklearn.model_selection import KFold
KFold(n_splits=2, random_state=408570344, shuffle=False)
'''

#Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC

xgb_params = {'max_depth': 1,
           'learning_rate': 0.01,
           'n_estimators': 300,
           'colsample_bytree': 1,
           'random_state': 408570344}

dt_params = {'criterion': "entropy",
           'splitter': "random",
           'min_samples_leaf': 10,
           'max_depth': 3,
           'random_state': 408570344}

svm_params = {'C': 1.0, 
              'loss': 'squared_hinge', 
              'max_iter': 1000, 
              'penalty': 'l2',
              'random_state': 408570344}
rf_params = {'criterion': 'gini',
             'max_depth': 10,
             'min_samples_leaf': 3,
             'n_estimators': 500}
#弱學習器
estimators = [
    ('xgb', XGBClassifier(**xgb_params)),
    ('svc', LinearSVC(**svm_params)),
    ('rf', RandomForestClassifier(**rf_params)),
]
#Stacking將不同模型優缺點進行加權，讓模型更好。
#final_estimator：集合所有弱學習器訓練出最終預測模型。預設為LogisticRegression。
stackModel = StackingClassifier(estimators = estimators,
#                                final_estimator = LogisticRegression(),
                                stack_method = 'predict',
                                cv = 10, #crossvalidator
                                )
'''
stackModel.fit(X_train, y_train)
stackScore = stackModel.score(X_test, y_test)
print("Correct rate after Stacking: ", stackScore)
'''
#Output predict data
stackModel.fit(X_df, y_df)
result = pd.DataFrame([], columns=['Id', 'Category'])
result['Id'] = [f'{i:03d}' for i in range(len(test))]
result['Category'] = stackModel.predict(test).astype(int)
result.to_csv("data/predict.csv", index = False)
