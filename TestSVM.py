import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_standardization(df_input):
    sc = StandardScaler()   
    df=sc.fit_transform(df_input.iloc[:,0:10])
    return df

warnings.filterwarnings('ignore')

#Read data
data = pd.read_csv('data/project1_train.csv')
test = pd.read_csv('data/project1_test.csv')
#data = data.drop([29, 60, 297, 347])

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

#Standardization
X_df = data_standardization(data)
#X_df = data_standardization(X_df)
y_df = data['Label'].astype('int')

#Dvide the data into validation and test sets
X_train , X_test , y_train , y_test = train_test_split(X_df ,y_df , test_size=0.3 , random_state=408570344)

test = data_standardization(test)

from sklearn.svm import LinearSVC
params = { 'penalty': ["l1", "l2"],
           'loss': ["hinge", "squared_hinge"],
           'C': [1.0, 3.0, 6.0, 10.0],
           'max_iter': [1000, 3000, 6000, 10000]}

from sklearn.model_selection import GridSearchCV
model = LinearSVC()
clf = GridSearchCV(estimator = model,
                   param_grid = params,
#                   scoring = 'neg_mean_squared_error',
                   verbose=1)
clf.fit(X_train, y_train)

print("Best parameters:", clf.best_params_)
print(clf.best_estimator_)
#Best parameters: {'C': 10.0, 'loss': 'squared_hinge', 'max_iter': 6000, 'penalty': 'l2'}
#LinearSVC(C=10.0, max_iter=6000)

model = LinearSVC(C=10.0, max_iter=6000)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred))) #0.6950

#Output predict data
result = pd.DataFrame([], columns=['Id', 'Category'])
result['Id'] = [f'{i:03d}' for i in range(len(test))]
result['Category'] = model.predict(test).astype(int)
result.to_csv("data/predict.csv", index = False) #0.69
