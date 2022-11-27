import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
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

data['Gender'].replace({'Male': 1, 'Female': 0}, inplace = True)
test['Gender'].replace({'Male': 1, 'Female': 0}, inplace = True)

from sklearn.impute import SimpleImputer
# 建立以平均值填補缺損值的實體
imp = SimpleImputer(strategy='most_frequent')  #mean, median, most_frequent
# 填補缺損值
imp.fit(data)
data2 = imp.transform(data)
data = pd.DataFrame(data2, index = data.index, columns = data.columns)

#Outliers
from scipy.stats import boxcox
out = ['Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio'] 
for i in out:
    transform_data = data[i]**(1/3)
#    transform_data2 = test[i]**(1/3)
#    transform_data, lam = boxcox(data[i])
    transform_data2, lam = boxcox(test[i])
    for j in range(0, data.shape[0]):
        data.loc[j, i] = transform_data[j]
    for j in range(0, test.shape[0]):
        test.loc[j, i] = transform_data2[j]
des = data.describe()
des2 = test.describe()

#Standardization
X_df = data_standardization(data.iloc[:,0:10])
y_df = data['Label'].astype('int')
test = data_standardization(test)

#Dvide the data into validation and test sets
X_train , X_test , y_train , y_test = train_test_split(X_df ,y_df , test_size=0.3 , random_state=408570344)

#best_k = KNN(X_train, X_test, y_train, y_test).findK()
'''
parameters = {'n_neighbors':[1,3,5,7,9,11,13]}
knn = KNeighborsClassifier() #注意：這裡不用指定引數
#通過GridSearchCV來搜尋最好的K值。這個模組的內部其實就是對每一個K值進行評估
clf = GridSearchCV(knn,parameters,cv=5) #5折
clf.fit(X_train,y_train)
#輸出最好的引數以及對應的準確率
#print("The best rate is ：%.5f"%clf.best_score_,"The best value of k is",clf.best_params_)
print("The best value of k is",clf.best_params_)
'''
model = KNeighborsClassifier(n_neighbors=1)
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
