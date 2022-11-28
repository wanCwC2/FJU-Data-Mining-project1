#Imbalanced Data
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
'''
#Draw pie chart
import matplotlib.pyplot as plt
plt.figure( figsize=(10,5) )
data['Label'].value_counts().plot( kind='pie', colors=['lightcoral','skyblue'], autopct='%1.2f%%' )
plt.title('Label')
plt.ylabel('')
plt.show()
'''
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

X_df = pd.DataFrame(X_df, columns = data.columns[0:10])
y_df = pd.DataFrame(y_df)

#Dvide the data into validation and test sets
X_train , X_test , y_train , y_test = train_test_split(X_df ,y_df , test_size=0.1 , random_state=408570344)

#SMOTE process imbalanced data
#from imblearn.over_sampling import SMOTE
#import matplotlib.pyplot as plt
#X_train, y_train = SMOTE().fit_resample(X_train, y_train)
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report

X_re, y_re = SMOTE(random_state=42).fit_resample(X_train, y_train)
X_rere, y_rere = TomekLinks().fit_resample(X_re, y_re)
'''
plt.figure( figsize=(10,5) )
y_train['Label'].value_counts().plot( kind='pie', colors=['lightcoral','skyblue'], autopct='%1.2f%%' )
plt.title('Label')
plt.ylabel('')
plt.show()
'''
'''
import sklearn.metrics
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)
from collections import Counter
print(sorted(Counter(y_train).items()))
'''
