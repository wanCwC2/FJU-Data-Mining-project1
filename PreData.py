import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_boston

#Read data
data = pd.read_csv('data/project1_train.csv')

des = data.describe()

out = ['Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase'] 

for i in out:
    plt.figure(figsize=(2,5))
    plt.boxplot(data[i],showmeans=True)
    plt.title(i)
    plt.show()

# skewness 與 kurtosis
for i in out:
    skewness = round(data[i].skew(), 2)
    kurtosis = round(data[i].kurt(), 2)
    print(f"偏度(Skewness): {skewness}, 峰度(Kurtosis): {kurtosis}")
    # 繪製分布圖
    sns.histplot(data[i], kde=True)
    plt.show()
#Alkaline_Phosphotase: 偏度(Skewness): 3.8, 峰度(Kurtosis): 18.36
#Alamine_Aminotransferase: 偏度(Skewness): 6.98, 峰度(Kurtosis): 59.36
#Aspartate_Aminotransferase: 偏度(Skewness): 10.74, 峰度(Kurtosis): 148.68

transform_data = data[out[0]]**(1/3)
# skewness 與 kurtosis
skewness = round(transform_data.skew(), 2)
kurtosis = round(transform_data.kurt(), 2)
print(f"偏度(Skewness): {skewness}, 峰度(Kurtosis): {kurtosis}")

# 繪製分布圖
sns.histplot(transform_data, kde=True)
plt.show()

from scipy.stats import boxcox
transform_data, lam = boxcox(data[out[1]])
transform_data = pd.DataFrame(transform_data, columns=[out[1]])[out[1]]
# skewness 與 kurtosis
skewness = round(transform_data.skew(), 2)
kurtosis = round(transform_data.kurt(), 2)
print(f"偏度(Skewness): {skewness}, 峰度(Kurtosis): {kurtosis}")
'''
for i in range(0, 467):
    data.loc[i, 'Alamine_Aminotransferase'] = transform_data[i]
'''

# 繪製分布圖
sns.histplot(transform_data, kde=True)
plt.show()
