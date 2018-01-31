# 主成分分析(principal component analysis,PCA)
#   將K維降至Z維(K>>Z)
#   Z=K*W 將一個K維記錄轉為一個Z維記錄

# 共變異數
# N 個數之間的相關度，如所有數都同時變大，則相關度大
# N 個數之間的相關度，如所有數都同時變小，則相關度小
# (x-mean(X))(y-mean(Y))/std(X)*std(Y)        X,Y為集合
# 值會落在   (完全不相關)-1 ~ 1(完全相關)   之間

# 共變異數矩陣, 每個a都是兩個集合的  共變異數
'''
    x1    x2    x3
x1  a11   a12   a13
x2  a21   a22   a23
x3  a31   a32   a33
'''


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df_wine = pd.read_csv('wine.data', header=None)
df_wine.columns = [
'class level',
'Alcohol',
'Malic acid',
'Ash',
'Alcalinity of ash',
'Magnesium',
'Total phenols',
'Flavanoids',
'Nonflavanoid phenols',
'Proanthocyanins',
'Color intensity',
'Hue',
'OD280/OD315 of diluted wines',
'Proline'
]

X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
# 資料切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# 1. 資料標準化，平均會為0
sc = StandardScaler()
sc.fit(X_train) 
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# 2. 共變異數矩陣
cov_mat = np.cov(X_train_std.T)
# 3. 共變異數矩陣中取 特徵值(主成分值，越大表越重要) 和 特徵向量 (以下2種方法都可)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

# 4. 選特徵值
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,14), var_exp, alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best') 
plt.show()
