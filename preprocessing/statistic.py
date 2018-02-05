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

# 4. 選特徵值(排序由大到小用圖表觀查)
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,14), var_exp, alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best') 
plt.show()

# 5. 將原始資料轉化為新特徵值
# 用特徵值排序取得特徵向量
eigen_pars= [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)
# 將前兩個特徵向量合併成13*2矩陣以利之後原始資料轉換成2個新特徵
# array[:,np.newaxis] 將原陣列多一 y 軸，(3) -> (3,1)
# np.hstack(a,b) 第一軸相加 --> (3,) (3,) --> (6,)  || (2,3) (3,3) --> (5,3) 第二軸向量數要相同
# np.vstack(a,b) 第二軸相加 --> (3,) (3,) --> (6,)  || (3,3) (3,2) --> (3,5) 第一軸向量數要相同
w = np.hstack((eigen_pairs[0][1][:,np.newaxis]),(eigen_pairs[1][1][:,np.newaxis]))
# 進行pca轉換 , 全部資料變2維
X_train_pca = X_train_std.dot(w)
colors= ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], X_traint_pca[y_train==l, 1], c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()    
