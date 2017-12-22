from sklearn import datasets
import numpy as np
import plot_decision_regions as pdr
import matplotlib.pyplot as plt

# 將資料分為訓練集與測試集函數
from sklearn.model_selection import train_test_split

# 將資料正規化函數
from sklearn.preprocessing import StandardScaler

# 感知器函數
from sklearn.svm import SVC 

# 效能指標模組
# accuracy_score 算正確率
from sklearn.metrics import accuracy_score

# 使用scikit-learn中內建的訓練資料
# 鳶尾花數據
iris = datasets.load_iris() 
X = iris.data[:, [2,3]]
y = iris.target

# 將資料分為訓練集(70%)與測試集(30%)
# random_state 0 表示不執行seed(), 即每次執行順序會不同
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 建立正規化物件
sc = StandardScaler()

# 利用現有資料計算正規化所需的平均值與標準差
# 不執行此指令則之後的動作都無法操作
sc.fit(X_train) 

# 查平均值與標準差，回傳陣列[平均值，標準差]
sc.mean_

# 利用sc.fit算出來的平均值與標準差來正規化訓練集與測試集
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 模型訓練
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

# 預測
y_pred = svm.predict(X_test_std)
print("預測準確率 : {:.2%}".format(accuracy_score(y_test, y_pred)))


# 繪圖
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
pdr.plot_decision_regions(X=X_combined_std, y=y_combined,
                          classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc="upper left")
plt.show()



# 不同C來控制錯誤分的的懲罰(penalty)
'''
Specifies the kernel type to be used in the algorithm.
It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
If none is given, ‘rbf’ will be used.
If a callable is given it is used to pre-compute the kernel matrix from data matrices;
that matrix should be an array of shape (n_samples, n_samples)
'''    
for c in np.arange(0.00001, 0.0001, 0.00001, dtype=float):
    svm = SVC(kernel='linear', C=c, random_state=0)
    svm.fit(X_train_std, y_train)
    print("預測準確率 : {:.2%}".format(accuracy_score(y_test, y_pred)))
