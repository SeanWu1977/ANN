from sklearn import datasets
import numpy as np
import plot_decision_regions as pdr
import matplotlib.pyplot as plt
import graphviz 

# 將資料分為訓練集與測試集函數
from sklearn.model_selection import train_test_split

# 將資料正規化函數
from sklearn.preprocessing import StandardScaler

# 決策樹
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.tree import export_graphviz as eg
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestClassifier as rfc

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
##sc = StandardScaler()

# 利用現有資料計算正規化所需的平均值與標準差
# 不執行此指令則之後的動作都無法操作
##sc.fit(X_train) 

# 查平均值與標準差，回傳陣列[平均值，標準差]
##sc.mean_

# 利用sc.fit算出來的平均值與標準差來正規化訓練集與測試集
##X_train_std = sc.transform(X_train)
##X_test_std = sc.transform(X_test)

# 模型訓練{'entropy' or 'gini'(default)}
tree = dt(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)


# 預測
y_pred = tree.predict(X_test)
print("預測準確率 : {:.2%}".format(accuracy_score(y_test, y_pred)))


# 繪圖
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plt.figure('Decision Tree')
pdr.plot_decision_regions(X=X_combined, y=y_combined,
                          classifier=tree, test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc="upper left")
plt.show()

# 將決策數視覺化
dot_data = eg(tree, out_file='tree.dot', filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dot_data)


##eg(tree, out_file='tree.dot',feature_names=['petal length','petal width']
##)

# cmd下執行dot -Tpng tree.dot -o tree.png

# n_estimators = 隨機樹數量  ，  n_jobs cpu使用量
forest = rfc(criterion='entropy', n_jobs=2, random_state=1, n_estimators=10)
forest.fit(X_train, y_train)
y_pred = tree.predict(X_test)
plt.figure('Random Forest')
print("預測準確率 : {:.2%}".format(accuracy_score(y_test, y_pred)))
pdr.plot_decision_regions(X=X_combined, y=y_combined,
                          classifier=forest, test_idx=range(105,150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc="upper left")

plt.show()



clf = dtr()
clf.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("預測準確率 : {:.2%}".format(accuracy_score(y_test, y_pred)))
plt.figure('Decision Tree Regression')
pdr.plot_decision_regions(X=X_combined, y=y_combined,
                          classifier=clf, test_idx=range(105,150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc="upper left")

plt.show()
