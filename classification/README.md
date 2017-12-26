## 程式檔說明

1. perceptron.py 單一感知器(AND / OR / XOR)模擬 

2.1 perceptron_cls.py 單一感知器, 類別方式建構
> 每次用一筆訓練資料來計算weight 


2.2 iris.py 用鳶尾花數據模擬單一感知器(類別)

3.1 Adaline.py 適應性神經元建構
> 用成本函數來計算 <br>
> 每次用所有訓練資料來計算weight 


3.2 iris_adaline.py 用鳶尾花數據模擬適應性神經元

4.1 AdalineSGD.py 適應性神經元建構-隨機梯度下降法(資料量過大時訓練用原方法效率
> 資料量過大時訓練用原方法訓練會很耗資源，因此改用此法
> 用成本函數來計算 <br>
> 每次用所有訓練資料來計算weight 


4.2 iris_adalineSGD.py 用鳶尾花數據模擬適應性神經元

5.1 skl_perceptron.py 用scikit-learn package模擬感知器(鳶尾花數據)<br>
5.2 skl_logisticRegression.py 用scikit-learn package模擬邏輯迴歸模型感知器(鳶尾花數據)<br>
5.3 skl_SVM.py 用scikit-learn package模擬support vector Machine(鳶尾花數據)<br>



6. 決策樹學習(用資訊增益information gain)<br>
6.1 impurity.py 不同的impurity方法(不純度)<br>
6.2 iris_decisiontree.py 用scikit-learn package模擬決策樹學習(鳶尾花數據)<br>
    tree.png GraphViz將結果視覺化
    

7. skl_neighbors.py 使用KNN(k-nearest neighbor classifier) 分類演算法
> 找距離最近的n個點，看那個分類比較多就是屬於該類

***
***
* 公用函式
> plot_decision_regions.py 畫色塊函數




