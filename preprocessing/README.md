# 資料預處理
1. 補缺失值(impute)


2. 分類數據(categorical data)
    * 名目特徵 (normal feature) --> 虛擬特微 (dummy feature)
    * 有序特徵 (ordinal feature)

3. 特微縮放(feature scaling)
    * 常態化 (normalization) : 值位於 0 ~ 1 , 對離群值較敏感
    * 標準化 (standardization) : 轉為平均值為 0 ，標準差為 1 , 對離群值較不敏感

4. overfitting
    * 對訓練集預測很準，但實際資料不準
    * 降維
      - 資料正規化(降低偏值影響) : LogisticRegression 利用調整懲罰項 (L2 -> L1 penalty，會產生稀疏權重) , 再調整正規化值達到降維
      - 特徵選擇 (feature selection) : 現有特徵選擇一個子集合
        - 循序特徵選擇(sequential feature selection) : 用於不支援正規化的演算法
          - 循序向後選擇(sequential back selection, SBS) : 每輪移除一個影響準確率(準則函數, criterion function)最少的特徵，不斷重覆此步驟直到設定特徵數。 
      - 特徵提取 (feature extraction) : 用現有特徵建構新特徵，可保留最多相關資訊的數據壓縮方法
         - 主成份分析(PCA，非監督式)
         - 線性判別分析(LCA,監督式)
    * 使用較少參數，做出較簡單模型
    * 收集更多的訓練數據集
