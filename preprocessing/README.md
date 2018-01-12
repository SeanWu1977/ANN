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
      - 資料正規化(降低偏值影響): LogisticRegression 利用調整懲罰項 (L2 -> L1 penalty，會產生稀疏權重) , 再調整正規化值達到降維
      - 特徵選擇 (feature selection)
      - 特徵提取 (feature extraction)
   * 使用較少參數，做出較簡單模型
   * 收集更多的訓練數據集
