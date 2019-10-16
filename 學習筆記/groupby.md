# groupby 用法
``` python

# 1. 依照 "Neighborhood" groupby
# 2. 每個 "Neighborhood" group 取出 "LotFrontage"
# 3. 每個 "Neighborhood" group 計算 "LotFrontage" 的 median
# 4. 每個 "Neighborhood" group 的 "LotFrontage" 執行fillna
# x 指的是 "LotFrontage"
# select median(LotFrontage) from df groupby Neighborhood
Neighborhood | LotFrontage
      1              5
      2              7
      1              5
      3              8
      1              4
      2              7
      1              3
      2              9
      3              4
      3              8
      
df.groupby("Neighborhood")["LotFrontage"].median()
Neighborhood | LotFrontage
      1              5
      2              7
      3              8      
      
# 使用transform時，則會針對每一資料進行轉換
df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
Neighborhood | new column
      1              5
      2              7
      1              5
      3              8
      1              5 <
      2              7 
      1              5 <
      2              7 <
      3              8 <
      3              8

```
