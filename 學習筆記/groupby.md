# groupby 用法
``` python

# 1. 依照 "Neighborhood" groupby
# 2. 每個 "Neighborhood" group 取出 "LotFrontage"
# 3. 每個 "Neighborhood" group 計算 "LotFrontage" 的 median
# 4. 每個 "Neighborhood" group 的 "LotFrontage" 執行fillna
# x 指的是 "LotFrontage"
df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

```
