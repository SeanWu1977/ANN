# 資料預處理

``` python 
import pandas as pd
from io import StringIO
csv_data = '''
A,B,C,D
1,2,3,4
,4,,5
3,6,33,77
123,66,33,'''
 
df=pd.read_csv(StringIO(csv_data))

# 判斷欄為是否為null
df.isnull()
       A      B      C      D
0  False  False  False  False
1   True  False   True  False
2  False  False  False  False
3  False  False  False   True

(1,1)(1,2)(1,3)
(2,1)(2,2)(2,3)

# 同行加總
df.sum()
A    127.0
B     78.0
C     69.0
D     86.0
# 同列加總
df.sum(axis=1)
0     10.0
1      9.0
2    119.0
3    222.0

# 將dataframe 轉為 array (pandas -> numpy)
df.values
array([[   1.,    2.,    3.,    4.],
       [  nan,    4.,   nan,    5.],
       [   3.,    6.,   33.,   77.],
       [ 123.,   66.,   33.,   nan]])
       
       
       

############ 預處理一，刪除欄位方法  ##############
# 刪除有Nan值的行(記錄)
df.dropna()
     A  B     C     D
0  1.0  2   3.0   4.0
2  3.0  6  33.0  77.0
# 刪除有Nan值的列(欄位)
df.dropna(axis=1)
    B
0   2
1   4
2   6
3  66

# 只刪除所有欄位均為NaN之記綠
df.dropna(how='all')

# 只保留至少有4個欄位有值的記錄
df.dropna(thresh=4)

# 只考慮欄位"C"有NaN時，刪除該筆記錄
df.dropna(subset=['C'])


############ 預處理二，插補技術(補值 interpolation techniques) ##############

from sklearn.preprocessing import Imputer

# 設定補值的策略  strategy = mean | median | most_frequent 
imr = Imputer(missing_values='NaN', strategy='mean',axis=0) 

# 從輸入值學習，產生NaN要換的值
imr = imr.fit(df)
# 將NaN取代
imputed_date = imr.transform(df.values)

# 以上兩個動作(fit & transform)可用以下一個動作取代，缺點是每一次都要執行一次學習(資料量大效能會不好)。
imputed_date = imr.fit_transform(df)

```
