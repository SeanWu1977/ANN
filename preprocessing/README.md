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
df.isnull()
       A      B      C      D
0  False  False  False  False
1   True  False   True  False
2  False  False  False  False
3  False  False  False   True

```
