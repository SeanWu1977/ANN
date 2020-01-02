``` python 

from sklearn.model_selection import train_test_split, KFold, GridSearchCV

# 判斷結果好壞
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)

# 有label用dataframe輸出
label=["ant", "cat", "bird"]
pd.DataFrame(confusion_matrix(y_true, y_pred,labels=label),columns=label,index=label)

```
