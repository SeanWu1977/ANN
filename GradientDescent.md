1. 針對權重(w)進行調整，決定要如何調整
2. 目標函數(loss function) L(w)
```
* 最簡單的loss function = Σ(y-yp)^2
* 要取最小，即預測值跟原值差最少
假設 yp = b + Σxw

▽=dL/dw == 算出斜率
當dL/dw > 0，則 w 要減少
當dL/dw < 0，則 w 要增加
dL/dw = 0，完美(但可能不是最低local optimal，有雙谷，但liner 不會有雙谷)

η == 學習率

w1 = w0 - η*(dL/dw0)
此處用(-)是因為斜率(dL/dw0)是(-)要增加w0
反之亦然。

此處也有斜率越陡，調整越多的效果。

=====================
同理，L(w,b)時，
可各別針對w,b進行調整。


=====================
進一步
yp = w1x + w2*(x^2)
...

每一組方程式(不同階的方程式)yp跟實際y的誤差(loss function)，
就可得到最好的線性方程式。

但：
model越複雜對training data計算出的loss function會越低(overfitting)，
但對testing data不一定會有一樣的趨勢。
所以要選一個最適合的model.

```
3. regularization 
```
在原本的L(w)中，在加入λΣw^2
L(w) = Σ(y-yp)^2 + λΣw^2
要L(w)越小，代表w也要越小，增加w的影響力。

如此L(w)也比較平滑，即輸入改變時，影響不大(雜訊影響不大)
yp = b + Σxw
當w小時，x有變化時，yp的變化也小。

```
4. bais & variance
```
bais : (中心點)偏移
variance : 分散程度
模型越複雜，越容易受資料影響，用不同資料算出來的模型會差越大(overfitting)。
```
