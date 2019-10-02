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
η => 學習率
調太大，容易錯過最低點的L(w)，甚至會越算越大。
調太小，需要執行很多次(epoch)才能找到最低點的L(w)，沒效率。
建議畫圖來看(x: epoch, y:L(w))趨勢。

另外也可以改變η，前幾輪比較大，後來慢慢變小。

* adgrad
  每輪的η會依據上一輪的η以及所有至本輪的(dL/dw)決定
  ηt = η/(t)^.5
  σ = ( (Σ(dL/dw)^2)/(t) + ε )^.5   (ε:未必免分母會0，是一個很小的數)  
  wt = w[t-1] - (ηt/σ)*(dL/dw)
  如此下來，越後輪，η會越小，w差異也會越小。
  
 * adam 也是，用此方法沿深出來。目前大多用adam
  改善學習率到後面幾乎小到0 導致無學習。
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

==> 如每次取n筆，共取m輪，
每輪算出來的yp曲線會比較平滑(variance會較小)。



```
4. bias & variance
```
bias : (中心點)偏移
variance : 分散程度

a) variance : 分散程度
模型越複雜，越容易受資料影響，用不同資料算出來的模型會差越大(variance大)。
ex:
f(x)=a
不論x的集合如何，預測結果f(x)的結果都一樣，variance=0
集合X1計算出來的f(x)=a
集合X2計算出來的f(x)=a
variance=0

f(x)=b+wx
集合X1計算出來的f(x)=b1+w1x
集合X2計算出來的f(x)=b2+w2x
variance>=0


b) bias: 所有預測值的平均與真實資料的平均差
假設已知真實預測線_f(x)，每次取n個點來預測，共執行m次。
將每次的f(x)加平均：f_a(x)
當模型越複雜，
> 雖然f(x)間的variance大，
> 但f_a(x)會越接近_f(x)，
> 即bias越小。


c) 結論
模型 | bias | variance 
----------------------
簡單 |  大  |    小     | 可多加feature或把模型用的更複雜   | 訓練資料跟測試資料都不準 | underfitting : error 來自於 bias 
複雜 |  小  |    大     | 增加訓練資料，或用regularization | 訓練資料準，測試資料不準 | overfitting  : error 來自於 variance
要在bias 與 variance之間取一個平衡點

另可用Cross validation


```
