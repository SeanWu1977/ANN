# Python in virtualenv

1. Install Python 3.x
2. 建立工作資料夾<br />
 `mkdir d:\workspace`
3. 生成虛擬環境<br />
 `cd d:\workspace`    # 切換至工作目錄<br />
 `virtualenv venv`    # 產生虛擬環境venv<br />
4. 啟用虛擬環境<br />
 `venv\Scripts\activate.bat`    # 啟用指令<br />
 `(venv) D:\workspace\venv\`    # 代表已進入虛擬環境
5. 虛擬環境內操作<br />
 `START /B python -m idlelib.idle`    # 在虛擬環境中啟用python IDLE(視窗模式)<br />
 `pip install numpy scipy scikit-learn matplotlib pandas`    # 安裝套件<br />
    * numpy 維度陣列與矩陣運算<br />
    * scipy 演算法、數學工具(迴歸、微積分..)<br />
    * scikit-learn 機器學習<br />
    * matplotlib 繪圖<br />
    * pandas 提供資料格式(Data Frame)來進行分析資料<br />


<br /><br /><br /><br /><br />


## 程式檔說明
```diff
- perceptron.py 單一感知器(AND / OR / XOR)模擬 
```



