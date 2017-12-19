
import numpy as np
from matplotlib import pyplot as plt
from decimal import *
 
#getcontext().prec = 4 #設定湖點數精度4位
#getcontext().rounding = ROUND_HALF_UP #4捨5入
#print(getcontext())
def Fx2(x0,s,w1,w2):
 r2 = (s-x0*w1)/w2
 #print(r1)
 return r2
#最大疊代次數Pmax =預設1000次 意為最大訓練次數
Pmax=22 
 # 產生輸入一維矩陣 x1 ,序列為[0.,0.,1.,1.,0.,0.,1.,1.,0.,0.,1.,1.,........]
xi1=[0.,0.,1.,1.]
x1=xi1*(Pmax//4)
#print(x1)
# 產生輸入一維矩陣 x2 ,序列為[0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,........]
xi2=[0.,1.,0.,1.]
x2=xi2*(Pmax//4)      
#print(x2)               
#Yd ,為x1 and x2 的預期輸出 ,故每次疊代後的預期輸出應為序列[0.,0.,0.,1., 0.,0.,0.,1.,0.,0.,0.,1.,....]
Yt=[0.,0.,0.,1. ]
#Yd ,為x1 or x2 的預期輸出 ,故每次疊代後的預期輸出應為序列[0.,0.,0.,1., 0.,0.,0.,1.,0.,0.,0.,1.,....]
#Yt=[0.,1.,1.,1. ]
Yd=Yt*(Pmax//4)   
#print(Yd)
#初始權重w1,w2,及臨界值s=0.2 ,學習率a=0.1
w1=np.zeros(Pmax) #初始為0.0
w1[0]=0.3    #第一次疊代初始值為0.3 ,預設buffer為可疊代Pmax次
#print(w1[0:4]) 
w2=np.zeros(Pmax) #初始為0.0
w2[0]=-0.1    #第一次疊代初始值為-0.1 ,預設buffer為可疊代Pmax次
#print(w2[0:4]) 
s=0.2
a=0.1
#宣告初始権重差值矩陣為0,只是for程式設計使用預設值
#Dw 用來修正每次疊代後須修正的權值
Dw1 =np.zeros(Pmax) #初始為0.0
Dw2 =np.zeros(Pmax) #初始為0.0
#宣告初始誤差E為0,只是for程式設計使用預設值
#E 為每次疊代後,期望值與實際輸出值的誤差
E =np.zeros(Pmax) #初始為0.0
#宣告初始實際輸出值Y矩陣為0,只是for程式設計使用預設值
#第p次疊代實際輸出Y
Y =np.zeros(Pmax) #初始為0.0
#Epoch ,疊代次數p
print("疊代次數|輸入x1|輸入x2|初始權重w1|初始權重w2|期望輸出Yd|實際輸出Y|  誤差E  |修正後權重w1|修正後權重w2|")
for p in range(Pmax-2):  #from  0,1...4 to Pmax
 #print("疊代次數:",p)
 #由於浮點數計算會有誤差,所以使用Decimal的quantize只取兩位並無條件捨去
 Y[p]=Decimal(x1[p]*w1[p]+x2[p]*w2[p]-s).quantize(Decimal('.01'), rounding=ROUND_DOWN)
 #print("實際輸出before F(step):",Y[p])
 #代入步階函數
 if  Y[p]>=0.0:
  Y[p]=1.0
 else:
  Y[p]=0.0
 #print("實際輸出2:",Y[p])
 #計算誤差並修改權重
 E[p]=Decimal(Yd[p]-Y[p]).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
 #print("誤差:",E[p])
 Dw1[p]=a*x1[p]*E[p]
 w1[p+1]=Decimal(w1[p]+Dw1[p]).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
 Dw2[p]=a*x2[p]*E[p]
 w2[p+1]=Decimal(w2[p]+Dw2[p]).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
 #if  (E[p,0]==0)&(E[p,1]==0)&(E[p,2]==0)&(E[p,3]==0)   :
  #break
 #print("修正後權重1:",w1[p+1])
 #print("修正後權重2:",w2[p+1])
 #print("疊代次數|輸入x1|輸入x2|初始權重w1|初始權重w2|期望輸出Yd|實際輸出Y|  誤差E  |修正後權重w1|修正後權重w2|")
 print('{0:1d} {1:1d} {2:1d} {3:.2f}{4:.2f} {5:1d} {6:1d} {7:.2f} {8:.2f} {9:.2f} '.format(p, int(x1[p]),int(x2[p]),w1[p],w2[p],int(Yd[p]),int(Y[p]),E[p],w1[p+1],w2[p+1]))
 
#得到最後訓練好的w1,w2 
print("得到最後訓練好的權重w1,w2=",w1[p+1],w2[p+1])
print('代入原輸出方程式Y= {0:.2f}X1+ {1:.2f}X2- {2:.2f} '.format(w1[p+1],w2[p+1],s))

