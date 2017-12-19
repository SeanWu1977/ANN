import numpy as np
from matplotlib import pyplot as plt
from decimal import *



def per(X1,X2,Yd,w1,w2,s,a):
    loop = True
    i=1
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{} ".format('時間','輸入','輸入',' 期望','權重','權重',' 實際','誤差',' 最終',' 最終'))
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{} ".format('  ',' X1',' X2','輸出Yd',' w1',' w2','輸出Y',' e','權重w1','權重w2'))
    while loop:
        inner_loop_e_nozero = False  #判斷每一輪是否有e不等於0

        for p in range(4):
            x1=Decimal(X1[p]).quantize(Decimal('.1'), rounding=ROUND_DOWN)
            x2=Decimal(X2[p]).quantize(Decimal('.1'), rounding=ROUND_DOWN)
            yd=Decimal(Yd[p]).quantize(Decimal('.1'), rounding=ROUND_DOWN)
            Sum=x1*w1+x2*w2-s
    
            if Sum >= 0:
                Y=Decimal(1.0).quantize(Decimal('.1'), rounding=ROUND_DOWN)
            else:
                Y=Decimal(0.0).quantize(Decimal('.1'), rounding=ROUND_DOWN)
            e=yd-Y
            w11=w1 + Decimal(a*x1*e).quantize(Decimal('.1'), rounding=ROUND_DOWN)
            w22=w2 + Decimal(a*x2*e).quantize(Decimal('.1'), rounding=ROUND_DOWN)
            if not inner_loop_e_nozero and e !=0 :
                inner_loop_e_nozero=True
            print("{0:1d}\t{1:.1f}\t{2:.1f}\t{3:.1f}\t{4:.1f}\t{5:.1f}\t{6:.1f}\t{7:.1f}\t{8:.1f}\t{9:.1f} ".format(i,x1,x2,yd,w1,w2,Y,e,w11,w22))
            w1=w11
            w2=w22
        if inner_loop_e_nozero == False :
            loop = False
        i+=1
        if i >100 :
            loop = False
        print("{}{}{}{}{}{}{}{}{}{} ".format('--------','--------','--------','--------','--------','--------','--------','--------','--------','--------'))        

    if i>100:
        print("Can not calculate!")
    else:
        print("Y={:2.1f} * X1 + {:2.1f} * X2 - {}".format(w1,w2,s))
        print("Y >= 0\tYd=1 \nY < 0\tYd=0")
        printfg(w1,w2,s)


def printfg(w1,w2,s):

    x1=[]
    x2=[]
    for i in range(5):
        
        x1.append(Decimal(i).quantize(Decimal('.1'), rounding=ROUND_DOWN))
        x2.append((s-x1[i]*w1)/w2)
    plt.plot(x1,x2)
    plt.show()

X1=np.array([0.,0.,1.,1.])
X2=np.array([0.,1.,0.,1.])

w1=Decimal(0.101).quantize(Decimal('.1'), rounding=ROUND_DOWN)
w2=Decimal(0.101).quantize(Decimal('.1'), rounding=ROUND_DOWN)
s=Decimal(0.201).quantize(Decimal('.1'), rounding=ROUND_DOWN)
a=Decimal(0.101).quantize(Decimal('.1'), rounding=ROUND_DOWN)
print("{}{}{}{}{}{}{}{}{}{} ".format('========','========','========','========','---AND--','========','========','========','========','========'))    
Yd=np.array([0.,0.,0.,1.])
per(X1,X2,Yd,w1,w2,s,a)



print("{}{}{}{}{}{}{}{}{}{} ".format('========','========','========','========','---OR---','========','========','========','========','========'))    
Yd=np.array([0.,1.,1.,1.])
per(X1,X2,Yd,w1,w2,s,a)



print("{}{}{}{}{}{}{}{}{}{} ".format('========','========','========','========','---XOR--','========','========','========','========','========'))    
Yd=np.array([0.,1.,1.,0.])
per(X1,X2,Yd,w1,w2,s,a)
