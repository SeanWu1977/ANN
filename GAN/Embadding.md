```
input_array = np.random.randint(255, size=(1, 3))
#輸入為1組，有3個值(1維)

model.add(Embedding(255, 64, input_length=3))
# 輸出為1組，每組有3個值，每個值有64維



input_array = np.random.randint(255, size=(3, 4))
#輸入為3組，有4個值(1維)

model.add(Embedding(255, 64, input_length=4))
# 輸出為3組，每組有4個值，每個值有64維


# input_length 不寫為 1
# 以下兩個input_length均為1，keras會自動轉換
[0,0,0,0]
[[0],[0],[0],[0]]


```
