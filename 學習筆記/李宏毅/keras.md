```
tensorflow or theano  (主要是算微分，不是做ML)
         \    /
         keras (一個interface, 因為上述的工具太難用)


batch_size = n => 在進行訓練時，將一次算n個資料(forward)，接著用所有的資料的算gradiant descent & backward。
epoch => 執行 m 輪 (所有的資料都看過算一輪)
Stochastic gradient descent(sgd) = > batch size = 1
      
      
target => keras.utils.to_categorical()
```


```
how to deal with bad model ?
1. early stopping
2. regularization (smooth)
3. Dropout
4. New activation function
5. Adaptive learning rate



```
