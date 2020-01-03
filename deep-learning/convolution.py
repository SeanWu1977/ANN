
# 看每個filter轉出來的圖

cnn1 = Sequential()
cnn1.add(Conv2D(6, (3, 3), 
               padding="same", 
               input_shape=(32, 32, 3)))
x = cnn1.predict(x_train)
plt.figure(figsize=(5, 5))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.axis("off")
    plt.imshow(x[1][:,:,i],cmap="gray")
    
    
cnn2 = Sequential()
cnn2.add(MaxPooling2D(input_shape=(32, 32, 3),pool_size=(2, 2)))
y = cnn2.predict(x_train)
plt.imshow(y[1].astype(int))
