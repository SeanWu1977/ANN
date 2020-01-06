# 只教懂(mlp)，不教看(cnn)
# cnn 主要是抓特徵
from keras.applications.vgg16 import VGG16
vgg = VGG16(include_top=False, input_shape=(224, 224, 3))
# include_top 是否要包含3 fully-connected layers at the top of the network, 即mlp
## 'Dense' is a name for a Fully connected 
# weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
# 建立model的兩種方法
# 1、  Sequential
# 2、  Model (有彈性)
from keras.models import Sequential, Model
vgg.input # tensor
vgg.output # tensor
