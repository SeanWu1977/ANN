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


# Series vs numpy.array
# Series 只有一維 [[1,1,1],[1,1,1]] (2,)
# Numpy array 可有多維[[1,1,1],[1,1,1]] (2,3)


from keras.preprocessing.image import load_img # 依路徑讀檔

from keras.applications.vgg16 import preprocess_input 
# 用別人的model 記得要用他的preprocess_input
# 他會幫忙把輸入轉為該模型的輸入大小
imagenet_utils.preprocess_input(x, mode='tf', **kwargs)

# 三種 mode
if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None
