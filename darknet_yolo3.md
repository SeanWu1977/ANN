```
cd ~

# 裡面已經有標準的config檔
git clone https://github.com/pjreddie/darknet
cd darknet
make


# 下載其它人訓練好的資料集
wget https://pjreddie.com/media/files/yolov3.weights

# 執行後，看~/darknet/predictions.jpg
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg

# 臉部辨識的configure file & weight file
# configure file 下載至 ~/darknet/cfg
https://raw.githubusercontent.com/sthanhng/yoloface/master/cfg/yolov3-face.cfg

# weight file 下載至 ~/darknet
https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view
or
wget https://doc-04-5o-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/lbparta50784utpmun2m4qaj6ers4nlr/1566187200000/08669234846404299145/*/1xYasjU52whXMLT5MtF7RCPQkV66993oR?e=download


# 下載圖片至~/darknet/data
https://github.com/sthanhng/yoloface/blob/master/assets/outside_000001_yoloface.jpg


# 執行後，看~/darknet/predictions.jpg
./darknet detect cfg/yolov3-face.cfg yolov3-wider_16000.weights data/outside_000001.jpg 


# yolo3 by pytorch
# https://github.com/ayooshkathuria/pytorch-yolo-v3
#

```
