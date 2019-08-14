``` 
sudo apt-get -y update

sudo apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    python3-pip 
    
sudo apt-get clean 

sudo rm -rf /tmp/* /var/tmp/*

cd ~ 

git clone https://github.com/ageitgey/face_recognition.git

cd ~ 

pip3 install setuptools

mkdir -p dlib 

git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ 

cd  dlib/ 

sudo python3 setup.py install --yes USE_AVX_INSTRUCTIONS

cd ~

cd face_recognition

pip3 install -r requirements.txt 

sudo python3 setup.py install

cd examples

python3 recognize_faces_in_pictures.py
```
