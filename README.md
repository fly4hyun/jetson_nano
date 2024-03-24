
pip install opencv-python
pip install ultralytics
pip install fastapi
pip install uvicorn

############################

sudo apt update
sudo apt upgrade

sudo apt install build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev libffi-dev libc6-dev

cd /
wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz

tar -xf Python-3.8.12.tar.xz
cd Python-3.8.12

./configure --enable-optimizations
make -j4

sudo make altinstall
python3.8 --version

python3.8 -m venv yolo_cam                                     
source yolo_cam/bin/activate

############################

cd /etc/init.d
vim start.sh 

###
#!/bin/bash
..(etc)

chmod +x start.sh
update-rc.d start.sh defaults
