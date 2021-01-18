# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
sudo apt update -y
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get install python3-pip -y
sudo apt-get install python3.7 -y
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
sudo apt install python-minimal -y
sudo apt-get install yasm -y
pip3 install gitpython

# FFmpeg
sudo apt-get -y install libv4l-dev
pushd /usr/include/linux
sudo ln -s ../libv4l1-videodev.h videodev.h
popd

# qemu
sudo apt-get -y install --fix-missing git libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev
sudo apt-get -y install --fix-missing git-email
sudo apt-get -y install --fix-missing libaio-dev libbluetooth-dev libbrlapi-dev libbz2-dev
sudo apt-get -y install --fix-missing libcap-dev libcap-ng-dev libcurl4-gnutls-dev libgtk-3-dev
sudo apt-get -y install --fix-missing libibverbs-dev libjpeg8-dev libncurses5-dev libnuma-dev
sudo apt-get -y install --fix-missing librbd-dev librdmacm-dev
sudo apt-get -y install --fix-missing libsasl2-dev libsdl1.2-dev libseccomp-dev libsnappy-dev libssh2-1-dev
sudo apt-get -y install --fix-missing libvde-dev libvdeplug-dev libvte-2.90-dev libxen-dev liblzo2-dev
sudo apt-get -y install --fix-missing valgrind xfslibs-dev 
sudo apt-get -y install --fix-missing libnfs-dev libiscsi-dev

# main file
nohup python3 src2asm.py --parallel_num 20 --optimization_level 1 --dataset FFmpeg > ffmpeg.log 2>&1 &
nohup python3 src2asm.py --parallel_num 20 --optimization_level 1 --dataset qemu > qemu.log 2>&1 &
