# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
sudo mkfs.ext3 /dev/nvme1n1
sudo mkfs.ext4 -E nodiscard /dev/nvme0n1
sudo mount /dev/nvme0n1 /home/ubuntu/
df -h
