import os
import sys
import time
import torch
import pickle
import numpy as np
import nvidia.dali.ops as ops
from base import DALIDataloader
from torchvision import datasets
from sklearn.utils import shuffle
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.49139968, 0.48215827, 0.44653124]
IMAGENET_STD = [0.24703233, 0.24348505, 0.26158768]
IMAGENET_IMAGES_NUM_TRAIN = 1281167
IMAGENET_IMAGES_NUM_TEST = 50000
IMG_DIR = '/gdata/ImageNet2012'
TRAIN_BS = 256
TEST_BS = 200
NUM_WORKERS = 4
VAL_SIZE = 256
CROP_SIZE = 224

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


if __name__ == '__main__':
    # iteration of DALI dataloader
    pip_train = HybridTrainPipe(batch_size=TRAIN_BS, num_threads=NUM_WORKERS, device_id=0, data_dir=IMG_DIR+'/train', crop=CROP_SIZE, world_size=1, local_rank=0)
    pip_test = HybridValPipe(batch_size=TEST_BS, num_threads=NUM_WORKERS, device_id=0, data_dir=IMG_DIR+'/val', crop=CROP_SIZE, size=VAL_SIZE, world_size=1, local_rank=0)
    train_loader = DALIDataloader(pipeline=pip_train, size=IMAGENET_IMAGES_NUM_TRAIN, batch_size=TRAIN_BS, onehot_label=True)
    test_loader = DALIDataloader(pipeline=pip_test, size=IMAGENET_IMAGES_NUM_TEST, batch_size=TEST_BS, onehot_label=True)
    # print("[DALI] train dataloader length: %d"%len(train_loader))
    # print('[DALI] start iterate train dataloader')
    # start = time.time()
    # for i, data in enumerate(train_loader):
    #     images = data[0].cuda(non_blocking=True)
    #     labels = data[1].cuda(non_blocking=True)
    # end = time.time()
    # train_time = end-start
    # print('[DALI] end train dataloader iteration')

    print("[DALI] test dataloader length: %d"%len(test_loader))
    print('[DALI] start iterate test dataloader')
    start = time.time()
    for i, data in enumerate(test_loader):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
    end = time.time()
    test_time = end-start
    print('[DALI] end test dataloader iteration')
    # print('[DALI] iteration time: %fs [train],  %fs [test]' % (train_time, test_time))
    print('[DALI] iteration time: %fs [test]' % (test_time))


    # iteration of PyTorch dataloader
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.08, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dst = datasets.ImageFolder(IMG_DIR+'/train', transform_train)
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=TRAIN_BS, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    transform_test = transforms.Compose([
        transforms.Resize(VAL_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dst = datasets.ImageFolder(IMG_DIR+'/val', transform_test)
    test_iter = torch.utils.data.DataLoader(test_dst, batch_size=TEST_BS, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    # print("[PyTorch] train dataloader length: %d"%len(train_loader))
    # print('[PyTorch] start iterate train dataloader')
    # start = time.time()
    # for i, data in enumerate(train_loader):
    #     images = data[0].cuda(non_blocking=True)
    #     labels = data[1].cuda(non_blocking=True)
    # end = time.time()
    # train_time = end-start
    # print('[PyTorch] end train dataloader iteration')

    print("[PyTorch] test dataloader length: %d"%len(test_loader))
    print('[PyTorch] start iterate test dataloader')
    start = time.time()
    for i, data in enumerate(test_loader):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
    end = time.time()
    test_time = end-start
    print('[PyTorch] end test dataloader iteration')
    # print('[PyTorch] iteration time: %fs [train],  %fs [test]' % (train_time, test_time))
    print('[PyTorch] iteration time: %fs [test]' % (test_time))
