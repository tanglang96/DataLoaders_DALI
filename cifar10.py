import os
import sys
import time
import torch
import pickle
import numpy as np
import nvidia.dali.ops as ops
from base import DALIDataloader
import nvidia.dali.types as types
from sklearn.utils import shuffle
from torchvision.datasets import CIFAR10
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
CIFAR_IMAGES_NUM_TRAIN = 50000
CIFAR_IMAGES_NUM_TEST = 10000
IMG_DIR = '/userhome/data/cifar10'
TRAIN_BS = 256
TEST_BS = 200
NUM_WORKERS = 4
CROP_SIZE = 32

class HybridTrainPipe_CIFAR(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop=32, dali_cpu=False, local_rank=0,
                 world_size=1,
                 cutout=0):
        super(HybridTrainPipe_CIFAR, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.iterator = iter(CIFAR_INPUT_ITER(batch_size, 'train', root=data_dir))
        dali_device = "gpu"
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.pad = ops.Paste(device=dali_device, ratio=1.25, fill_value=0)
        self.uniform = ops.Uniform(range=(0., 1.))
        self.crop = ops.Crop(device=dali_device, crop_h=crop, crop_w=crop)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.],
                                            std=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
                                            )
        self.coin = ops.CoinFlip(probability=0.5)

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images, layout="HWC")
        self.feed_input(self.labels, labels)

    def define_graph(self):
        rng = self.coin()
        self.jpegs = self.input()
        self.labels = self.input_label()
        output = self.jpegs
        output = self.pad(output.gpu())
        output = self.crop(output, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        output = self.cmnp(output, mirror=rng)
        return [output, self.labels]


class HybridTestPipe_CIFAR(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridTestPipe_CIFAR, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.iterator = iter(CIFAR_INPUT_ITER(batch_size, 'val', root=data_dir))
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.],
                                            std=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
                                            )

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images, layout="HWC")  # can only in HWC order
        self.feed_input(self.labels, labels)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        output = self.jpegs
        output = self.cmnp(output.gpu())
        return [output, self.labels]


class CIFAR_INPUT_ITER():
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, batch_size, type='train', root='/userhome/memory_data/cifar10'):
        self.root = root
        self.batch_size = batch_size
        self.train = (type == 'train')
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.targets = np.vstack(self.targets)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        np.save("cifar.npy", self.data)
        self.data = np.load('cifar.npy')  # to serialize, increase locality

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            if self.train and self.i % self.n == 0:
                self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__

if __name__ == '__main__':
    # iteration of DALI dataloader
    pip_train = HybridTrainPipe_CIFAR(batch_size=TRAIN_BS, num_threads=NUM_WORKERS, device_id=0, data_dir=IMG_DIR, crop=CROP_SIZE, world_size=1, local_rank=0, cutout=0)
    pip_test = HybridTestPipe_CIFAR(batch_size=TEST_BS, num_threads=NUM_WORKERS, device_id=0, data_dir=IMG_DIR, crop=CROP_SIZE, size=CROP_SIZE, world_size=1, local_rank=0)
    train_loader = DALIDataloader(pipeline=pip_train, size=CIFAR_IMAGES_NUM_TRAIN, batch_size=TRAIN_BS, onehot_label=True)
    test_loader = DALIDataloader(pipeline=pip_test, size=CIFAR_IMAGES_NUM_TEST, batch_size=TEST_BS, onehot_label=True)
    print("[DALI] train dataloader length: %d"%len(train_loader))
    print('[DALI] start iterate train dataloader')
    start = time.time()
    for i, data in enumerate(train_loader):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
    end = time.time()
    train_time = end-start
    print('[DALI] end train dataloader iteration')

    print("[DALI] test dataloader length: %d"%len(test_loader))
    print('[DALI] start iterate test dataloader')
    start = time.time()
    for i, data in enumerate(test_loader):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
    end = time.time()
    test_time = end-start
    print('[DALI] end test dataloader iteration')
    print('[DALI] iteration time: %fs [train],  %fs [test]' % (train_time, test_time))


    # iteration of PyTorch dataloader
    transform_train = transforms.Compose([
        transforms.RandomCrop(CROP_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_dst = CIFAR10(root=IMG_DIR, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=TRAIN_BS, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_dst = CIFAR10(root=IMG_DIR, train=False, download=True, transform=transform_test)
    test_iter = torch.utils.data.DataLoader(test_dst, batch_size=TEST_BS, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    print("[PyTorch] train dataloader length: %d"%len(train_loader))
    print('[PyTorch] start iterate train dataloader')
    start = time.time()
    for i, data in enumerate(train_loader):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
    end = time.time()
    train_time = end-start
    print('[PyTorch] end train dataloader iteration')

    print("[PyTorch] test dataloader length: %d"%len(test_loader))
    print('[PyTorch] start iterate test dataloader')
    start = time.time()
    for i, data in enumerate(test_loader):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
    end = time.time()
    test_time = end-start
    print('[PyTorch] end test dataloader iteration')
    print('[PyTorch] iteration time: %fs [train],  %fs [test]' % (train_time, test_time))
