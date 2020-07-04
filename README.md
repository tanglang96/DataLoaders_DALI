# PyTorch DataLoaders with DALI

PyTorch DataLoaders implemented with [nvidia-dali](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html), we've implemented **CIFAR-10** and **ImageNet** dataloaders, more dataloaders will be added in the future.

With 2 processors of Intel(R) Xeon(R) Gold 6154 CPU, 1 Tesla V100 GPU and all dataset in memory disk, we can **extremely** **accelerate image preprocessing** with DALI.

| Iter Training Data Cost(bs=256) | CIFAR-10 | ImageNet |
| :-----------------------------: | :------: | :------: |
|              DALI               |   1.4s(2 processors)   | 625s(8 processors)  |
|           torchvision           |  280.1s(2 processors)  | 13400s(8 processors)  |

In CIFAR-10 training, we can reduce tranining time **from** **1 day to 1 hour** with our hardware setting.

## Requirements

You only need to install nvidia-dali package and version should be >= 0.12, we've tested version 0.11 and it didn't work

```bash
#for cuda9.0
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali
#for cuda10.0
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
```

More details and documents can be found [here](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html#)

## Usage

You can use these dataloaders easily as the following example

```python
from base import DALIDataloader
from cifar10 import HybridTrainPipe_CIFAR
pip_train = HybridTrainPipe_CIFAR(batch_size=TRAIN_BS,
                                  num_threads=NUM_WORKERS,
                                  device_id=0, 
                                  data_dir=IMG_DIR, 
                                  crop=CROP_SIZE, 
                                  world_size=1, 
                                  local_rank=0, 
                                  cutout=0)
train_loader = DALIDataloader(pipeline=pip_train,
                              size=CIFAR_IMAGES_NUM_TRAIN, 
                              batch_size=TRAIN_BS, 
                              onehot_label=True)
for i, data in enumerate(train_loader): # Using it just like PyTorch dataloader
    images = data[0].cuda(non_blocking=True)
    labels = data[1].cuda(non_blocking=True)
```

If you have large enough memory for storing dataset, we strongly recommend you to mount a memory disk and put the whole dataset in it to accelerate I/O, like this

```bash
mount  -t tmpfs -o size=20g  tmpfs /userhome/memory_data
```

It's noteworthy that `20g` above is a ceiling but **not** occupying `20g` memory at the moment you mount the tmpfs, memories are occupied as you putting dataset in it. Compressed files should **not** be extracted before you've copied them into memory, otherwise it could be much slower.
