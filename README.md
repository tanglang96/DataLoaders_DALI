# PyTorch DataLoaders with DALI

PyTorch DataLoaders implemented with [nvidia-dali](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html), we've implemented **CIFAR-10** and **ImageNet** dataloaders, more dataloaders will be added in the future.

With 2 processors of Intel(R) Xeon(R) Gold 6154 CPU, 1 Tesla V100 GPU and all dataset in memory disk, we can **extremely** **boost image preprocessing** with DALI.

| Iter Training Data Cost(bs=256) | CIFAR-10 | ImageNet |
| :-----------------------------: | :------: | :------: |
|              DALI               |   1.4s   | testing  |
|           torchvision           |  280.1s  | testing  |

In CIFAR-10 training, we can reduce tranining time **from** **1 day to 1 hour** with our hardware setting.

## Requirements

You only need to install nvidia-dali package

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
from cifar10 import get_cifar_iter_dali
train_loader = get_cifar_iter_dali(type='train',
                                   image_dir='/userhome/memory_data/cifar10',                                              batch_size=256,num_threads=4)
for i, data in enumerate(train_loader):
    images = data[0]["data"].cuda(async=True)
    labels = data[0]["label"].squeeze().long().cuda(async=True)
```

