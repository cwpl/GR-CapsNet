# GR-CapsNet
## train
To train a model for CIFAR-10 or other datasets, run:
```
python main.py --dataset=[cifar10] --name=resnet_[method] --epochs=400
```
```cifar10 ```should be one of [cifar10, svhn, fashion-mnist, AffNIST, CIFAR-100, UCF101, GTSRB, MNIST]. ```method ```should be one of [GR, DR, EM]. ```train ```should be one of [train, test].
## test
To test a model, simply run:
```
python train.py --dataset [cifar10] --method [test] --resume False --op_votes [MF]
```
