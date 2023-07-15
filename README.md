# GR-CapsNet
## train
To train a model for CIFAR-10 or other datasets, run:
```
python main.py --dataset=[cifar10] --name=resnet_[method] --epochs=400 --is_train=True
```
```cifar10 ```should be one of [cifar10, svhn, fashion-mnist, AffNIST, CIFAR-100, UCF101, GTSRB, MNIST]. ```method ```should be one of [global_routing, em_routing].
## test
To test a model, simply run:
```
python main.py --dataset=[cifar10] --name=resnet_[method] --epochs=400 --is_train=False
```
```cifar10 ```should be one of [cifar10, svhn, fashion-mnist, AffNIST, CIFAR-100, UCF101, GTSRB, MNIST]. ```method ```should be one of [global_routing, em_routing].
