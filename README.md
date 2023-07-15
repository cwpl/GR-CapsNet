# GR-CapsNet
## train
To train a model for CIFAR-10 or other datasets, run:
```
cd MF Routing
python train.py --epochs 350 --lr 0.002 --dataset [cifar10] --method [train] --resume False --op_votes [MF]
```
```cifar10 ```should be one of [cifar10, svhn, fashion-mnist]. ```MF ```should be one of [EM, ME, MF]. ```train ```should be one of [train, test].
## test
To test a model, simply run:
```
python train.py --dataset [cifar10] --method [test] --resume False --op_votes [MF]
```
