# GR-CapsNet
# MF Routing

## Environment
- torch 1.6.0
- python 3.7
- torchvision 0.7.0
- tensorboard 2.2.0
- tensorboardX 2.1.0
- ml-collections 0.1.0
- medpy 0.4.0
- SimpleITK 2.0.2
- scipy 1.5.3 
- h5py 2.10.0
- CUDA >= 10.2 supported GPU

## train
- MF Routing

To train a model for CIFAR-10 or other datasets, run:
```
cd MF Routing
python train.py --epochs 350 --lr 0.002 --dataset [cifar10] --method [train] --resume False --op_votes [MF]
```
```cifar10 ```should be one of [cifar10, svhn, fashion-mnist]. ```MF ```should be one of [EM, ME, MF]. ```train ```should be one of [train, test].
- MF-CapsUNet

Run the train script on synapse dataset.
```
python train.py --dataset [Synapse] --max_epochs 1000 --batch_size 12 --base_lr 0.0002
```
```Synapse ```should be one of [Synapse, ADCD].
## test
- MF Routing

To test a model, simply run:
```
cd MF Routing
python train.py --dataset [cifar10] --method [test] --resume False --op_votes [MF]
```
- MF-CapsUNet

Run the test script on synapse dataset. 
```
python test.py --dataset Synapse --max_epochs 1000 --batch_size 12 --base_lr 0.0002
```
```Synapse ```should be one of [Synapse, ADCD].
# References

CapsNet:https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch

Unet:https://github.com/Beckschen/TransUNet;https://github.com/HuCaoFighting/Swin-Unet
