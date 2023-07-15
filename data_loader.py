import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
from norb import smallNORBViewPoint, smallNORB
from aff_nist import affNIST


def get_train_valid_loader(data_dir,
                           dataset,
                           batch_size,
                           random_seed,
                           exp='azimuth',
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=0,
                           pin_memory=False):

    data_dir = data_dir + '/' + dataset

    if dataset == "cifar10":
        trans = [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                transform=transforms.Compose(trans))
        #a=2

    elif dataset == "svhn":
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        trans = [transforms.RandomCrop(32, padding=4),
                 transforms.ToTensor(),
                 normalize]
        dataset = datasets.SVHN(data_dir, split='train', download=True,
                transform=transforms.Compose(trans))

    elif dataset == "smallnorb":
        trans = [transforms.Resize(48),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(brightness=32./255, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.7199,), (0.117,))
                ]
        if exp in VIEWPOINT_EXPS:
            train_set = smallNORBViewPoint(data_dir, exp=exp, train=True, download=True,
                    transform=transforms.Compose(trans))
            trans = trans[:1] + [transforms.CenterCrop(32)]  +trans[3:]
            valid_set = smallNORBViewPoint(data_dir, exp=exp, train=False, familiar=False, download=False,
                    transform=transforms.Compose(trans))
        elif exp == "full":
            dataset = smallNORB(data_dir, train=True, download=True,
                                           transform=transforms.Compose(trans))
    elif dataset =="cifar100":
        trans = [
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ]

        dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                   transform=transforms.Compose(trans))
    elif dataset == "fashionmnist":

        trans = [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]

        dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                    transform=transforms.Compose(trans))

    elif dataset == 'Affnist':
        trans = [#transforms.Resize(32),
                 transforms.ToTensor(),
                 # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        dataset = affNIST(data_dir, is_Train=True, transform=transforms.Compose(trans))

    elif dataset == 'mnist':
        trans =[transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]
        dataset = datasets.MNIST(data_dir, train=True, download=True,transform=transforms.Compose(trans))


    if exp not in VIEWPOINT_EXPS:
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx = indices[split:]
        valid_idx = indices[:split]

        train_set = Subset(dataset, train_idx)
        valid_set = Subset(dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader

def get_test_loader(data_dir,
                    dataset,
                    batch_size,
                    exp='azimuth',
                    familiar=True,
                    num_workers=4,
                    pin_memory=False):

    data_dir = data_dir + '/' + dataset

    if dataset == "cifar10":
        trans = [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        dataset = datasets.CIFAR10(data_dir, train=False, download=False,
                transform=transforms.Compose(trans))

    elif dataset == "svhn":
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        trans = [transforms.ToTensor(),
                 normalize]
        dataset = datasets.SVHN(data_dir, split='test', download=True,
                transform=transforms.Compose(trans))

    elif dataset == "smallnorb":
        trans = [transforms.Resize(48),
                 transforms.CenterCrop(32),
                 transforms.ToTensor(),
                 transforms.Normalize((0.7199,), (0.117,))
                 ]
        if exp in VIEWPOINT_EXPS:
            dataset = smallNORBViewPoint(data_dir, exp=exp, familiar=familiar, train=False, download=True,
                                transform=transforms.Compose(trans))
        elif exp == "full":
            dataset = smallNORB(data_dir, train=False, download=True,
                                transform=transforms.Compose(trans))
    elif dataset == "cifar100":
        trans = [transforms.ToTensor(),
                 transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                                      np.array([63.0, 62.1, 66.7]) / 255.0),
        ]

        dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                    transform=transforms.Compose(trans))
    elif dataset == "fashionmnist":
        trans = [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
        ]
        dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                    transform=transforms.Compose(trans))
    elif dataset == "mnist":
        trans = [transforms.Resize(32),
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,)),
                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                 ]
        #dataset = affNIST('/home/zhao/CR/datasets/data/Affnist', is_Train=False, transform=transforms.Compose(trans))
        dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.Compose(trans))


    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},
    'smallnorb': {'size': 32, 'channels': 1, 'classes': 5},
    'cifar100': {'size':32, 'channels': 3, 'classes': 100},
    'fashionmnist':{'size':28, 'channels':1, 'classes':10},
    'Affnist':{'size':40, 'channels':1, 'classes':10},
    'mnist':{'size':28, 'channels':1,'classes':10},
}

VIEWPOINT_EXPS = ['azimuth', 'elevation']
