import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import shutil
import math

from tqdm import tqdm
from utils import AverageMeter, save_config
from tensorboardX import SummaryWriter

from models import *
from loss import *
from data_loader import DATASET_CONFIGS

from attack import Attack, extract_adv_images



device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class Trainer(object):


    def __init__(self, config, data_loader):

        self.config = config

        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)

        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.
        self.counter = 0
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq

        self.attack_type = config.attack_type
        self.attack_eps = config.attack_eps
        self.targeted = config.targeted

        self.name = config.name

        self.dataset=config.dataset


        if config.name.endswith('em_routing'):
            self.mode = 'EM'
        elif config.name.endswith('global_routing'):
            self.mode = 'GR'
        else:
            raise NotImplementedError("Unknown model postfix")


        if config.name.startswith('resnet'):
            self.model = resnet20(config.planes, DATASET_CONFIGS[config.dataset], config.num_caps, config.caps_size, config.depth, mode=self.mode).to(device)
        else:
            raise NotImplementedError("Unknown model prefix")


        self.loss = nn.CrossEntropyLoss().to(device)
        if self.mode in ['EM', 'GR']:
            if config.dataset in ['cifar10', 'svhn']:
                print("using NLL loss")
                self.loss = nn.NLLLoss().to(device)
            elif config.dataset == "smallnorb":
                if self.mode == 'EM':
                    print("using EM loss")
                    self.loss = EmRoutingLoss(self.epochs).to(device)
                elif self.mode == 'GR':
                    print("using NLL loss")
                    self.loss = nn.NLLLoss().to(device)

        self.params = self.model.parameters()
        self.optimizer = optim.SGD(self.params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)  #Adam

        if config.dataset == "cifar10":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 250], gamma=0.1)
        elif config.dataset == "svhn":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 150], gamma=0.1)
        elif config.dataset == "smallnorb":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5,10,20,30,40,50,60], gamma=0.1)
        elif config.dataset == "cifar100":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[200,300], gamma=0.1)
        elif config.dataset == "fashionmnist":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100,150], gamma=0.1)
        elif config.dataset == "Affnist":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5,10,15,20,30], gamma=0.1)
        elif config.dataset == "mnist":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150,250,350,], gamma=0.1)
        save_config(self.name, self.config)
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            self.writer = SummaryWriter(tensorboard_dir)

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

    def train(self):

        if self.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = float(param_group['lr'])
                break

            print(
                    '\nEpoch: {}/{} - LR: {:.1e}'.format(epoch+1, self.epochs, lr)
            )

            train_loss, train_acc = self.train_one_epoch(epoch)

            with torch.no_grad():
                valid_loss, valid_acc = self.validate(epoch)


            msg1 = "train loss: {:.3f} - train acc: {:.3f}"
            msg2 = " - val loss: {:.3f} - val acc: {:.3f}"

            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.counter = 0
                msg2 += " [*]"

            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc))

            # check for improvement
            if not is_best:
                self.counter += 1

            self.scheduler.step()

            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'scheduler_state': self.scheduler.state_dict(),
                 'best_valid_acc': self.best_valid_acc
                 }, is_best
            )

        if self.use_tensorboard:
            self.writer.close()

        print(self.best_valid_acc)

    def train_one_epoch(self, epoch):

        self.model.train()

        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):

                x, y = x.to(device), y.to(device)

                b = x.shape[0]
                out = self.model(x)
                if isinstance(self.loss, EmRoutingLoss):
                    loss = self.loss(out, y, epoch=epoch)
                else:
                    loss = self.loss(out, y)

                pred = torch.max(out, 1)[1]
                correct = (pred == y).float()
                acc = 100 * (correct.sum() / len(y))

                losses.update(loss.data.item(), x.size()[0])
                accs.update(acc.data.item(), x.size()[0])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                toc = time.time()
                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc-tic), loss.data.item(), acc.data.item()
                        )
                    )
                )
                pbar.update(b)

                if self.use_tensorboard:
                    iteration = epoch*len(self.train_loader) + i
                    self.writer.add_scalar('train_loss', loss, iteration)
                    self.writer.add_scalar('train_acc', acc, iteration)

        return losses.avg, accs.avg

    def validate(self, epoch):

        self.model.eval()

        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(device), y.to(device)

            out = self.model(x)
            if isinstance(self.loss, EmRoutingLoss):
                loss = self.loss(out, y, epoch=epoch)
            else:
                loss = self.loss(out, y)

            pred = torch.max(out, 1)[1]
            correct = (pred == y).float()
            acc = 100 * (correct.sum() / len(y))

            losses.update(loss.data.item(), x.size()[0])
            accs.update(acc.data.item(), x.size()[0])

        if self.use_tensorboard:
            self.writer.add_scalar('valid_loss', losses.avg, epoch)
            self.writer.add_scalar('valid_acc', accs.avg, epoch)

        return losses.avg, accs.avg

    def test(self):

        correct = 0
        testend = time.time()
        test_time = AverageMeter()
        self.load_checkpoint(best=self.best)
        self.model.eval()

        for i, (x, y) in enumerate(self.test_loader):
            x, y = x.to(device), y.to(device)

            out = self.model(x)
            pred = torch.max(out, 1)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100. * correct.data.item()) / (self.num_test)
        error = 100 - perc
        test_time.update(time.time() - testend)
        print(
            '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%) Time: {:.3f}'.format(
                correct, self.num_test, perc, error, test_time.val)
        )

    def test_attack(self):
        correct = 0
        self.load_checkpoint(best=self.best)
        self.model.eval()

        attacker = Attack(self.model, self.loss, self.attack_type, self.attack_eps)
        adv_data, num_examples = extract_adv_images(attacker, self.test_loader, self.targeted, DATASET_CONFIGS[self.config.dataset]["classes"])

        with torch.no_grad():
            for i, (x, y) in enumerate(adv_data):
                x, y = x.to(device), y.to(device)

                out = self.model(x)

                pred = torch.max(out, 1)[1]
                correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        if self.targeted:
            success = correct
        else:
            success = num_examples - correct

        perc = (100. * success.data.item()) / (num_examples)

        print(
                '[*] Attack success rate ({}, targeted={}, eps={}): {}/{} ({:.2f}% - {:.2f}%)'.format(
                self.attack_type, self.targeted, self.attack_eps, success, num_examples, perc, 100. - perc)
        )

    def test_view(self):
        self.load_checkpoint(best=self.best)
        self.model.eval()

    def save_checkpoint(self, state, is_best):

        filename = self.name + '_' + self.dataset + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.name + '_' + self.dataset +  '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False):

        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.name + '_' + self.dataset + '_ckpt.pth.tar'
        if best:
            filename = self.name + '_' + self.dataset + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )

