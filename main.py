import torch
import time
from torchvision import datasets, transforms

from trainer import Trainer
from config import get_config
from utils import prepare_dirs
from data_loader import get_test_loader, get_train_valid_loader, VIEWPOINT_EXPS


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config):

    prepare_dirs(config)

    torch.manual_seed(config.random_seed)
    kwargs = {}
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': False}

    if config.is_train:
        data_loader = get_train_valid_loader(
            config.data_dir, config.dataset, config.batch_size,
            config.random_seed, config.exp, config.valid_size,
            config.shuffle, **kwargs
        )
    else:
        data_loader = get_test_loader(
            config.data_dir, config.dataset, config.batch_size, config.exp, config.familiar,
            **kwargs
        )

    trainer = Trainer(config, data_loader)

    if config.is_train:
            #tic = time.time()
            trainer.train()
            #toc = time.time()
            #print("Train time: {:1f}s".format(toc-tic))
    else:
        if config.attack:
            trainer.test_attack()
        else:
            if config.view:
                trainer.test_view()
            else:
                trainer.test()

if __name__ == '__main__':
    with torch.cuda.device(2):
        config, unparsed = get_config()
        main(config)
