import logging

import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from . import randaugment

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)


def get_cifar10(root, num_labeled, num_classes):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        base_dataset.targets, num_labeled, num_classes)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFix(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    logger.info(f"#Labeled: {len(train_labeled_idxs)}"
                f" #Unlabeled: {len(train_unlabeled_idxs)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(root, num_labeled, num_classes):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    # transform_val = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    # ])
    transform_val = None
    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        base_dataset.targets, num_labeled, num_classes)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFix(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    logger.info(f"#Labeled: {len(train_labeled_idxs)}"
                f" #Unlabeled: {len(train_unlabeled_idxs)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(labels, num_labeled, num_classes):
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


def fast_collate(batch):
    print(type(batch[0][0]))
    exit()
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    assert len(targets) == batch_size
    tensor = torch.zeros(
        (batch_size, *batch[0][0].shape), dtype=torch.uint8)
    for i in range(batch_size):
        tensor[i].copy_(batch[i][0])
    return tensor, targets


def fast_collate_u(batch):
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    assert len(targets) == batch_size
    tensor = torch.zeros(
        (batch_size, *batch[0][0].shape), dtype=torch.uint8)
    for i in range(batch_size):
        tensor[i].copy_(batch[i][0])
    return tensor, targets


class TransformFix(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')
        ])
        self.strong = randaugment.RandAugCutout(n=2, m=10)
        # self.normalize = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std)
        # ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        # return self.normalize(weak), self.normalize(strong)
        return weak, strong


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class PrefetchedWrapper(object):
    def prefetched_loader(loader, mean, std, device):
        mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.to(device, non_blocking=True)
                next_target = next_target.to(device, non_blocking=True)
                next_input = next_input.float().sub_(mean).div_(std)

            if not first:
                yield inputs, targets
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            inputs = next_input
            targets = next_target

        yield inputs, targets

    def __init__(self, dataloader, mean, std, device):
        self.dataloader = dataloader
        self.epoch = 0
        self.mean = [x * 255 for x in mean]
        self.std = [x * 255 for x in std]
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader,
                                                   self.mean,
                                                   self.std,
                                                   self.device)


class PrefetchedWrapperX(object):
    def prefetched_loader(loader, mean, std, device):
        mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        stream = torch.cuda.Stream()
        first = True
        loader_iter = iter(loader)

        try:
            next_input, next_target = loader_iter.next()
        except:
            loader_iter = iter(loader)
            next_input, next_target = loader_iter.next()

        with torch.cuda.stream(stream):
            next_input = next_input.to(device, non_blocking=True)
            next_target = next_target.to(device, non_blocking=True)
            next_input = next_input.float().sub_(mean).div_(std)

            if not first:
                yield inputs, targets
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            inputs = next_input
            targets = next_target

        yield inputs, targets

    def __init__(self, dataloader, mean, std, device):
        self.dataloader = dataloader
        self.epoch = 0
        self.mean = [x * 255 for x in mean]
        self.std = [x * 255 for x in std]
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader,
                                                   self.mean,
                                                   self.std,
                                                   self.device)


class PrefetchedWrapperU(object):
    def prefetched_loader(loader, mean, std, device):
        mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        stream = torch.cuda.Stream()
        first = True
        loader_iter = iter(loader)

        try:
            next_input_w, next_input_s, _ = loader_iter.next()
        except:
            loader_iter = iter(loader)
            next_input_w, next_input_s, _ = loader_iter.next()

        with torch.cuda.stream(stream):
            next_input_w = next_input_w.to(device, non_blocking=True)
            next_input_s = next_input_s.to(device, non_blocking=True)
            next_input_w = next_input_w.float().sub_(mean).div_(std)
            next_input_s = next_input_s.float().sub_(mean).div_(std)

            if not first:
                yield input_w, input_s
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input_w, input_s = next_input_w, next_input_s

        yield input_w, input_s

    def __init__(self, dataloader, mean, std, device):
        self.dataloader = dataloader
        self.epoch = 0
        self.mean = [x * 255 for x in mean]
        self.std = [x * 255 for x in std]
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader,
                                                   self.mean,
                                                   self.std,
                                                   self.device)
