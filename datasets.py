import torch
from torchvision import datasets, transforms

def get_dataset(dir, name):
    if name == 'cifar':
        # Set two conversion formats
        # transforms.Compose is a combination of multiple transforms (a list of transforms)
        transform_train = transforms.Compose([
            # transforms.RandomCrop： 切割中心点的位置随机选取
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize： 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                                         transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

    return train_dataset, eval_dataset
