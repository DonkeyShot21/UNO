import torchvision.transforms as transforms

from PIL import ImageFilter
import random


class DiscoveryTargetTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, y):
        y = self.mapping[y]
        return y


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MultiTransform:
    def __init__(self, times, transform):
        self.times = times
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.times)]


def get_transforms(mode, dataset, num_views=None):

    mean, std = {
        "CIFAR10": [(0.491, 0.482, 0.447), (0.202, 0.199, 0.201)],
        "CIFAR100": [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)],
        "ImageNet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    }[dataset]

    transform = {
        "ImageNet": {
            "unsupervised": MultiTransform(
                num_views,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224, (0.5, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply(
                            [transforms.ColorJitter(0.3, 0.3, 0.15, 0.1)], p=0.5
                        ),
                        transforms.RandomGrayscale(p=0.1),
                        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                ),
            ),
            "eval": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        },
        "CIFAR100": {
            "unsupervised": MultiTransform(
                num_views,
                transforms.Compose(
                    [
                        transforms.RandomChoice(
                            [
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomResizedCrop(32, (0.8, 1.0)),
                            ]
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply(
                            [transforms.ColorJitter(0.3, 0.3, 0.15, 0.1)], p=0.5
                        ),
                        transforms.RandomGrayscale(p=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                ),
            ),
            "supervised": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "eval": transforms.Compose(
                [
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        },
        "CIFAR10": {
            "unsupervised": MultiTransform(
                num_views,
                transforms.Compose(
                    [
                        transforms.RandomChoice(
                            [
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomResizedCrop(32, (0.8, 1.0)),
                            ]
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply(
                            [transforms.ColorJitter(0.3, 0.3, 0.15, 0.1)], p=0.5
                        ),
                        transforms.RandomGrayscale(p=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                ),
            ),
            "supervised": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "eval": transforms.Compose(
                [
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        },
    }[dataset][mode]

    return transform
