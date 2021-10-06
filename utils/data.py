import torch
import torchvision
import pytorch_lightning as pl

from utils.transforms import get_transforms
from utils.transforms import DiscoverTargetTransform

import numpy as np
import os


def get_datamodule(args, mode):
    if mode == "pretrain":
        if args.dataset == "ImageNet":
            return PretrainImageNetDataModule(args)
        else:
            return PretrainCIFARDataModule(args)
    elif mode == "discover":
        if args.dataset == "ImageNet":
            return DiscoverImageNetDataModule(args)
        else:
            return DiscoverCIFARDataModule(args)


class PretrainCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.dataset_class = getattr(torchvision.datasets, args.dataset)
        self.transform_train = get_transforms("unsupervised", args.dataset)
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes = range(self.num_labeled_classes)

        # train dataset
        self.train_dataset = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_train
        )
        train_indices_lab = np.where(
            np.isin(np.array(self.train_dataset.targets), labeled_classes)
        )[0]
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices_lab)

        # val datasets
        self.val_dataset = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val
        )
        val_indices_lab = np.where(np.isin(np.array(self.val_dataset.targets), labeled_classes))[0]
        self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices_lab)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )


class DiscoverCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.dataset_class = getattr(torchvision.datasets, args.dataset)
        self.transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
        )
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes = range(self.num_labeled_classes)
        unlabeled_classes = range(
            self.num_labeled_classes, self.num_labeled_classes + self.num_unlabeled_classes
        )

        # train dataset
        self.train_dataset = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_train
        )

        # val datasets
        val_dataset_train = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_val
        )
        val_dataset_test = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val
        )
        # unlabeled classes, train set
        val_indices_unlab_train = np.where(
            np.isin(np.array(val_dataset_train.targets), unlabeled_classes)
        )[0]
        val_subset_unlab_train = torch.utils.data.Subset(val_dataset_train, val_indices_unlab_train)
        # unlabeled classes, test set
        val_indices_unlab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), unlabeled_classes)
        )[0]
        val_subset_unlab_test = torch.utils.data.Subset(val_dataset_test, val_indices_unlab_test)
        # labeled classes, test set
        val_indices_lab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), labeled_classes)
        )[0]
        val_subset_lab_test = torch.utils.data.Subset(val_dataset_test, val_indices_lab_test)

        self.val_datasets = [val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for dataset in self.val_datasets
        ]


IMAGENET_CLASSES_118 = [
    "n01498041",
    "n01537544",
    "n01580077",
    "n01592084",
    "n01632777",
    "n01644373",
    "n01665541",
    "n01675722",
    "n01688243",
    "n01729977",
    "n01775062",
    "n01818515",
    "n01843383",
    "n01883070",
    "n01950731",
    "n02002724",
    "n02013706",
    "n02092339",
    "n02093256",
    "n02095314",
    "n02097130",
    "n02097298",
    "n02098413",
    "n02101388",
    "n02106382",
    "n02108089",
    "n02110063",
    "n02111129",
    "n02111500",
    "n02112350",
    "n02115913",
    "n02117135",
    "n02120505",
    "n02123045",
    "n02125311",
    "n02134084",
    "n02167151",
    "n02190166",
    "n02206856",
    "n02231487",
    "n02256656",
    "n02398521",
    "n02480855",
    "n02481823",
    "n02490219",
    "n02607072",
    "n02666196",
    "n02672831",
    "n02704792",
    "n02708093",
    "n02814533",
    "n02817516",
    "n02840245",
    "n02843684",
    "n02870880",
    "n02877765",
    "n02966193",
    "n03016953",
    "n03017168",
    "n03026506",
    "n03047690",
    "n03095699",
    "n03134739",
    "n03179701",
    "n03255030",
    "n03388183",
    "n03394916",
    "n03424325",
    "n03467068",
    "n03476684",
    "n03483316",
    "n03627232",
    "n03658185",
    "n03710193",
    "n03721384",
    "n03733131",
    "n03785016",
    "n03786901",
    "n03792972",
    "n03794056",
    "n03832673",
    "n03843555",
    "n03877472",
    "n03899768",
    "n03930313",
    "n03935335",
    "n03954731",
    "n03995372",
    "n04004767",
    "n04037443",
    "n04065272",
    "n04069434",
    "n04090263",
    "n04118538",
    "n04120489",
    "n04141975",
    "n04152593",
    "n04154565",
    "n04204347",
    "n04208210",
    "n04209133",
    "n04258138",
    "n04311004",
    "n04326547",
    "n04367480",
    "n04447861",
    "n04483307",
    "n04522168",
    "n04548280",
    "n04554684",
    "n04597913",
    "n04612504",
    "n07695742",
    "n07697313",
    "n07697537",
    "n07716906",
    "n12998815",
    "n13133613",
]

IMAGENET_CLASSES_30 = {
    "A": [
        "n01580077",
        "n01688243",
        "n01883070",
        "n02092339",
        "n02095314",
        "n02098413",
        "n02108089",
        "n02120505",
        "n02123045",
        "n02256656",
        "n02607072",
        "n02814533",
        "n02840245",
        "n02843684",
        "n02877765",
        "n03179701",
        "n03424325",
        "n03483316",
        "n03627232",
        "n03658185",
        "n03785016",
        "n03794056",
        "n03899768",
        "n04037443",
        "n04069434",
        "n04118538",
        "n04154565",
        "n04311004",
        "n04522168",
        "n07695742",
    ],
    "B": [
        "n01883070",
        "n02013706",
        "n02093256",
        "n02097130",
        "n02101388",
        "n02106382",
        "n02112350",
        "n02167151",
        "n02490219",
        "n02814533",
        "n02843684",
        "n02870880",
        "n03017168",
        "n03047690",
        "n03134739",
        "n03394916",
        "n03424325",
        "n03483316",
        "n03658185",
        "n03721384",
        "n03733131",
        "n03786901",
        "n03843555",
        "n04120489",
        "n04152593",
        "n04208210",
        "n04258138",
        "n04522168",
        "n04554684",
        "n12998815",
    ],
    "C": [
        "n01580077",
        "n01592084",
        "n01632777",
        "n01775062",
        "n01818515",
        "n02097130",
        "n02097298",
        "n02098413",
        "n02111500",
        "n02115913",
        "n02117135",
        "n02398521",
        "n02480855",
        "n02817516",
        "n02843684",
        "n02877765",
        "n02966193",
        "n03095699",
        "n03394916",
        "n03424325",
        "n03710193",
        "n03733131",
        "n03785016",
        "n03995372",
        "n04090263",
        "n04120489",
        "n04326547",
        "n04522168",
        "n07697537",
        "n07716906",
    ],
}


class DiscoverDataset:
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def __len__(self):
        return max([len(self.labeled_dataset), len(self.unlabeled_dataset)])

    def __getitem__(self, index):
        labeled_index = index % len(self.labeled_dataset)
        labeled_data = self.labeled_dataset[labeled_index]
        unlabeled_index = index % len(self.unlabeled_dataset)
        unlabeled_data = self.unlabeled_dataset[unlabeled_index]
        return (*labeled_data, *unlabeled_data)


class DiscoverImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.imagenet_split = args.imagenet_split
        self.dataset_class = torchvision.datasets.ImageFolder
        self.transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
        )
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        # train dataset
        train_dataset = self.dataset_class(train_data_dir, transform=self.transform_train)

        # split classes
        mapping = {c[:9]: i for c, i in train_dataset.class_to_idx.items()}
        labeled_classes = list(set(mapping.keys()) - set(IMAGENET_CLASSES_118))
        labeled_classes.sort()
        labeled_class_idxs = [mapping[c] for c in labeled_classes]
        unlabeled_classes = IMAGENET_CLASSES_30[self.imagenet_split]
        unlabeled_classes.sort()
        unlabeled_class_idxs = [mapping[c] for c in unlabeled_classes]

        # target transform
        all_classes = labeled_classes + unlabeled_classes
        target_transform = DiscoverTargetTransform(
            {mapping[c]: i for i, c in enumerate(all_classes)}
        )
        train_dataset.target_transform = target_transform

        # train set
        targets = np.array([img[1] for img in train_dataset.imgs])
        labeled_idxs = np.where(np.isin(targets, np.array(labeled_class_idxs)))[0]
        labeled_subset = torch.utils.data.Subset(train_dataset, labeled_idxs)
        unlabeled_idxs = np.where(np.isin(targets, np.array(unlabeled_class_idxs)))[0]
        unlabeled_subset = torch.utils.data.Subset(train_dataset, unlabeled_idxs)
        self.train_dataset = DiscoverDataset(labeled_subset, unlabeled_subset)

        # val datasets
        val_dataset_train = self.dataset_class(
            train_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        val_dataset_test = self.dataset_class(
            val_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        targets_train = np.array([img[1] for img in val_dataset_train.imgs])
        targets_test = np.array([img[1] for img in val_dataset_test.imgs])
        # unlabeled classes, train set
        unlabeled_idxs = np.where(np.isin(targets_train, np.array(unlabeled_class_idxs)))[0]
        unlabeled_subset_train = torch.utils.data.Subset(val_dataset_train, unlabeled_idxs)
        # unlabeled classes, test set
        unlabeled_idxs = np.where(np.isin(targets_test, np.array(unlabeled_class_idxs)))[0]
        unlabeled_subset_test = torch.utils.data.Subset(val_dataset_test, unlabeled_idxs)
        # labeled classes, test set
        labeled_idxs = np.where(np.isin(targets_test, np.array(labeled_class_idxs)))[0]
        labeled_subset_test = torch.utils.data.Subset(val_dataset_test, labeled_idxs)

        self.val_datasets = [unlabeled_subset_train, unlabeled_subset_test, labeled_subset_test]

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // 2,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for dataset in self.val_datasets
        ]


class PretrainImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_class = torchvision.datasets.ImageFolder
        self.transform_train = get_transforms("unsupervised", args.dataset)
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        # train dataset
        train_dataset = self.dataset_class(train_data_dir, transform=self.transform_train)

        # find labeled classes
        mapping = {c[:9]: i for c, i in train_dataset.class_to_idx.items()}
        labeled_classes = list(set(mapping.keys()) - set(IMAGENET_CLASSES_118))
        labeled_classes.sort()
        labeled_class_idxs = [mapping[c] for c in labeled_classes]

        # target transform
        target_transform = DiscoverTargetTransform(
            {mapping[c]: i for i, c in enumerate(labeled_classes)}
        )
        train_dataset.target_transform = target_transform

        # train set
        targets = np.array([img[1] for img in train_dataset.imgs])
        labeled_idxs = np.where(np.isin(targets, np.array(labeled_class_idxs)))[0]
        self.train_dataset = torch.utils.data.Subset(train_dataset, labeled_idxs)

        # val datasets
        val_dataset = self.dataset_class(
            val_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        targets = np.array([img[1] for img in val_dataset.imgs])
        # labeled classes, test set
        labeled_idxs = np.where(np.isin(targets, np.array(labeled_class_idxs)))[0]
        self.val_dataset = torch.utils.data.Subset(val_dataset, labeled_idxs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
