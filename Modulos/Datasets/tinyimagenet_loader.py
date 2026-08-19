"""
    Dataloader builder for open set recognition experiments on Tiny ImageNet.

    Train images are drawn from the `train` split (500 images per known class).
    Validation and test images are drawn from the `val` split, which holds 50
    images per class: each known class contributes 25 images to validation and
    25 to test (disjoint), each unknown class contributes 2 images to
    validation and 2 to test (also disjoint).

    Every transform (train, val and test) normalizes colours using the mean
    and std of the *known-class training images* of the corresponding split,
    since different splits have different known classes.
"""

import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class RemappedSubset(Dataset):
    """
        Wraps an ImageFolder dataset (built with transform=None, so it hands
        back raw PIL images), restricting it to a fixed list of sample
        indices, applying a given transform and remapping original class
        labels to known-class labels. Any class not present in label_map
        (i.e. an unknown class) is assigned unknown_label.
    """

    def __init__(self, dataset, indices, label_map, transform, unknown_label=-1):
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map
        self.transform = transform
        self.unknown_label = unknown_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, target = self.dataset[self.indices[i]]
        image = self.transform(image)
        target = self.label_map.get(target, self.unknown_label)
        return image, target


class TinyImageNet_loader:

    NUM_VAL_KNOWN = 25
    NUM_TEST_KNOWN = 25
    NUM_VAL_UNKNOWN = 2
    NUM_TEST_UNKNOWN = 2

    def __init__(self, data_dir="datasets/data/tiny-imagenet-200",
                 splits_dir="datasets/TinyImageNet/data/class_splits",
                 batch_size=32, shuffle=True, num_workers=4,
                 image_size=64, seed=42, unknown_label=-1):
        self.data_dir = data_dir
        self.splits_dir = splits_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.image_size = image_size
        self.seed = seed
        self.unknown_label = unknown_label

        self._train_base = None
        self._val_base = None
        self._norm_stats_cache = {}
        self._transform_cache = {}

    # ---- underlying ImageFolders (built once, cached, no transform applied) ----
    def _get_train_base(self):
        if self._train_base is None:
            self._train_base = datasets.ImageFolder(os.path.join(self.data_dir, 'train'))
        return self._train_base

    def _get_val_base(self):
        if self._val_base is None:
            self._val_base = datasets.ImageFolder(os.path.join(self.data_dir, 'val'))
        return self._val_base

    def get_class_splits(self, split_num):
        path = os.path.join(self.splits_dir, "{}.json".format(split_num))
        with open(path) as f:
            class_splits = json.load(f)
        return class_splits['Known'], class_splits['Unknown']

    @staticmethod
    def _target_map(known_classes):
        return {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted(known_classes))}

    @staticmethod
    def _indices_by_class(dataset):
        by_class = {}
        for idx, (_, target) in enumerate(dataset.samples):
            by_class.setdefault(target, []).append(idx)
        return by_class

    def _val_test_split(self, dataset, classes, n_val, n_test):
        """
            Deterministically splits each class's val-folder images into a
            val slice and a disjoint test slice, using a fixed seed so the
            split is stable across calls.
        """
        by_class = self._indices_by_class(dataset)
        rng = random.Random(self.seed)
        val_idxs, test_idxs = [], []
        for cls in classes:
            idxs = list(by_class.get(cls, []))
            rng.shuffle(idxs)
            val_idxs += idxs[:n_val]
            test_idxs += idxs[n_val:n_val + n_test]
        return val_idxs, test_idxs

    def _known_train_indices(self, split_num):
        known, _ = self.get_class_splits(split_num)
        dataset = self._get_train_base()
        by_class = self._indices_by_class(dataset)
        indices = []
        for cls in known:
            indices += by_class.get(cls, [])
        return dataset, known, indices

    def _eval_indices(self, split_num):
        known, unknown = self.get_class_splits(split_num)
        dataset = self._get_val_base()
        val_known, test_known = self._val_test_split(
            dataset, known, self.NUM_VAL_KNOWN, self.NUM_TEST_KNOWN)
        val_unknown, test_unknown = self._val_test_split(
            dataset, unknown, self.NUM_VAL_UNKNOWN, self.NUM_TEST_UNKNOWN)
        return dataset, known, val_known, test_known, val_unknown, test_unknown

    def _compute_norm_stats(self, split_num):
        """
            Computes per-channel mean/std over the known-class training
            images of this split (no augmentation, just resize + ToTensor).
        """
        if split_num not in self._norm_stats_cache:
            dataset, known, indices = self._known_train_indices(split_num)
            label_map = self._target_map(known)
            stats_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])
            subset = RemappedSubset(dataset, indices, label_map, stats_transform, self.unknown_label)
            loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=self.num_workers)

            channel_sum = torch.zeros(3)
            channel_sqsum = torch.zeros(3)
            n_pixels = 0
            for images, _ in loader:
                channel_sum += images.sum(dim=(0, 2, 3))
                channel_sqsum += (images ** 2).sum(dim=(0, 2, 3))
                n_pixels += images.size(0) * images.size(2) * images.size(3)

            mean = channel_sum / n_pixels
            var = channel_sqsum / n_pixels - mean ** 2
            std = torch.sqrt(var.clamp(min=1e-8))
            self._norm_stats_cache[split_num] = (mean.tolist(), std.tolist())
        return self._norm_stats_cache[split_num]

    def _get_transforms(self, split_num):
        """
            Returns (train_transform, eval_transform) for this split, both
            normalized with the mean/std of that split's known-class
            training images.
        """
        if split_num not in self._transform_cache:
            mean, std = self._compute_norm_stats(split_num)
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            eval_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            self._transform_cache[split_num] = (train_transform, eval_transform)
        return self._transform_cache[split_num]

    def _make_loader(self, dataset, indices, label_map, transform, shuffle):
        subset = RemappedSubset(dataset, indices, label_map, transform, self.unknown_label)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=shuffle,
                           num_workers=self.num_workers)

    # ---- public API ----
    def get_train_loader(self, split_num):
        dataset, known, indices = self._known_train_indices(split_num)
        train_transform, _ = self._get_transforms(split_num)
        label_map = self._target_map(known)
        return self._make_loader(dataset, indices, label_map, train_transform, self.shuffle)

    def get_val_loader(self, split_num):
        dataset, known, val_known, _, val_unknown, _ = self._eval_indices(split_num)
        _, eval_transform = self._get_transforms(split_num)
        label_map = self._target_map(known)
        return self._make_loader(dataset, val_known + val_unknown, label_map, eval_transform, False)

    def get_test_loader(self, split_num):
        dataset, known, _, test_known, _, test_unknown = self._eval_indices(split_num)
        _, eval_transform = self._get_transforms(split_num)
        label_map = self._target_map(known)
        return self._make_loader(dataset, test_known + test_unknown, label_map, eval_transform, False)

    def get_val_known_loader(self, split_num):
        dataset, known, val_known, _, _, _ = self._eval_indices(split_num)
        _, eval_transform = self._get_transforms(split_num)
        label_map = self._target_map(known)
        return self._make_loader(dataset, val_known, label_map, eval_transform, False)

    def get_val_unknown_loader(self, split_num):
        dataset, known, _, _, val_unknown, _ = self._eval_indices(split_num)
        _, eval_transform = self._get_transforms(split_num)
        label_map = self._target_map(known)
        return self._make_loader(dataset, val_unknown, label_map, eval_transform, False)

    def get_test_known_loader(self, split_num):
        dataset, known, _, test_known, _, _ = self._eval_indices(split_num)
        _, eval_transform = self._get_transforms(split_num)
        label_map = self._target_map(known)
        return self._make_loader(dataset, test_known, label_map, eval_transform, False)

    def get_test_unknown_loader(self, split_num):
        dataset, known, _, _, _, test_unknown = self._eval_indices(split_num)
        _, eval_transform = self._get_transforms(split_num)
        label_map = self._target_map(known)
        return self._make_loader(dataset, test_unknown, label_map, eval_transform, False)