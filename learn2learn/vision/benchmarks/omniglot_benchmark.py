#!/usr/bin/env python3

import random
import learn2learn as l2l

from torchvision import transforms
from PIL.Image import LANCZOS


def omniglot_tasksets(
    train_ways,
    train_samples,
    test_ways,
    test_samples,
    root,
    scope,
    **kwargs
):
    """
    Benchmark definition for Omniglot.
    """
    data_transforms = transforms.Compose([
        transforms.Resize(28, interpolation=LANCZOS),
        transforms.ToTensor(),
        lambda x: 1.0 - x,
    ])
    omniglot = l2l.vision.datasets.FullOmniglot(
        root=root,
        transform=data_transforms,
        download=True,
    )
    dataset = l2l.data.MetaDataset(omniglot)
    train_dataset = dataset
    validation_datatset = dataset
    test_dataset = dataset
    ##만약에 train/test split(at disjoint)에 문제가 발생했다면 여기서 발생했을 것이다.
    #여기서 고정시켜주는 게 필요할듯
    classes = list(range(1623))
    #print("셔플 전에",classes[:100])
    random.shuffle(classes)
    #print("셔플 후에", classes[:100])
    print("Scope [",scope,"] ")
    train_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=train_ways,
                                             k=train_samples,
                                             filter_labels=classes[:int(1100*scope)]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    validation_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=test_ways,
                                             k=test_samples,
                                             filter_labels=classes[1100:1200]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    test_labels = classes[1200:]
    test_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=test_ways,
                                             k=test_samples,
                                             filter_labels=classes[1200:]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]

    _datasets = (train_dataset, validation_datatset, test_dataset)
    _transforms = (train_transforms, validation_transforms, test_transforms)
    return _datasets, _transforms,classes,test_labels
