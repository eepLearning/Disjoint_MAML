#!/usr/bin/env python3

"""
The benchmark modules provides a convenient interface to standardized benchmarks in the literature.
It provides train/validation/test TaskDatasets and TaskTransforms for pre-defined datasets.

This utility is useful for researchers to compare new algorithms against existing benchmarks.
For a more fine-grained control over tasks and data, we recommend directly using `l2l.data.TaskDataset` and `l2l.data.TaskTransforms`.
"""

import os
import learn2learn as l2l

from collections import namedtuple
from .omniglot_benchmark import omniglot_tasksets
from .mini_imagenet_benchmark import mini_imagenet_tasksets
from .tiered_imagenet_benchmark import tiered_imagenet_tasksets
from .fc100_benchmark import fc100_tasksets
from .cifarfs_benchmark import cifarfs_tasksets

from torchvision import transforms
from PIL.Image import LANCZOS
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels, FilterLabels


__all__ = ['list_tasksets', 'get_tasksets']


BenchmarkTasksets = namedtuple('BenchmarkTasksets', ('train', 'validation', 'test'))

_TASKSETS = {
    'omniglot': omniglot_tasksets,
    'mini-imagenet': mini_imagenet_tasksets,
    'tiered-imagenet': tiered_imagenet_tasksets,
    'fc100': fc100_tasksets,
    'cifarfs': cifarfs_tasksets,
}


def list_tasksets():
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/benchmarks/)

    **Description**

    Returns a list of all available benchmarks.

    **Example**
    ~~~python
    for name in l2l.vision.benchmarks.list_tasksets():
        print(name)
        tasksets = l2l.vision.benchmarks.get_tasksets(name)
    ~~~
    """
    return _TASKSETS.keys()


def get_tasksets(
    name,
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    num_tasks=-1,
    evaluation_tasks = -1,
    is_disjoint = None,
    root='~/data',
    device=None,
    **kwargs,
):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/benchmarks/)

    **Description**

    Returns the tasksets for a particular benchmark, using literature standard data and task transformations.

    The returned object is a namedtuple with attributes `train`, `validation`, `test` which
    correspond to their respective TaskDatasets.
    See `examples/vision/maml_miniimagenet.py` for an example.

    **Arguments**

    * **name** (str) - The name of the benchmark. Full list in `list_tasksets()`.
    * **train_ways** (int, *optional*, default=5) - The number of classes per train tasks.
    * **train_samples** (int, *optional*, default=10) - The number of samples per train tasks.
    * **test_ways** (int, *optional*, default=5) - The number of classes per test tasks. Also used for validation tasks.
    * **test_samples** (int, *optional*, default=10) - The number of samples per test tasks. Also used for validation tasks.
    * **num_tasks** (int, *optional*, default=-1) - The number of tasks in each TaskDataset.
    * **root** (str, *optional*, default='~/data') - Where the data is stored.

    **Example**
    ~~~python
    train_tasks, validation_tasks, test_tasks = l2l.vision.benchmarks.get_tasksets('omniglot')
    batch = train_tasks.sample()

    or:

    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot')
    batch = tasksets.train.sample()
    ~~~
    """
    root = os.path.expanduser(root)

    if device is not None:
        raise NotImplementedError('Device other than None not implemented. (yet)')

    # Load task-specific data and transforms
    datasets, transforms,classes,test_labels = _TASKSETS[name](train_ways=train_ways,
                                           train_samples=train_samples,
                                           test_ways=test_ways,
                                           test_samples=test_samples,
                                           root=root,
                                           **kwargs)
    train_dataset, validation_dataset, test_dataset = datasets
    train_transforms, validation_transforms, test_transforms = transforms

    # Instantiate the tasksets
    train_tasks = l2l.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = l2l.data.TaskDataset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=evaluation_tasks ,
    )
    test_tasks = l2l.data.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=evaluation_tasks ,
    )
    if is_disjoint == None:
        return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks),None
    else:
        if name == "omniglot":
           dataset = train_dataset
           print("disjount client sampling>>>>>>>")
           client = []
           for pool in is_disjoint:
              filter_labels = [classes[i] for i in pool]
              #filter_labels = pool
              #train(disjoint)/test split check
              intersection = len(list(set(filter_labels) & set(test_labels)))
              if intersection == 0:
                 print("Train(Disjoint)/Test Split Perpect!")
              else:
                 raise ('Invalid data_augmentation argument.')
              
              custom_transforms = [
                  l2l.data.transforms.FusedNWaysKShots(dataset,
                                                       n=train_ways,
                                                       k=train_samples,
                                                       filter_labels=filter_labels ),
                  l2l.data.transforms.LoadData(dataset),
                  l2l.data.transforms.RemapLabels(dataset),
                  l2l.data.transforms.ConsecutiveLabels(dataset),
                  l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
               ]
              custom_tasks = l2l.data.TaskDataset(
                  dataset=train_dataset,
                  task_transforms=custom_transforms,
                  num_tasks=num_tasks,
               )
              client.append(custom_tasks)
           print("Done...!: ", len(client),"Clients Disjoint Complete")
           print("Disjoint information: ")
           for idx,pool in enumerate(is_disjoint):
               print("Client ",idx," class_index: [",pool[0]," , ",pool[-1]," ]")
           return  BenchmarkTasksets(train_tasks, validation_tasks, test_tasks),client
       
        elif name == "mini-imagenet" :
           
           print("disjount client sampling>>>>>>>")
           client = []
           for pool in is_disjoint:
              #print("pool: ",)
              filter_labels = [classes[i] for i in pool]
              #print("filter_labels :",filter_labels)
              #print("test_labels :", test_labels)
              # filter_labels = pool
              # train(disjoint)/test split check
              intersection = len(list(set(filter_labels) & set(test_labels)))
              #print("interection",intersection)
              if intersection == 0:
                 print("Train(Disjoint)/Test Split Perpect!")
              else:
                 raise ('Invalid data_data split argument.')
              dataset = l2l.data.FilteredMetaDataset(train_dataset, filter_labels)
              #이 부분을 train_dataset에서 dataset으로 바꿨는데
              #어떻게 될려나?

              custom_transforms = [
                 NWays(dataset, train_ways),
                 KShots(dataset, train_samples),
                 LoadData(dataset),
                 RemapLabels(dataset),
                 ConsecutiveLabels(dataset),
              ]
              custom_tasks = l2l.data.TaskDataset(
                 dataset=dataset,
                 task_transforms=custom_transforms,
                 num_tasks=num_tasks,
              )
              client.append(custom_tasks)
           print("Done...!: ", len(client), "Clients Disjoint Complete")
           print("Disjoint information: ")
           for idx, pool in enumerate(is_disjoint):
              print("Client ", idx, " class_index: [", pool[0], " , ", pool[-1], " ]")
           return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks), client
           
           
           
       
       

def get_custom_tasksets(
    name,custom_filter,
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    root='~/data',
    device=None,
    **kwargs,
):
 
    root = os.path.expanduser(root)

    if device is not None:
        raise NotImplementedError('Device other than None not implemented. (yet)')
    datasets, transforms = _TASKSETS[name](train_ways=train_ways,
                                           train_samples=train_samples,
                                           test_ways=test_ways,
                                           test_samples=test_samples,
                                           root=root,
                                           **kwargs)
    custom_datasets = datsets[0]

    custom_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=train_ways,
                                             k=train_samples,
                                             filter_labels=custom_filter),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    
    custom_tasks = l2l.data.TaskDataset(
        dataset=custom_datasets,
        task_transforms=custom_transforms,
        num_tasks=num_tasks,
    )


    return custom_tasks
