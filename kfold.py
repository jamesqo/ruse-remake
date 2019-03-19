from collections import defaultdict
import math

def get_groups(dataset, grouping):
    groups = defaultdict(list)
    for instance in dataset:
        key = grouping(instance)
        groups[key].append(instance)
    return groups

def partition_groups(groups, k):
    partitions = dict()
    for key, group in groups.items():
        partitions[key] = split_into_k(group, k)
    return partitions

def split_into_k(group, k):
    splits = []
    splitlen = math.floor(len(group) / k)
    for n in range(0, k - 1):
        start, stop = n*splitlen, (n+1)*splitlen
        splits.append(group[start:stop])
    # Unless k evenly divides len(group), the last split will be larger than the rest
    splits.append(group[(k-1)*splitlen:])
    return splits

def get_folds(partitions, k):
    folds = []
    for i in range(0, k):
        # Add 1 split from each class partition.
        splits = [partition[i] for partition in partitions]
        folds.append([inst for split in splits for inst in split]) # Flattens 'splits' into a single list
    return folds

class StratifiedKFold(object):
    def __init__(self, dataset, k, grouping):
        groups = get_groups(dataset, grouping)
        partitions = partition_groups(groups, k)
        self.folds = get_folds(partitions, k)
        self.num_folds = k
        self.excl_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.excl_index == self.num_folds:
            raise StopIteration("Already yielded K train/test splits.")

        test = self.folds[self.excl_index]
        train = self.folds.copy()
        del train[self.excl_index]

        self.excl_index += 1
        return train, test
