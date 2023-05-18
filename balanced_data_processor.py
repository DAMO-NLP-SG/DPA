from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
import numpy as np
import random

"""
Sample label balanced samples in the given dataset, return the balanced indices of this dataset
"""
def balanced_sampler(dataset: DatasetDict, num_samples: int, label_list: list) -> list:
    labels = np.array(dataset["label"])
    indices = list()
    for i in range(len(label_list)):
        label_index = random.sample(list(np.where(labels==i)[0]), num_samples)
        indices.extend(label_index)
    random.shuffle(indices)
    return indices



if __name__ == '__main__':
    train_dataset = load_dataset('xnli', 'en', split='train', cache_dir='xnli-data')
    indices = balanced_sampler(train_dataset, 128, ["e", "c", "n"])
    labels = np.array(train_dataset["label"])
    # labels = np.array(labels)
    # print(labels.shape)
    # label_index = random.sample(list(np.where(labels==1)[0]), 100)
    # this should be all ones
    assert((labels[indices]==2).sum() == 128)
    assert((labels[indices]==1).sum() == 128)
    assert((labels[indices]==0).sum() == 128)
