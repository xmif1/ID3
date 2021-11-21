import pandas as pd
import numpy as np


def load_dataset(data_file, header_file, train_frac=0.7):
    dataset = pd.read_csv(data_file)

    with open(header_file, "r") as header_file_stream:
        dataset.columns = header_file_stream.readline().split(",")

    train = dataset.sample(frac=train_frac, random_state=0)  # remove random_state once debugging is over
    test = dataset.drop(train.index)

    return train, test


def entropy(dataset, target):
    counts = dataset[target].value_counts()
    counts = counts.to_numpy(dtype=np.int)

    probabilities = np.divide(counts, dataset[target].shape[0])

    return -1 * np.sum([p * log2_p for p, log2_p in zip(probabilities, np.log2(probabilities))])


def information_gain(dataset, dataset_entropy, target, attribute):
    conditional_entropy = 0
    for value in dataset[attribute].unique():
        subset = dataset.loc[dataset[attribute] == value]
        conditional_entropy = conditional_entropy + (subset.shape[0] * entropy(subset, target))

    conditional_entropy = conditional_entropy / dataset[target].shape[0]

    return dataset_entropy - conditional_entropy


def best_attribute(dataset, target):
    dataset_entropy = entropy(dataset, target)
    max_information_gain = 0
    max_ig_attr = None

    for attr in dataset.header:
        ig = information_gain(dataset, dataset_entropy, target, attr)
        if max_information_gain <= ig:
            max_information_gain = ig
            max_ig_attr = attr

    return max_ig_attr
